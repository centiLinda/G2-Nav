#!/path_to_your_env/bin/python

import rospy
import os
import json
import threading
import time
import re
import base64
import cv2
import numpy as np
import math
from collections import deque
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage
from nav_msgs.msg import Odometry
from diagnostic_msgs.msg import DiagnosticArray
from openai import OpenAI

class VLMAnalyzerLocalNode:
    def __init__(self):
        rospy.init_node('vlm_analyzer_local_node', anonymous=True)
        
        # --- Local VLM Configuration ---
        # Point to the local OpenAI-compatible server run by vLLM
        self.base_url = rospy.get_param('~base_url', 'http://localhost:8000/v1')
        self.api_key = rospy.get_param('~api_key', 'EMPTY') # vLLM default
        self.model_name = rospy.get_param('~vlm_model', 'Qwen/Qwen3.5-2B')
        
        rospy.loginfo(f"Connecting to local VLM '{self.model_name}' at '{self.base_url}'")
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        
        # --- Node State ---
        self.image_buffer = deque(maxlen=3)
        self.last_img_time = 0
        self.latest_request = None
        self.last_processed_req_ts = 0.0  # Keep track of the last processed request to drop historical queries
        self.is_processing = False
        self.query_sent_time = 0.0
        self.new_response_ready = False
        self.last_response_text = ""
        self.last_response_ts = 0.0
        self.last_num_input = 0
        self.last_num_output = 0
        self.last_input_objects = []
        self.last_generation_time = 0.0
        
        self.region_list = []
        self.traversable_region = ""
        self.region_pub = rospy.Publisher('/traversable_region', String, queue_size=1, latch=True)

        # --- Odometry State ---
        self.latest_yaw = 0.0
        self.latest_robot_x = 0.0
        self.latest_robot_y = 0.0
        rospy.Subscriber('/odom', Odometry, self.odom_cb, queue_size=1)

        # --- ROS Communication ---
        rospy.Subscriber('/image_raw/compressed', CompressedImage, self.image_cb, queue_size=1, buff_size=2**24)
        rospy.Subscriber('/tracked_obj', DiagnosticArray, self.tracked_obj_cb, queue_size=1)
        
        # Publish candidate objects and their scores to `/vlm_obj`
        self.score_pub = rospy.Publisher('/vlm_obj', String, queue_size=1)
        self.status_pub = rospy.Publisher('/vlm_status', String, queue_size=1)
        
        rospy.Timer(rospy.Duration(0.5), self.process_vlm)
        rospy.loginfo("VLM Local Analyzer Node Initialized.")

    def odom_cb(self, msg):
        self.latest_robot_x = msg.pose.pose.position.x
        self.latest_robot_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y**2 + q.z**2)
        self.latest_yaw = math.atan2(siny_cosp, cosy_cosp)

    def image_cb(self, msg):
        current_time = msg.header.stamp.to_sec()
        if current_time - self.last_img_time > 1.0: # Downsample without tied to hz
            np_arr = np.frombuffer(msg.data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            h, w = img.shape[:2]
            scale = min(512/w, 512/h) 
            # Only resize if image is larger than target to avoid upscaling
            if scale < 1.0:
                new_w, new_h = int(w * scale), int(h * scale)
                img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                img_resized = img
            
            _, buffer = cv2.imencode('.jpg', img_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            b64_img = base64.b64encode(buffer).decode('utf-8')
            
            self.image_buffer.append(b64_img)
            self.last_img_time = current_time

    def tracked_obj_cb(self, msg):
        if not msg.status:
            return
            
        timestamp = msg.header.stamp.to_sec()
        try:
            tracked_objects = json.loads(msg.status[0].message)
        except Exception as e:
            rospy.logwarn(f"Failed to parse tracked objects: {e}")
            return
            
        request_data = {"timestamp": timestamp, "objects": []}
        for obj in tracked_objects:
            # Filter objects that have a valid bounding box attached
            if 'bbox' in obj and obj['bbox'] is not None:
                request_data["objects"].append(obj)
                
        # Only update the latest request if there are actual valid objects to analyze
        if len(request_data["objects"]) > 0:
            self.latest_request = request_data

    def process_vlm(self, event):
        # Populate dashboard metrics
        status_dict = {
            "received_query": self.latest_request is not None,
            "req_ts": self.latest_request.get('timestamp', 0.0) if self.latest_request else 0.0,
            "is_processing": self.is_processing,
            "wait_time": time.time() - self.query_sent_time if self.is_processing else 0.0,
            "new_response_ready": self.new_response_ready,
            "response_text": self.last_response_text,
            "response_ts": self.last_response_ts,
            "num_input": self.last_num_input,
            "num_output": self.last_num_output,
            "input_objects": self.last_input_objects,
            "generation_time": self.last_generation_time,
            "traversable_region": self.traversable_region,
            "region_list": self.region_list
        }
        self.status_pub.publish(String(data=json.dumps(status_dict)))
        
        if self.new_response_ready and not self.is_processing:
            self.new_response_ready = False
            
        req_ts = self.latest_request.get('timestamp', 0.0) if self.latest_request else 0.0
            
        # Wait until we have a query, 3 frames, and are not currently processing one
        if self.is_processing or self.latest_request is None or len(self.image_buffer) < 3: 
            return
            
        # Do not process historic, outdated requests
        if req_ts <= self.last_processed_req_ts:
            return 
            
        self.is_processing = True
        self.last_processed_req_ts = req_ts
        self.query_sent_time = time.time()
        request_data = self.latest_request
        images = list(self.image_buffer)
        
        threading.Thread(target=self.call_vlm, args=(images, request_data)).start()

    def call_vlm(self, images, request_data):
        try:
            self.last_num_input = len(request_data.get('objects', []))
            
            # --- 1. Prepare Prompt ---
            lines = []
            dashboard_input_objs = []
            for obj in request_data['objects']:
                label = obj.get('label', '').lower()
                if label in self.region_list or label == self.traversable_region:
                    continue

                x1, y1, x2, y2 = obj['bbox']
                vx = obj.get('vx', 0.0)
                vy = obj.get('vy', 0.0)
                
                speed = math.hypot(vx, vy)
                if speed < 0.2:
                    motion_str = f"static (speed={speed:.1f}m/s)"
                else:
                    # Rotate global velocity to robot's local frame
                    yaw = self.latest_yaw
                    local_vx = vx * math.cos(yaw) + vy * math.sin(yaw)
                    local_vy = -vx * math.sin(yaw) + vy * math.cos(yaw)
                    
                    angle_deg = math.degrees(math.atan2(local_vy, local_vx))
                    clock_dir = round(12 - angle_deg / 30) % 12
                    if clock_dir == 0:
                        clock_dir = 12
                        
                    # Add explicit semantic hints for small language models
                    if clock_dir in [11, 12, 1]: hint = "moving away"
                    elif clock_dir in [2, 3, 4]: hint = "moving right"
                    elif clock_dir in [5, 6, 7]: hint = "approaching"
                    else: hint = "moving left"
                        
                    motion_str = f"speed={speed:.1f}m/s, direction={clock_dir} o'clock ({hint})"
                
                dist = math.hypot(obj.get('x', 0.0) - self.latest_robot_x, obj.get('y', 0.0) - self.latest_robot_y)
                lines.append(f"- {obj['label']}_{obj['id']} {obj['score']:.2f}: bbox=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}), tracking_distance: {dist:.1f}m, tracking_motion: {motion_str}")
                dashboard_input_objs.append(f"{obj['label']}_{obj['id']} | dist={dist:.1f}m | {motion_str}")
            
            objects_text = "\n".join(lines)
            self.last_input_objects = dashboard_input_objs
            
            prompt_path = os.path.join(os.path.dirname(__file__), 'prompt.txt')
            with open(prompt_path, 'r') as f:
                prompt = f.read()
            prompt = prompt.replace('{objects_text}', objects_text)
            prompt = prompt.replace('{region_list}', str(self.region_list))
            prompt = prompt.replace('{traversable_region}', self.traversable_region)

            # --- 2. Construct messages ---
            content_list = []
            for i, b64 in enumerate(images):
                content_list.append({"type": "text", "text": f"--- Frame {i+1} ---"})
                content_list.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
            content_list.append({"type": "text", "text": prompt})
            
            # --- 3. Call Local VLM with recommended parameters ---
            start_gen_time = time.time()
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": content_list}],
                max_tokens=4096,
                temperature=0.6,
                top_p=0.95,
                presence_penalty=0.0, 
                extra_body={
                    "top_k": 20,
                    "min_p": 0.0,
                    "repetition_penalty": 1.0,
                    "enable_thinking": True, 
                },
            )
            self.last_generation_time = time.time() - start_gen_time
            
            response_text = response.choices[0].message.content
            
            response_text = re.sub(r'<think>.*?(</think>|$)', '', response_text, flags=re.DOTALL).strip()

            self.last_response_text = response_text
            self.last_response_ts = request_data.get('timestamp', 0.0)
            self.new_response_ready = True
            
            # --- 4. Parse Response ---
            trav_match = re.search(r'TRAVERSABLE_REGION:\s*([^\n]+)', response_text)
            if trav_match:
                new_trav = re.sub(r'_\d+', '', trav_match.group(1).strip().strip("'\"").lower())
                new_trav = re.sub(r'[^a-z0-9\s\-]', '', new_trav).strip('- ')
                new_trav = re.sub(r'\s+', ' ', new_trav)
                
                if new_trav and new_trav not in ["none", "null", "unknown"]:
                    if "label" not in new_trav and "traversable" not in new_trav and len(new_trav.split()) <= 3:
                        if self.traversable_region and new_trav != self.traversable_region:
                            if self.traversable_region not in self.region_list:
                                self.region_list.append(self.traversable_region)
                        self.traversable_region = new_trav
                        if self.traversable_region in self.region_list:
                            self.region_list.remove(self.traversable_region)
                        self.region_pub.publish(String(data=self.traversable_region))
                    
            new_reg_match = re.search(r'NEW_NON_TRAVERSABLE_REGIONS:\s*\[(.*?)\]', response_text)
            if new_reg_match:
                raw_regs = [re.sub(r'_\d+', '', r.strip().strip("'\"").lower()) for r in new_reg_match.group(1).split(',')]
                new_regs = [re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9\s\-]', '', r).strip('- ')) for r in raw_regs]
                for r in new_regs:
                    if not r or r in ["none", "null", "unknown"]:
                        continue
                    # Block prompt hallucination leakage and overly long sentences
                    if "label" in r or "traversable" in r or len(r.split()) > 3:
                        continue
                    if r != self.traversable_region and r not in self.region_list:
                        self.region_list.append(r)

            scores = {}
            # Programmatic safeguard: only allow IDs that were actually in the request
            valid_ids = {str(obj['id']) for obj in request_data.get('objects', [])}
            
            # Parse response block by block for robust multi-field extraction
            block_pattern = r'(?:^|\n)\s*[-*]*\s*(?:(?:label_id|id|label|object)[^\w\n]*)?([a-zA-Z0-9_\- \t]+)_(\d+)[:\s]*\n?([\s\S]*?)(?=(?:^|\n)\s*[-*]*\s*(?:(?:label_id|id|label|object)[^\w\n]*)?[a-zA-Z0-9_\- \t]+_\d+[:\s]*\n?|$)'
            for match in re.finditer(block_pattern, response_text, re.IGNORECASE):
                tid_str = match.group(2)
                block = match.group(3)
                
                score_match = re.search(r'[Ss]core(?:[^:\n]*[:=])?[^\d\-]*(-1|[0-5](?:\.\d+)?)', block)
                valid_match = re.search(r'tracking_valid(?:[^:\n]*[:=])?\s*(yes|no)', block, re.IGNORECASE)
                depth_match = re.search(r'depth_valid(?:[^:\n]*[:=])?\s*(yes|no)', block, re.IGNORECASE)
                
                if score_match and tid_str in valid_ids:
                    score_val = float(score_match.group(1))
                    tracking_valid = False if (valid_match and valid_match.group(1).lower() == 'no') else True
                    depth_valid = False if (depth_match and depth_match.group(1).lower() == 'no') else True
                    
                    scores[tid_str] = {"score": score_val, "tracking_valid": tracking_valid, "depth_valid": depth_valid}
                    
            # Mark evaluated but unselected objects as 0.0 (ignored)
            for tid_str in valid_ids:
                if tid_str not in scores:
                    scores[tid_str] = {"score": 0.0, "tracking_valid": True, "depth_valid": True}

            self.last_num_output = sum(1 for v in scores.values() if v.get("score", 0.0) != 0.0)
            
            if scores:
                self.score_pub.publish(String(data=json.dumps(scores)))
        except Exception as e:
            rospy.logwarn(f"Local VLM Analysis Error: {e}")
            self.last_response_text = f"Error: {e}"
            self.new_response_ready = True
        finally:
            self.is_processing = False

if __name__ == '__main__':
    try:
        VLMAnalyzerLocalNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass