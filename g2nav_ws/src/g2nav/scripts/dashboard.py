#!/path_to_your_env/bin/python

import rospy
import json
import time
import tkinter as tk
from tkinter import scrolledtext
from diagnostic_msgs.msg import DiagnosticArray
from std_msgs.msg import String

class DashboardGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("System Dashboard")
        self.root.geometry("800x800")
        
        # Configure dark theme
        bg_color = "#1E1E1E"
        fg_color = "#D4D4D4"
        
        self.text_area = scrolledtext.ScrolledText(
            self.root, 
            wrap=tk.WORD, 
            font=("Monospace", 11), 
            bg=bg_color, 
            fg=fg_color,
            padx=10, 
            pady=10
        )
        self.text_area.pack(expand=True, fill='both')
        
        self.raw_text = "Raw Detections: Waiting for data..."
        self.track_text = "Tracked Objects: Waiting for data..."
        self.vlm_input_text = "VLM Input Objects: Waiting for data..."
        self.vlm_obj_text = "VLM Objects: Waiting for data..."
        self.final_obj_text = "Final Objects (Costmap Input): Waiting for data..."
        self.vlm_status_text = "VLM Status: Waiting for data..."
        self.region_text = "--- Region Information ---\nWaiting for data..."
        self.current_text = "Initializing..."
        
        self.track_labels = {}
        self.last_seen_response_ts = 0.0
        self.last_response_recv_time = 0.0
        
        # Used to detect if the rosbag is paused
        self.last_bag_msg_wall_time = time.time()
        
        # Initialize ROS
        rospy.init_node('dashboard_gui', anonymous=True)
        rospy.Subscriber('/tracked_obj', DiagnosticArray, self.tracked_obj_cb)
        rospy.Subscriber('/vlm_obj', String, self.vlm_obj_cb)
        rospy.Subscriber('/vlm_status', String, self.vlm_status_cb)
        rospy.Subscriber('/costmap_input_obj', String, self.costmap_obj_cb)
        
        self.update_gui()

    def tracked_obj_cb(self, msg):
        self.last_bag_msg_wall_time = time.time()

        try:
            for status in msg.status:
                if status.name == "TrackedObjects":
                    data = json.loads(status.message)
                    count = len(data)
                    lines = [f"--- Tracked: {count} ---"]
                    for item in data:
                        obj_id = str(item.get('id', 'unknown'))
                        label = item.get('label', 'unknown')
                        self.track_labels[obj_id] = label
                        lines.append(f"  - {label}_{obj_id}")
                    self.track_text = "\n".join(lines)
                elif status.name == "DetectionDebug":
                    data = json.loads(status.message)
                    count = len(data)
                    lines = [f"--- Detected: {count} ---"]
                    for item in data:
                        lines.append(f"  - {item.get('label', 'unknown')}_{item.get('index', 0)} | {item.get('reason', 'Unknown')}")
                    self.raw_text = "\n".join(lines)
        except Exception as e:
            self.track_text = f"Error parsing tracked objects: {e}"
            
        self.refresh_content()

    def vlm_obj_cb(self, msg):
        try:
            data = json.loads(msg.data)
            
            count = len(data)
            lines = [f"--- VLM Evaluated: {count} ---"]
            for obj_id, score_data in data.items():
                label = self.track_labels.get(str(obj_id), "unknown")
                if isinstance(score_data, dict):
                    score = score_data.get("score", 0.0)
                    valid = score_data.get("tracking_valid", True)
                    depth_valid = score_data.get("depth_valid", True)
                    valid_str = "OK" if valid else "MISMATCH"
                    depth_str = "OK" if depth_valid else "FALLBACK"
                    lines.append(f"  - {label}_{obj_id} | Score: {score} | Kinematic: {valid_str} | Depth: {depth_str}")
                else:
                    lines.append(f"  - {label}_{obj_id} | Score: {score_data}")
            self.vlm_obj_text = "\n".join(lines)
        except Exception as e:
            self.vlm_obj_text = f"Error parsing VLM objects: {e}"
            
        self.refresh_content()

    def costmap_obj_cb(self, msg):
        try:
            data = json.loads(msg.data)
            count = len(data)
            final_lines = []
            for item in data:
                valid_str = "OK" if item.get("tracking_valid", True) else "MISMATCH"
                depth_str = "OK" if item.get("depth_valid", True) else "FALLBACK"
                score_str = f"{item.get('score', 0.0)} (Default)" if item.get("is_default", False) else f"{item.get('score', 0.0)}"
                final_lines.append(f"  - {item.get('label', 'unknown')}_{item.get('id', 'unknown')} | Score: {score_str} | Kinematic: {valid_str} | Depth: {depth_str}")
                
            header = [f"--- Final Objects (Costmap Input): {count} ---"]
            self.final_obj_text = "\n".join(header + final_lines)
        except Exception as e:
            self.final_obj_text = f"Error parsing costmap objects: {e}"
            
        self.refresh_content()

    def vlm_status_cb(self, msg):
        try:
            data = json.loads(msg.data)
            req_ts = data.get('response_ts', 0.0)
            gen_time = data.get('generation_time', 0.0)
            response_text = data.get('response_text', '')
            
            if req_ts != self.last_seen_response_ts and req_ts != 0.0:
                self.last_seen_response_ts = req_ts
                self.last_response_recv_time = time.time() % 100000
                
            input_objs = data.get('input_objects', [])
            count_in = len(input_objs)
            lines_in = [f"--- VLM Input: {count_in} ---"]
            for obj in input_objs:
                lines_in.append(f"  - {obj}")
            self.vlm_input_text = "\n".join(lines_in)

            # Update Regions Text
            trav_reg = data.get('traversable_region', '')
            reg_list = data.get('region_list', [])
            self.region_text = f"--- Region Information ---\n  - Traversable Region: '{trav_reg}'\n  - Non-Traversable Regions: {reg_list}"

            lines = ["--- VLM Output Status ---"]
            lines.append(f"Based on frame (bag time): {req_ts:.3f} s")
            lines.append(f"Generation time:           {gen_time:.3f} s")
            if self.last_response_recv_time > 0.0:
                lines.append(f"Received at (sys time):    {self.last_response_recv_time:.3f}")
            else:
                lines.append(f"Received at (sys time):    Waiting...")
            lines.append("\n--- Raw VLM Output ---")
            lines.append(response_text if response_text else "Waiting for first response...")
            
            self.vlm_status_text = "\n".join(lines)
        except Exception as e:
            self.vlm_status_text = f"Error parsing VLM status: {e}"
            
        self.refresh_content()

    def refresh_content(self):
        # If no sensor-driven messages arrive for 0.3s, the rosbag is likely paused.
        # Freeze the dashboard output, but append a [PAUSED] indicator.
        if time.time() - self.last_bag_msg_wall_time > 0.3:
            if "=== System Dashboard ===" in self.current_text and "[PAUSED]" not in self.current_text:
                self.current_text = self.current_text.replace("=== System Dashboard ===", "=== System Dashboard === [PAUSED]")
            return
            
        self.current_text = f"=== System Dashboard ===\n\n{self.region_text}\n\n{self.final_obj_text}\n\n{self.vlm_status_text}\n\n{self.vlm_input_text}\n\n{self.raw_text}\n\n{self.track_text}\n\n{self.vlm_obj_text}\n"




    def update_gui(self):
        if not hasattr(self, 'displayed_text'):
            self.displayed_text = ""
            
        # Skip GUI update if nothing has changed
        if self.current_text == self.displayed_text:
            self.root.after(100, self.update_gui)
            return

        self.text_area.config(state=tk.NORMAL)
        
        # Full replacement for dynamic length content
        scroll_pos = self.text_area.yview()
        self.text_area.delete(1.0, tk.END)
        self.text_area.insert(tk.END, self.current_text)
        self.text_area.yview_moveto(scroll_pos[0])
            
        self.text_area.config(state=tk.DISABLED)
        self.displayed_text = self.current_text
        
        self.root.after(100, self.update_gui) # Check for updates at 10Hz

if __name__ == '__main__':
    root = tk.Tk()
    app = DashboardGUI(root)
    root.mainloop()