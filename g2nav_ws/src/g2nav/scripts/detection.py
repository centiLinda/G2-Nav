#!/path_to_your_env/bin/python

import os
import json

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import rospy
import cv2
import torch
import torchvision
import torchvision.transforms as TS
import numpy as np
import time
from PIL import Image
from torch import nn

from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus

# RAM++, GroundingDINO, SAM
from ram.models import ram_plus
from ram import inference_ram_openset
from ram.utils import build_openset_llm_label_embedding
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from segment_anything.build_sam import build_sam
from segment_anything.predictor import SamPredictor

# ======================================
# --- TUNABLE PARAMETERS: THRESHOLDS ---
# ======================================
BOX_THRESHOLD = 0.4   # Minimum confidence score for bounding boxes
TEXT_THRESHOLD = 0.4  # Minimum confidence score for text labels
IOU_THRESHOLD = 0.5   # Overlap threshold to remove redundant overlapping boxes (NMS)
MIN_BOX_AREA = 10.0    # Minimum area (in pixels) for a bounding box to be kept
PROXIMITY_THRESHOLD_FACTOR = 0.4 # Maximum pixel gap factor (relative to avg width) for grouping
HEIGHT_RATIO_THRESHOLD = 0.7     # Minimum height ratio for two bounding boxes to be grouped
# ==============================================================================

class DetectionOpensetNode:
    def __init__(self):
        rospy.init_node('detection_openset_node', anonymous=True)
        self.bridge = CvBridge()

        # --- Open-Set Model Parameters & Initialization ---
        self.ram_device = rospy.get_param('~ram_device', 'cuda:0')
        self.gd_device = rospy.get_param('~gd_device', 'cuda:0')
        self.sam_device = rospy.get_param('~sam_device', 'cuda:0')
        ram_checkpoint = rospy.get_param('~ram_checkpoint', '/path_to_your_ram_checkpoint.pth')
        grounded_checkpoint = rospy.get_param('~grounded_checkpoint', '/path_to_your_groundingdino_checkpoint.pth')
        sam_checkpoint = rospy.get_param('~sam_checkpoint', '/path_to_your_sam_checkpoint.pth')
        config_file = rospy.get_param('~config_file', '/path_to_your_groundingdino_config.py')
        llm_tag_des = rospy.get_param('~llm_tag_des', '/path_to_your_ram_tag.json')

        self.box_threshold = BOX_THRESHOLD
        self.text_threshold = TEXT_THRESHOLD
        self.iou_threshold = IOU_THRESHOLD

        self.traversable_region = ""
        rospy.Subscriber('/traversable_region', String, self.traversable_region_cb, queue_size=1)

        rospy.loginfo("Initializing RAM++ with open-set capability...")
        normalize = TS.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.ram_transform = TS.Compose([
            TS.Resize((384, 384)),
            TS.ToTensor(),
            normalize
        ])
        self.ram_model = ram_plus(pretrained=ram_checkpoint, image_size=384, vit='swin_l')
        self.ram_model.eval()
        self.ram_model = self.ram_model.to(self.ram_device)

        with open(llm_tag_des, 'r') as f:
            llm_tag_dict = json.load(f)

        openset_label_embedding, openset_categories = build_openset_llm_label_embedding(llm_tag_dict)
        self.ram_model.tag_list = np.array(openset_categories)
        self.ram_model.label_embed = nn.Parameter(openset_label_embedding.float().to(self.ram_device))
        self.ram_model.num_class = len(openset_categories)
        self.ram_model.class_threshold = torch.ones(self.ram_model.num_class, device=self.ram_device) * 0.4

        rospy.loginfo("Initializing Grounding DINO...")
        self.gd_model = self.load_gd_model(config_file, grounded_checkpoint, device=self.gd_device)
        self.gd_transform = T.Compose([
            T.RandomResize([512], max_size=768),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        
        rospy.loginfo("Initializing SAM...")
        # The build_sam helper function infers the model type from the checkpoint.
        self.sam_predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(self.sam_device))

        # --- Publishers ---
        self.det_pub = rospy.Publisher('/raw_det', DiagnosticArray, queue_size=1)
        self.img_pub = rospy.Publisher('/raw_det_annotated_img/compressed', CompressedImage, queue_size=1)
        self.seg_pub = rospy.Publisher('/traversability_seg_img/compressed', CompressedImage, queue_size=1)
        self.mask_pub = rospy.Publisher('/floor_mask/compressed', CompressedImage, queue_size=1)
        
        # --- Subscribers ---
        rospy.Subscriber('/image_raw/compressed', CompressedImage, self.image_callback, queue_size=1, buff_size=2**24)
        
        rospy.loginfo("Detection Node Ready.")

    def traversable_region_cb(self, msg):
        self.traversable_region = msg.data.strip()

    def load_gd_model(self, model_config_path, model_checkpoint_path, device):
        args = SLConfig.fromfile(model_config_path)
        args.device = device
        model = build_model(args)
        checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
        model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        model.eval()
        return model

    def get_grounding_output(self, model, image, caption, box_threshold, text_threshold, device="cpu"):
        caption = caption.lower().strip()
        if not caption.endswith("."):
            caption = caption + "."
        model = model.to(device)
        image = image.to(device)
        with torch.no_grad():
            outputs = model(image[None], captions=[caption])
            
        logits = outputs["pred_logits"].cpu().sigmoid()[0]
        boxes = outputs["pred_boxes"].cpu()[0]

        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]
        boxes_filt = boxes_filt[filt_mask]

        tokenlizer = model.tokenizer
        tokenized = tokenlizer(caption)
        pred_phrases = []
        scores = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            scores.append(logit.max().item())
        return boxes_filt, torch.Tensor(scores), pred_phrases

    def image_callback(self, img_msg):
        try:
            time_sec = img_msg.header.stamp.to_sec()
            frame = self.bridge.compressed_imgmsg_to_cv2(img_msg, "bgr8")
            
            # 1. Vision Detection (RAM++)
            image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            raw_image = self.ram_transform(image_pil).unsqueeze(0).to(self.ram_device)
            
            with torch.no_grad():
                tags = inference_ram_openset(raw_image, self.ram_model)
            
            results_list = []
            image_tensor, _ = self.gd_transform(image_pil, None)
            
            if tags and tags.strip() != "":
                formatted_tags = tags.replace(' | ', '. ').replace('|', '.').replace(',', '. ')
                
                # 2. Vision Localization (GroundingDINO)
                boxes_filt, scores, pred_phrases = self.get_grounding_output(
                    self.gd_model, image_tensor, formatted_tags, self.box_threshold, self.text_threshold, device=self.gd_device
                )

                if boxes_filt.size(0) > 0:
                    H, W = frame.shape[:2]
                    # Convert normalized [cx, cy, w, h] to absolute [xmin, ymin, xmax, ymax]
                    for i in range(boxes_filt.size(0)):
                        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
                        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
                        boxes_filt[i][2:] += boxes_filt[i][:2]

                    boxes_filt = boxes_filt.cpu()
                    scores = scores.cpu()

                    # NMS filter
                    nms_idx = torchvision.ops.nms(boxes_filt, scores, self.iou_threshold).numpy().tolist()
                    if len(nms_idx) > 0:
                        filtered_boxes = boxes_filt[nms_idx].numpy()
                        filtered_scores = scores[nms_idx].numpy()
                        filtered_phrases = [pred_phrases[i] for i in nms_idx]

                        # --- 1. Small Object Removal ---
                        valid_indices = []
                        for i, box in enumerate(filtered_boxes):
                            xmin, ymin, xmax, ymax = box
                            area = (xmax - xmin) * (ymax - ymin)
                            if area >= MIN_BOX_AREA:
                                valid_indices.append(i)
                                
                        filtered_boxes = filtered_boxes[valid_indices]
                        filtered_scores = filtered_scores[valid_indices]
                        filtered_phrases = [filtered_phrases[i] for i in valid_indices]

                        # --- 2. Human Grouping Logic ---
                        human_aliases = ['person', 'human', 'pedestrian', 'man', 'woman', 'boy', 'girl', 'people', 'guy']
                        human_indices = []
                        for idx, phrase in enumerate(filtered_phrases):
                            label = phrase.split('(')[0].strip().lower()
                            if any(alias in label for alias in human_aliases):
                                human_indices.append(idx)

                        if len(human_indices) > 1:
                            adj = {i: [] for i in human_indices}
                            for i in range(len(human_indices)):
                                for j in range(i + 1, len(human_indices)):
                                    idx1, idx2 = human_indices[i], human_indices[j]
                                    box1, box2 = filtered_boxes[idx1], filtered_boxes[idx2]
                                    
                                    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
                                    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
                                    
                                    avg_width, avg_height = (w1 + w2) / 2.0, (h1 + h2) / 2.0
                                    
                                    gap_x = max(0, box1[0] - box2[2], box2[0] - box1[2]) # left-right gap
                                    gap_y = max(0, box1[1] - box2[3], box2[1] - box1[3]) # top-bottom gap

                                    if gap_x < (avg_width * PROXIMITY_THRESHOLD_FACTOR) and gap_y < (avg_height * PROXIMITY_THRESHOLD_FACTOR):
                                        height_ratio = min(h1, h2) / max(h1, h2)
                                        if height_ratio > HEIGHT_RATIO_THRESHOLD:
                                            adj[idx1].append(idx2)
                                            adj[idx2].append(idx1)
                            
                            visited, groups = set(), []
                            for idx in human_indices:
                                if idx not in visited:
                                    comp, q = [], [idx]
                                    while q:
                                        curr = q.pop(0)
                                        if curr not in visited:
                                            visited.add(curr)
                                            comp.append(curr)
                                            q.extend(adj[curr])
                                    if len(comp) > 1:
                                        groups.append(comp)

                            if groups:
                                new_boxes, new_scores, new_phrases = [], [], []
                                grouped_indices = set([idx for comp in groups for idx in comp])
                                
                                # Append un-grouped original objects
                                for idx in range(len(filtered_boxes)):
                                    if idx not in grouped_indices:
                                        new_boxes.append(filtered_boxes[idx])
                                        new_scores.append(filtered_scores[idx])
                                        new_phrases.append(filtered_phrases[idx])
                                        
                                # Append the new unified group boxes
                                for comp in groups:
                                    group_boxes = filtered_boxes[comp]
                                    xmin, ymin = np.min(group_boxes[:, 0]), np.min(group_boxes[:, 1])
                                    xmax, ymax = np.max(group_boxes[:, 2]), np.max(group_boxes[:, 3])
                                    score = np.mean(filtered_scores[comp])
                                    
                                    new_boxes.append([xmin, ymin, xmax, ymax])
                                    new_scores.append(score)
                                    new_phrases.append(f"group({score:.2f})")
                                    
                                filtered_boxes = np.array(new_boxes)
                                filtered_scores = np.array(new_scores)
                                filtered_phrases = new_phrases

                        # --- 3. Extract final detections & draw ---
                        for i in range(len(filtered_boxes)):
                            box = filtered_boxes[i]
                            score = float(filtered_scores[i])
                            label = filtered_phrases[i].split('(')[0].strip()
                            xmin, ymin, xmax, ymax = box
                            
                            results_list.append({
                                "label": label,
                                "score": score,
                                "bbox": [float(xmin), float(ymin), float(xmax), float(ymax)]
                            })
                            
                            # Draw Bbox & Label onto frame
                            cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
                            label_str = f"{label}_{i}: {score:.2f}"
                            cv2.putText(frame, label_str, (int(xmin), max(int(ymin)-10, 10)), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Draw Bag timestamp and Universal System Time
            H, W = frame.shape[:2]
            sys_time = time.time() % 100000
            cv2.rectangle(frame, (5, H - 65), (210, H - 5), (255, 255, 255), -1)
            cv2.putText(frame, f"Bag: {time_sec % 10000:.3f}s", (10, H - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Sys: {sys_time:.3f}", (10, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

            # 3. Traversability Segmentation (SAM)
            try:
                seg_overlay = np.zeros((H, W, 3), dtype=np.uint8)
                seg_frame = frame.copy()
                combined_mask = None

                if self.traversable_region:
                    floor_boxes, floor_scores, _ = self.get_grounding_output(
                        self.gd_model, image_tensor, self.traversable_region, self.box_threshold, self.text_threshold, device=self.gd_device
                    )

                    if floor_boxes.numel() > 0:
                        for i in range(floor_boxes.size(0)):
                            floor_boxes[i] = floor_boxes[i] * torch.Tensor([W, H, W, H])
                            floor_boxes[i][:2] -= floor_boxes[i][2:] / 2
                            floor_boxes[i][2:] += floor_boxes[i][:2]

                        # NMS needs CPU tensors, which they already are from get_grounding_output
                        nms_idx = torchvision.ops.nms(floor_boxes, floor_scores, self.iou_threshold).numpy().tolist()

                        if len(nms_idx) > 0:
                            filtered_floor_boxes_tensor = floor_boxes[nms_idx]
                            self.sam_predictor.set_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                            
                            # Transform boxes for SAM and move to the correct device
                            transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(filtered_floor_boxes_tensor, frame.shape[:2]).to(self.sam_device)
                            
                            # Predict masks in a batch
                            masks, _, _ = self.sam_predictor.predict_torch(point_coords=None, point_labels=None, boxes=transformed_boxes, multimask_output=False)
                            
                            combined_mask = torch.any(masks, dim=0).squeeze(0).cpu().numpy()
                            seg_overlay[combined_mask] = (255, 0, 0)  # Blue for floor

                    cv2.putText(seg_frame, self.traversable_region, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
                seg_frame = cv2.addWeighted(seg_frame, 0.6, seg_overlay, 0.4, 0)
                seg_msg = self.bridge.cv2_to_compressed_imgmsg(seg_frame)
                seg_msg.header = img_msg.header
                self.seg_pub.publish(seg_msg)
                
                # Publish the raw binary mask for the costmap
                if combined_mask is not None:
                    mask_img = (combined_mask * 255).astype(np.uint8)
                else:
                    mask_img = np.zeros((H, W), dtype=np.uint8)
                mask_msg = self.bridge.cv2_to_compressed_imgmsg(mask_img, dst_format='png')
                mask_msg.header = img_msg.header
                self.mask_pub.publish(mask_msg)
            except Exception as e:
                rospy.logerr(f"SAM segmentation failed: {e}")

            # Publish stamped detection results using DiagnosticArray
            diag_array = DiagnosticArray()
            diag_array.header = img_msg.header
            status = DiagnosticStatus()
            status.name = "Detections"
            status.message = json.dumps(results_list)
            diag_array.status.append(status)
            self.det_pub.publish(diag_array)

            # Publish Annotated Image
            annotated_msg = self.bridge.cv2_to_compressed_imgmsg(frame)
            annotated_msg.header = img_msg.header
            self.img_pub.publish(annotated_msg)

        except Exception as e:
            rospy.logerr(f"Error in detection callback: {e}")

if __name__ == '__main__':
    try:
        DetectionOpensetNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass