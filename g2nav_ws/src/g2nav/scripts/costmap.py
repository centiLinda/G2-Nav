#!/path_to_your_env/bin/python

import rospy
import numpy as np
import cv2
import json
import time
from cv_bridge import CvBridge

from sensor_msgs.msg import PointCloud2, Image
import sensor_msgs.point_cloud2 as pc2
from nav_msgs.msg import Odometry, OccupancyGrid, MapMetaData
from diagnostic_msgs.msg import DiagnosticArray
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
from geometry_msgs.msg import PointStamped

import message_filters
from scipy.spatial.transform import Rotation
import math
from sklearn.cluster import DBSCAN

# ===================================
# --- TUNABLE PARAMETERS: COSTMAP ---
# ===================================
RESOLUTION = 0.15                # Meters per pixel
MAP_SIZE = 40.0                  # Width and height of the map in meters
Z_MIN = -0.5                     # Minimum height of Lidar points to consider
Z_MAX = 2.0                      # Maximum height of Lidar points to consider

# --- Cost and Thresholds ---
LIDAR_OBSTACLE_COST = 15.0       # Cost amplitude for LiDAR hits
OCCUPANCY_THRESHOLD = 10.0       # Cost threshold to display as occupied in discrete grid
BASE_OBSTACLE_SIGMA = 0.4        # Base Gaussian sigma (radius) for tracked objects in meters
DEFAULT_VLM_SCORE = 3            # Default VLM score if none provided (-1 = leader, 0 = ignore, 1-5 = obstacle)
MAX_VLM_SCORE = 5.0              # Score assigned when the VLM flags a tracking (heading) mismatch
VLM_COST_MULTIPLIER = 10.0       # Multiplier converting VLM score (1-5) to costmap amplitude

# --- Camera Projection for Semantic Mask ---
CAM_INTRINSICS = np.array([
    [608.115906, 0.0,        639.186401],
    [0.0,        607.871704, 363.069061],
    [0.0,        0.0,        1.0]
])
CAM_HEIGHT = 0.70                # Estimated camera height in meters, increase to spread further
CAM_PITCH = 0.0                  # Camera pitch in radians (Positive = looking down)
NON_FLOOR_COST = 20.0            # Cost penalty for non-floor areas

# --- Artificial Potential Fields ---
RADIATION_AMPLITUDE = 10.0       # Cost penalty for facing/moving backward
FORWARD_PULL_WEIGHT = -0.2       # Gradient slope pulling the robot forward
TRAIL_VELOCITY_THRESHOLD = 0.5   # Min speed (m/s) for a leader to generate a dynamic trail
PREDICTION_TIME = 0.5            # Seconds to project moving obstacles forward

# --- Gaussian Shape Tuning ---
GAUSSIAN_FRONT_STRETCH = 0.5     # 0.8 Speed multiplier for front elongation of moving obstacles
GAUSSIAN_SIDE_STRETCH = 0.1      # Speed multiplier for side elongation of moving obstacles
# attraction
TRAIL_DISTANCES = [1.0, 2.0, 3.0, 4.0]  # Meters behind the leader to place attractive points
TRAIL_BASE_AMP = -15.0                  # Peak amplitude of the first trail point (negative = attractive)
TRAIL_AMP_STEP = 2.5                    # Amplitude change per subsequent trail point (fading attraction)
TRAIL_SIGMA_MULTIPLIER = 2.5            # Multiplier for base sigma to make trail points wider

# --- Processing & Visualization ---
LIDAR_SMOOTHING_KERNEL = 5       # Kernel size for LiDAR costmap smoothing (must be odd)
LIDAR_SMOOTHING_SIGMA = 1.0      # Gaussian sigma for LiDAR costmap smoothing
VIZ_VELOCITY_SCALE = 1.5         # Scale multiplier for velocity arrows in visualization

# --- System-1 Safety Guard ---
REFLEX_TIME = 1.5                # Seconds of forward prediction for the collision tube
REFLEX_TUBE_WIDTH = 0.8          # Width of the collision tube in meters
REFLEX_MULTIPLIER = 50.0         # Amplitude multiplier for reflex spikes (scales by 1/TTC)
UNASSIGNED_DIST_THRES = 0.8      # Meters. If a LiDAR point is further than a VLM object, it's considered unassigned
# ==============================================================================

def convert_to_homo_matrix(translation, rotation_quat):
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(rotation_quat).as_matrix()
    matrix[:3, 3] = translation
    return matrix

class CostmapGeneratorNode:
    def __init__(self):
        rospy.init_node('costmap_generator_node', anonymous=True)
        
        self.bridge = CvBridge()
        self.grid_size = int(MAP_SIZE / RESOLUTION)
        
        self.global_goal = None
        
        self.vlm_scores = {}
        rospy.Subscriber('/vlm_obj', String, self.vlm_cb)
        
        self.traversable_region = ""
        self.region_list = []
        rospy.Subscriber('/vlm_status', String, self.vlm_status_cb)
        
        rospy.Subscriber('/global_goal', PointStamped, self.goal_cb)
        
        self.occupancy_pub = rospy.Publisher('/occupancy_viz', OccupancyGrid, queue_size=1)
        self.continuous_map_pub = rospy.Publisher('/costmap_data', Image, queue_size=1)
        self.map_metadata_pub = rospy.Publisher('/costmap_metadata', MapMetaData, queue_size=1, latch=True)
        self.continuous_map_image_pub = rospy.Publisher('/costmap_viz', Image, queue_size=1)
        self.continuous_safe_map_image_pub = rospy.Publisher('/costmap_safe_viz', Image, queue_size=1)
        self.bev_seg_pub = rospy.Publisher('/bev_seg_viz', Image, queue_size=1)
        self.costmap_obj_pub = rospy.Publisher('/costmap_input_obj', String, queue_size=1)
        
        lidar_sub = message_filters.Subscriber('/velodyne_points', PointCloud2, queue_size=50)
        odom_sub = message_filters.Subscriber('/odom', Odometry, queue_size=50)
        tracked_obj_sub = message_filters.Subscriber('/tracked_obj', DiagnosticArray, queue_size=50)
        floor_mask_sub = message_filters.Subscriber('/floor_mask/compressed', CompressedImage, queue_size=50)
        
        # Increased slop slightly to account for potential timing differences from the conversion node
        self.ts = message_filters.ApproximateTimeSynchronizer([lidar_sub, odom_sub, tracked_obj_sub, floor_mask_sub], queue_size=50, slop=0.3)
        self.ts.registerCallback(self.data_callback)

        rospy.loginfo("Costmap Generator Node Ready.")

    def goal_cb(self, msg):
        self.global_goal = np.array([msg.point.x, msg.point.y, msg.point.z])

    def get_local_pose(self, obj, inv_ego, depth_valid, local_pts, fx, fy, cx, cy):
        h_odom = np.array([obj["x"], obj["y"], 0.0, 1.0])
        h_local = inv_ego @ h_odom
        is_depth_overridden = False
        
        if not depth_valid and obj.get("bbox") and any(a in obj.get("label", "").lower() for a in ['person', 'human', 'pedestrian', 'man', 'woman', 'boy', 'girl', 'people', 'guy']):
            xmin, ymin, xmax, ymax = obj["bbox"]
            new_cluster_found = False
            
            # --- 1. Secondary Depth Re-search in PointCloud ---
            if local_pts is not None and len(local_pts) > 0:
                # Calculate azimuth angle limits
                phi_max = math.atan((cx - xmin) / fx)
                phi_min = math.atan((cx - xmax) / fx)
                
                angles = np.arctan2(local_pts[:, 1], local_pts[:, 0])
                in_fov_mask = (angles >= phi_min) & (angles <= phi_max)
                fov_pts = local_pts[in_fov_mask]
                
                if fov_pts.shape[0] > 0:
                    # Exclude the original bad cluster (e.g. background wall) by removing points within 1m
                    dist_to_orig = np.hypot(fov_pts[:, 0] - h_local[0], fov_pts[:, 1] - h_local[1])
                    remaining_pts = fov_pts[dist_to_orig > 1.0]
                    
                    if remaining_pts.shape[0] > 0:
                        # Loosened DBSCAN criteria for second search
                        ranges = np.hypot(remaining_pts[:, 0], remaining_pts[:, 1]).reshape(-1, 1)
                        db = DBSCAN(eps=0.8, min_samples=1).fit(ranges)
                        
                        unique_labels = set(db.labels_)
                        if -1 in unique_labels:
                            unique_labels.remove(-1)
                            
                        if unique_labels:
                            clusters = [remaining_pts[db.labels_ == label] for label in unique_labels]
                            # Assume the closest cluster is the missed human (occluding the background)
                            closest_cluster = min(clusters, key=lambda pts: np.min(np.hypot(pts[:, 0], pts[:, 1])))
                            
                            h_local[0] = np.median(closest_cluster[:, 0])
                            h_local[1] = np.median(closest_cluster[:, 1])
                            is_depth_overridden = True
                            new_cluster_found = True

            # --- 2. Fallback to Pinhole if re-search failed ---
            if not new_cluster_found:
                h_px = ymax - ymin
                if h_px > 0:
                    # Fallback to Pinhole using average human height (1.7m)
                    d_guess = (fy * 1.7) / h_px
                    cx_px = (xmin + xmax) / 2.0
                    x_cam = (cx_px - cx) * d_guess / fx
                    h_local[0] = d_guess
                    h_local[1] = -x_cam
                    is_depth_overridden = True
                
        return h_local, is_depth_overridden

    def vlm_cb(self, msg):
        try:
            self.vlm_scores = json.loads(msg.data)
        except Exception as e:
            rospy.logwarn(f"Failed to parse VLM scores: {e}")

    def vlm_status_cb(self, msg):
        try:
            data = json.loads(msg.data)
            self.traversable_region = data.get('traversable_region', '').lower()
            self.region_list = [r.lower() for r in data.get('region_list', [])]
        except Exception as e:
            rospy.logwarn(f"Failed to parse VLM status in costmap: {e}")

    def data_callback(self, lidar_msg, odom_msg, tracked_obj_msg, floor_mask_msg):
        try:
            # Parse tracked objects from the synchronized message
            if tracked_obj_msg.status:
                tracked_objects = json.loads(tracked_obj_msg.status[0].message)
            else:
                tracked_objects = []
        except Exception as e:
            rospy.logwarn(f"Failed to parse tracked objects: {e}")
            tracked_objects = []
            
        try:

            translation = np.array([
                odom_msg.pose.pose.position.x,
                odom_msg.pose.pose.position.y,
                odom_msg.pose.pose.position.z
            ])
            rotation_quat = np.array([
                odom_msg.pose.pose.orientation.x,
                odom_msg.pose.pose.orientation.y,
                odom_msg.pose.pose.orientation.z,
                odom_msg.pose.pose.orientation.w
            ])
            ego_homo_trans = convert_to_homo_matrix(translation, rotation_quat)
            inv_ego = np.linalg.inv(ego_homo_trans)

            lidar_grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
            local_origin_x = -MAP_SIZE / 2.0
            local_origin_y = -MAP_SIZE / 2.0

            # --- 1. Process Lidar PointCloud ---
            point_generator = pc2.read_points(lidar_msg, field_names=("x", "y", "z"), skip_nans=True)
            raw_lidar_data = np.array(list(point_generator))
            local_pts = np.array([])

            if raw_lidar_data.shape[0] > 0:
                valid_mask = (raw_lidar_data[:, 2] >= Z_MIN) & (raw_lidar_data[:, 2] <= Z_MAX)
                local_pts = raw_lidar_data[valid_mask]

                if local_pts.shape[0] > 0:
                    gx = ((local_pts[:, 0] - local_origin_x) / RESOLUTION).astype(np.int32)
                    gy = ((local_pts[:, 1] - local_origin_y) / RESOLUTION).astype(np.int32)

                    in_bounds = (gx >= 0) & (gx < self.grid_size) & (gy >= 0) & (gy < self.grid_size)
                    lidar_grid[gy[in_bounds], gx[in_bounds]] = LIDAR_OBSTACLE_COST  

            # Smooth the raw lidar for the continuous map
            continuous_cost_map = cv2.GaussianBlur(lidar_grid.astype(np.float32), (LIDAR_SMOOTHING_KERNEL, LIDAR_SMOOTHING_KERNEL), LIDAR_SMOOTHING_SIGMA)
            
            # --- Extract Floor Mask ---
            try:
                floor_mask = self.bridge.compressed_imgmsg_to_cv2(floor_mask_msg, "passthrough")
                if len(floor_mask.shape) == 3:
                    floor_mask = cv2.cvtColor(floor_mask, cv2.COLOR_BGR2GRAY)
                mask_h, mask_w = floor_mask.shape
            except Exception as e:
                rospy.logwarn(f"Failed to decode floor mask: {e}")
                floor_mask = None
                mask_h, mask_w = 720, 1280  # Default fallback
                
            # Dynamically scale intrinsics to match actual image resolution
            scale_x = mask_w / 1280.0
            scale_y = mask_h / 720.0
            fx = CAM_INTRINSICS[0, 0] * scale_x
            cx = CAM_INTRINSICS[0, 2] * scale_x
            fy = CAM_INTRINSICS[1, 1] * scale_y
            cy = CAM_INTRINSICS[1, 2] * scale_y

            # --- 2. Goal attraction (forward direction) ---
            h, w = self.grid_size, self.grid_size
            Y_img, X_img = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
            
            # In local frame: dx is X_img (forward), dy is Y_img (left)
            dx_px = X_img - (self.grid_size / 2.0)
            dy_px = Y_img - (self.grid_size / 2.0)
            
            # Calculate vector to global goal in local frame
            if self.global_goal is not None:
                v_goal_global = self.global_goal - translation
                v_goal_local = inv_ego[:3, :3] @ v_goal_global
    
                goal_dist = np.hypot(v_goal_local[0], v_goal_local[1])
                if goal_dist > 1e-3:
                    u_x = v_goal_local[0] / goal_dist
                    u_y = v_goal_local[1] / goal_dist
                else:
                    u_x, u_y = 1.0, 0.0
            else:
                u_x, u_y = 1.0, 0.0
                
            goal_angle = math.atan2(u_y, u_x)
            theta = np.arctan2(dy_px, dx_px)
            
            # Penalize moving away from the global goal direction
            radiation_cost = RADIATION_AMPLITUDE * (1.0 - np.cos(theta - goal_angle))
            
            # Pull towards the global goal direction instead of strictly straight ahead
            d_pull = dx_px * u_x + dy_px * u_y
            forward_pull = FORWARD_PULL_WEIGHT * d_pull
            
            continuous_cost_map += (radiation_cost + forward_pull).astype(np.float32)

            # --- 2.5 Apply Non-Floor Semantic Cost ---
            bev_seg_img = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)
            if floor_mask is not None:
                gy, gx = np.meshgrid(np.arange(self.grid_size), np.arange(self.grid_size), indexing='ij')
                x_local = gx * RESOLUTION + local_origin_x
                y_local = gy * RESOLUTION + local_origin_y
                
                # Only consider cells strictly in front of the camera (x > 0)
                front_mask = x_local > 0.1
                x_front = x_local[front_mask]
                y_front = y_local[front_mask]
                
                # Base projection logic: X_cam = -y_local, Y_cam = CAM_HEIGHT, Z_cam = x_local
                Y_cam = CAM_HEIGHT
                Z_cam = x_front
                
                # Apply Pitch rotation (positive means looking down)
                Y_pitched = Y_cam * np.cos(CAM_PITCH) - Z_cam * np.sin(CAM_PITCH)
                Z_pitched = Y_cam * np.sin(CAM_PITCH) + Z_cam * np.cos(CAM_PITCH)
                
                # Prevent division by zero or projecting negative depths
                Z_pitched = np.maximum(Z_pitched, 1e-5)
                
                u = (fx * (-y_front / Z_pitched) + cx).astype(np.int32)
                v = (fy * (Y_pitched / Z_pitched) + cy).astype(np.int32)
                
                # Filter valid pixels within image bounds
                valid_img_mask = (u >= 0) & (u < mask_w) & (v >= 0) & (v < mask_h) & (Z_pitched > 0.1)
                u_valid = u[valid_img_mask]
                v_valid = v[valid_img_mask]
                
                # Identify non-floor pixels (mask < 128 means it is NOT the floor)
                is_non_floor = floor_mask[v_valid, u_valid] < 128
                is_floor = floor_mask[v_valid, u_valid] >= 128
                
                # Map back to 2D grid indices
                valid_idx = np.where(valid_img_mask)[0]
                invalid_idx = np.where(~valid_img_mask)[0]
                
                gy_non_floor = gy[front_mask][valid_idx[is_non_floor]]
                gx_non_floor = gx[front_mask][valid_idx[is_non_floor]]
                
                gy_floor = gy[front_mask][valid_idx[is_floor]]
                gx_floor = gx[front_mask][valid_idx[is_floor]]
                
                gy_blind = np.concatenate((gy[front_mask][invalid_idx], gy[~front_mask]))
                gx_blind = np.concatenate((gx[front_mask][invalid_idx], gx[~front_mask]))
                
                # Apply higher cost for non-traversable and blind spot areas
                continuous_cost_map[gy_non_floor, gx_non_floor] += NON_FLOOR_COST
                continuous_cost_map[gy_blind, gx_blind] += NON_FLOOR_COST
                
                # Colorize the BEV segmentation map
                bev_seg_img[gy_floor, gx_floor] = [255, 0, 0]         # Blue for Floor
                bev_seg_img[gy_non_floor, gx_non_floor] = [0, 0, 255] # Red for Non-Floor

            # Rotate and flip BEV image to align with robot facing UP
            bev_seg_img = cv2.rotate(bev_seg_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            bev_seg_img = cv2.flip(bev_seg_img, 1)
            
            # Draw robot facing UP
            bev_h, bev_w = bev_seg_img.shape[:2]
            robot_gx = (0.0 - local_origin_x) / RESOLUTION
            robot_gy = (0.0 - local_origin_y) / RESOLUTION
            robot_img_x = int(round(bev_w - 1 - robot_gy))
            robot_img_y = int(round(bev_h - 1 - robot_gx))
            cv2.circle(bev_seg_img, (robot_img_x, robot_img_y), 5, (255, 255, 255), -1)
            cv2.circle(bev_seg_img, (robot_img_x, robot_img_y), 6, (0, 0, 0), 1)
            cv2.arrowedLine(bev_seg_img, (robot_img_x, robot_img_y), (robot_img_x, robot_img_y - 15), (0, 0, 0), 2, tipLength=0.5)

            # Publish BEV Segmentation Map
            bev_seg_msg = self.bridge.cv2_to_imgmsg(bev_seg_img, encoding="bgr8")
            bev_seg_msg.header = lidar_msg.header
            bev_seg_msg.header.frame_id = odom_msg.header.frame_id
            self.bev_seg_pub.publish(bev_seg_msg)

            # --- 3. Tracked Objects Costs ---
            tracked_local_positions = []
            tracked_local_ids = []
            dashboard_final_objs = []
            for obj in tracked_objects:
                label_lower = obj.get("label", "").lower()
                if label_lower == self.traversable_region or label_lower in self.region_list:
                    continue

                obj_id_str = str(obj.get("id", ""))
                
                is_default = False
                if obj_id_str not in self.vlm_scores:
                    vlm_score = DEFAULT_VLM_SCORE
                    tracking_valid = True
                    depth_valid = True
                    is_default = True
                else:
                    score_data = self.vlm_scores[obj_id_str]
                    if isinstance(score_data, dict):
                        vlm_score = float(score_data.get("score", 0.0))
                        tracking_valid = score_data.get("tracking_valid", True)
                        depth_valid = score_data.get("depth_valid", True)
                    else:
                        vlm_score = float(score_data)
                        tracking_valid = True
                        depth_valid = True

                if vlm_score == 0.0:
                    continue # Object was evaluated and ignored by VLM

                # Upstream verification: a heading mismatch means uncertain motion,
                # so raise the social score to the maximum
                if not tracking_valid:
                    vlm_score = MAX_VLM_SCORE

                dashboard_final_objs.append({
                    "id": obj_id_str,
                    "label": obj.get("label", "unknown"),
                    "score": vlm_score,
                    "tracking_valid": tracking_valid,
                    "depth_valid": depth_valid,
                    "is_default": is_default
                })

                # Transform object from odom to local frame
                h_local, _ = self.get_local_pose(obj, inv_ego, depth_valid, local_pts, fx, fy, cx, cy)
                
                px = int((h_local[0] - local_origin_x) / RESOLUTION)
                py = int((h_local[1] - local_origin_y) / RESOLUTION)
                
                if px < 0 or px >= w or py < 0 or py >= h:
                    continue

                tracked_local_positions.append([h_local[0], h_local[1]])
                tracked_local_ids.append(obj_id_str)
                    
                # Transform velocity from odom to local frame
                v_odom = np.array([obj["vx"], obj["vy"], 0.0])
                v_local = inv_ego[:3, :3] @ v_odom
                
                vx_px = v_local[0] / RESOLUTION
                vy_px = v_local[1] / RESOLUTION
                base_sigma_px = BASE_OBSTACLE_SIGMA / RESOLUTION
                
                # Heading mismatch: the tracked velocity is unreliable, so treat the object as static
                if not tracking_valid:
                    vx_px = 0.0
                    vy_px = 0.0

                if vlm_score < 0.0:
                    continuous_cost_map = self.add_trailing_attraction(
                        continuous_cost_map, px, py, vx_px, vy_px, 
                        self.grid_size / 2.0, self.grid_size / 2.0, base_sigma_px
                    )
                else:
                    peak_cost = VLM_COST_MULTIPLIER * vlm_score
                    continuous_cost_map = self.add_directional_gaussian(
                        continuous_cost_map, px, py, vx_px, vy_px, 
                        amplitude=peak_cost, base_sigma=base_sigma_px
                    )

            self.costmap_obj_pub.publish(String(data=json.dumps(dashboard_final_objs)))

            # --- Publish Pre-Guard Visualization (/costmap_viz) ---
            self.publish_costmap_viz(continuous_cost_map.copy(), tracked_obj_msg.header, tracked_objects, inv_ego, local_origin_x, local_origin_y, self.continuous_map_image_pub, local_pts, odom_msg.header.frame_id, fx, fy, cx, cy)

            # --- 4. System-1 Safety Guard: Raw-Sensor Reflex Zone ---
            v_robot = max(odom_msg.twist.twist.linear.x, 0.0)
            w_robot = odom_msg.twist.twist.angular.z
            
            if local_pts.shape[0] > 0:
                if abs(w_robot) < 0.05:
                    in_tube = (local_pts[:, 0] > 0) & (local_pts[:, 0] < v_robot * REFLEX_TIME) & (np.abs(local_pts[:, 1]) < REFLEX_TUBE_WIDTH / 2.0)
                    ttc = local_pts[:, 0] / (v_robot + 1e-5)
                else:
                    R_turn = v_robot / w_robot
                    dist_to_icc = np.hypot(local_pts[:, 0], local_pts[:, 1] - R_turn)
                    in_tube_width = np.abs(dist_to_icc - abs(R_turn)) < (REFLEX_TUBE_WIDTH / 2.0)
                    
                    # Calculate arc angle (safeguarded turning directions)
                    theta = np.arctan2(local_pts[:, 0], np.abs(R_turn) - local_pts[:, 1] * np.sign(w_robot))
                    in_tube_length = (theta > 0) & (theta < abs(w_robot) * REFLEX_TIME)
                    
                    in_tube = in_tube_width & in_tube_length
                    ttc = theta / (abs(w_robot) + 1e-5)
                
                reflex_pts = local_pts[in_tube]
                reflex_ttc = ttc[in_tube]
                
                if reflex_pts.shape[0] > 0:
                    # Unassigned check: filter out points already explained by tracked objects
                    if len(tracked_local_positions) > 0:
                        tracked_arr = np.array(tracked_local_positions)
                        diffs = reflex_pts[:, None, :2] - tracked_arr[None, :, :]
                        dists = np.linalg.norm(diffs, axis=-1)
                        min_dists = np.min(dists, axis=1)
                        unassigned_mask = min_dists > UNASSIGNED_DIST_THRES
                        
                        assigned_mask = ~unassigned_mask
                        if np.any(unassigned_mask):
                            rospy.logwarn_throttle(1.0, f"System-1 Reflex Guard Activated! {np.sum(unassigned_mask)} rogue points in collision tube.")
                        elif np.any(assigned_mask):
                            explaining_indices = np.unique(np.argmin(dists, axis=1)[assigned_mask])
                            explaining_ids = [tracked_local_ids[idx] for idx in explaining_indices]
                            rospy.loginfo_throttle(1.0, f"Reflex tube points ({np.sum(assigned_mask)}) all safely explained by VLM obj IDs: {explaining_ids}")
                    else:
                        unassigned_mask = np.ones(reflex_pts.shape[0], dtype=bool)
                        rospy.logwarn_throttle(1.0, f"System-1 Reflex Guard Activated! {reflex_pts.shape[0]} rogue points in collision tube (no tracked objects).")
                        
                    unassigned_reflex_pts = reflex_pts[unassigned_mask]
                    unassigned_ttc = reflex_ttc[unassigned_mask]
                    
                    for i in range(unassigned_reflex_pts.shape[0]):
                        pt = unassigned_reflex_pts[i]
                        t_val = max(unassigned_ttc[i], 0.1) # Minimum TTC buffer
                        rx = int((pt[0] - local_origin_x) / RESOLUTION)
                        ry = int((pt[1] - local_origin_y) / RESOLUTION)
                        if 0 <= rx < w and 0 <= ry < h:
                            continuous_cost_map = self.add_gaussian(
                                continuous_cost_map, rx, ry, 
                                amplitude=REFLEX_MULTIPLIER / t_val, 
                                sigma=BASE_OBSTACLE_SIGMA / RESOLUTION
                            )

            # --- 6. Publish OccupancyGrid (pure lidar data) ---
            display_map = np.zeros_like(lidar_grid, dtype=np.int8)
            display_map[lidar_grid >= OCCUPANCY_THRESHOLD] = 100
            
            origin_odom = ego_homo_trans @ np.array([local_origin_x, local_origin_y, 0.0, 1.0])
            
            grid_msg = OccupancyGrid()
            grid_msg.header.stamp = lidar_msg.header.stamp
            grid_msg.header.frame_id = odom_msg.header.frame_id
            grid_msg.info.resolution = RESOLUTION
            grid_msg.info.width = self.grid_size
            grid_msg.info.height = self.grid_size
            grid_msg.info.origin.position.x = origin_odom[0]
            grid_msg.info.origin.position.y = origin_odom[1]
            grid_msg.info.origin.position.z = translation[2]
            grid_msg.info.origin.orientation.x = rotation_quat[0]
            grid_msg.info.origin.orientation.y = rotation_quat[1]
            grid_msg.info.origin.orientation.z = rotation_quat[2]
            grid_msg.info.origin.orientation.w = rotation_quat[3]
            grid_msg.data = display_map.flatten().tolist()
            
            self.occupancy_pub.publish(grid_msg)

            # --- 7. Publish Guarded Continuous Costmap for downstream planning ---
            float_image_msg = self.bridge.cv2_to_imgmsg(continuous_cost_map, encoding="32FC1")
            float_image_msg.header = lidar_msg.header
            float_image_msg.header.frame_id = odom_msg.header.frame_id
            self.continuous_map_pub.publish(float_image_msg)
            
            meta_msg = MapMetaData()
            meta_msg.map_load_time = lidar_msg.header.stamp
            meta_msg.resolution = RESOLUTION
            meta_msg.width = self.grid_size
            meta_msg.height = self.grid_size
            meta_msg.origin.position.x = origin_odom[0]
            meta_msg.origin.position.y = origin_odom[1]
            meta_msg.origin.position.z = translation[2]
            meta_msg.origin.orientation.x = rotation_quat[0]
            meta_msg.origin.orientation.y = rotation_quat[1]
            meta_msg.origin.orientation.z = rotation_quat[2]
            meta_msg.origin.orientation.w = rotation_quat[3]
            self.map_metadata_pub.publish(meta_msg)

            # --- 8. Publish Post-Guard Visualization (/costmap_safe_viz) ---
            self.publish_costmap_viz(continuous_cost_map, tracked_obj_msg.header, tracked_objects, inv_ego, local_origin_x, local_origin_y, self.continuous_safe_map_image_pub, local_pts, odom_msg.header.frame_id, fx, fy, cx, cy)

        except Exception as e:
            rospy.logerr(f"Error generating costmap: {e}")

    def publish_costmap_viz(self, cost_map, header, tracked_objects, inv_ego, local_origin_x, local_origin_y, publisher, local_pts, frame_id, fx, fy, cx, cy):
        h, w = cost_map.shape
        norm_map = cv2.normalize(cost_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        color_map = cv2.applyColorMap(norm_map, cv2.COLORMAP_JET)
        
        # Rotate so Robot Forward is UP, Left is LEFT
        color_map = cv2.rotate(color_map, cv2.ROTATE_90_COUNTERCLOCKWISE)
        color_map = cv2.flip(color_map, 1)
        
        t_sec = header.stamp.to_sec()
        sys_time = time.time() % 100000
        cv2.rectangle(color_map, (0, h - 30), (110, h), (255, 255, 255), -1)
        cv2.putText(color_map, f"Bag: {t_sec % 10000:.3f}s", (5, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(color_map, f"Sys: {sys_time:.3f}", (5, h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
        
        # Draw robot facing UP
        robot_gx = (0.0 - local_origin_x) / RESOLUTION
        robot_gy = (0.0 - local_origin_y) / RESOLUTION
        robot_img_x = int(round(w - 1 - robot_gy))
        robot_img_y = int(round(h - 1 - robot_gx))
        cv2.circle(color_map, (robot_img_x, robot_img_y), 5, (255, 255, 255), -1)
        cv2.circle(color_map, (robot_img_x, robot_img_y), 6, (0, 0, 0), 1)
        cv2.arrowedLine(color_map, (robot_img_x, robot_img_y), (robot_img_x, robot_img_y - 15), (0, 0, 0), 2, tipLength=0.5)
        
        for obj in tracked_objects:
            label_lower = obj.get("label", "").lower()
            if label_lower == self.traversable_region or label_lower in self.region_list:
                continue

            obj_id_str = str(obj.get("id", ""))
            if obj_id_str in self.vlm_scores:
                score_data = self.vlm_scores[obj_id_str]
                if isinstance(score_data, dict):
                    score_val = float(score_data.get("score", 0.0))
                    is_valid = score_data.get("tracking_valid", True)
                    depth_valid = score_data.get("depth_valid", True)
                else:
                    score_val = float(score_data)
                    is_valid = True
                    depth_valid = True
                if score_val == 0.0:
                    continue # Skip drawing explicitly ignored objects
            else:
                is_valid = True
                depth_valid = True
            
            h_local, is_depth_overridden = self.get_local_pose(obj, inv_ego, depth_valid, local_pts, fx, fy, cx, cy)
            px = (h_local[0] - local_origin_x) / RESOLUTION
            py = (h_local[1] - local_origin_y) / RESOLUTION
            
            if 0 <= px < w and 0 <= py < h:
                img_x = int(w - 1 - py)
                img_y = int(h - 1 - px)
                label_str = f"{obj['label']}_{obj['id']}"
                
                marker_color = (0, 255, 0) # for viz
                    
                cv2.circle(color_map, (img_x, img_y), 4, marker_color, -1)
                cv2.circle(color_map, (img_x, img_y), 5, (0, 0, 0), 1)
                cv2.putText(color_map, label_str, (img_x + 6, img_y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                
                if obj.get("is_dynamic", False):
                    v_odom = np.array([obj["vx"], obj["vy"], 0.0])
                    v_local = inv_ego[:3, :3] @ v_odom
                    vx_px = v_local[0] / RESOLUTION
                    vy_px = v_local[1] / RESOLUTION
                    end_px = px + vx_px * VIZ_VELOCITY_SCALE
                    end_py = py + vy_px * VIZ_VELOCITY_SCALE
                    end_img_x = int(w - 1 - end_py)
                    end_img_y = int(h - 1 - end_px)
                    arrow_color = (0, 0, 0) if is_valid else (0, 0, 255)
                    cv2.arrowedLine(color_map, (img_x, img_y), (end_img_x, end_img_y), arrow_color, 2, tipLength=0.3)

        color_image_msg = self.bridge.cv2_to_imgmsg(color_map, encoding="bgr8")
        color_image_msg.header = header
        color_image_msg.header.frame_id = frame_id
        publisher.publish(color_image_msg)

    def add_gaussian(self, cost_map, cx, cy, amplitude, sigma):
        h, w = cost_map.shape
        radius = int(3 * sigma)
        x_min = max(0, int(cx) - radius)
        x_max = min(w, int(cx) + radius + 1)
        y_min = max(0, int(cy) - radius)
        y_max = min(h, int(cy) + radius + 1)
        if x_min >= x_max or y_min >= y_max:
            return cost_map
        X, Y = np.meshgrid(np.arange(x_min, x_max), np.arange(y_min, y_max))
        G = amplitude * np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * sigma**2))
        cost_map[y_min:y_max, x_min:x_max] += G
        return cost_map
        
    def add_directional_gaussian(self, cost_map, cx, cy, vx_px, vy_px, amplitude, base_sigma):
        h, w = cost_map.shape
        speed_px = np.hypot(vx_px, vy_px)
        if speed_px < 1e-2:
            return self.add_gaussian(cost_map, cx, cy, amplitude, base_sigma)
        shift_time = PREDICTION_TIME
        mu_x = cx + vx_px * shift_time
        mu_y = cy + vy_px * shift_time
        sigma_front = base_sigma + GAUSSIAN_FRONT_STRETCH * speed_px
        sigma_side = base_sigma + GAUSSIAN_SIDE_STRETCH * speed_px
        radius = int(3 * max(sigma_front, sigma_side))
        x_min = max(0, int(mu_x) - radius)
        x_max = min(w, int(mu_x) + radius + 1)
        y_min = max(0, int(mu_y) - radius)
        y_max = min(h, int(mu_y) + radius + 1)
        if x_min >= x_max or y_min >= y_max:
            return cost_map
        X, Y = np.meshgrid(np.arange(x_min, x_max), np.arange(y_min, y_max))
        theta = np.arctan2(vy_px, vx_px)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        dx = X - mu_x
        dy = Y - mu_y
        x_rot = dx * cos_t + dy * sin_t
        y_rot = -dx * sin_t + dy * cos_t
        G = amplitude * np.exp(-((x_rot**2) / (2 * sigma_front**2) + (y_rot**2) / (2 * sigma_side**2)))
        cost_map[y_min:y_max, x_min:x_max] += G
        return cost_map

    def add_trailing_attraction(self, cost_map, cx, cy, vx_px, vy_px, robot_px, robot_py, base_sigma):
        speed_px = np.hypot(vx_px, vy_px)
        if speed_px > TRAIL_VELOCITY_THRESHOLD / RESOLUTION:  
            dir_x = -vx_px / speed_px
            dir_y = -vy_px / speed_px
        else:
            dx = robot_px - cx
            dy = robot_py - cy
            dist = np.hypot(dx, dy)
            if dist > 1e-3:
                dir_x = dx / dist
                dir_y = dy / dist
            else:
                dir_x, dir_y = 0.0, 1.0
        for i, dist_m in enumerate(TRAIL_DISTANCES):
            trail_dist_px = dist_m / RESOLUTION
            mu_x = cx + dir_x * trail_dist_px
            mu_y = cy + dir_y * trail_dist_px
            amp = TRAIL_BASE_AMP + i * TRAIL_AMP_STEP 
            cost_map = self.add_gaussian(cost_map, int(mu_x), int(mu_y), amplitude=amp, sigma=base_sigma * TRAIL_SIGMA_MULTIPLIER)
        cost_map = self.add_gaussian(cost_map, cx, cy, amplitude=LIDAR_OBSTACLE_COST, sigma=base_sigma)
        return cost_map

if __name__ == '__main__':
    try:
        CostmapGeneratorNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass