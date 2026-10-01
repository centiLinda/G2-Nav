#!/path_to_your_env/bin/python

import rospy
import numpy as np
import math
import json
import cv2
import time

from sensor_msgs.msg import LaserScan, CompressedImage
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus
from cv_bridge import CvBridge
import message_filters

from diagnostic_msgs.msg import DiagnosticArray
from sklearn.cluster import DBSCAN
from scipy.stats.distributions import chi2
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation

# ====================================================
# --- TUNABLE PARAMETERS: TRACKING & SENSOR FUSION ---
# ====================================================
MAX_OBJECT_SPEED = 15.0       # Max speed constraint for Kalman Filter (m/s)
CONFIRMATION_HITS = 3         # Frames required to confirm a new track
TRACKING_COAST_FRAMES = 15    # Frames a track is kept alive without a visual detection
IOU_MATCH_THRESHOLD = 0.1     # Minimum 2D BBox IoU to associate detection with track
MIN_CLUSTER_WIDTH = 0.05      # Minimum physical width (m) from LiDAR to be considered an object
MAX_CLUSTER_WIDTH = 10.0      # Maximum physical width (m) from LiDAR to be considered an object
KF_MEASUREMENT_NOISE = 1.0    # Variance of LiDAR measurement noise
KF_PROCESS_NOISE = 5.0        # Variance of object acceleration/velocity change
KF_CHI2_GATE_PROBABILITY = 0.99 # Probability threshold for Mahalanobis distance gate

# Camera intrinsics (Used to project bounding boxes onto the LiDAR scan), from SCAND website
CAM_INTRINSICS = np.array([
    [608.115906, 0.0,        639.186401],
    [0.0,        607.871704, 363.069061],
    [0.0,        0.0,        1.0]
])
# ==============================================================================

def bb_iou(boxA, boxB):
    """Calculate Intersection over Union (IoU) of two bounding boxes [xmin, ymin, xmax, ymax]"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, float(xB - xA)) * max(0, float(yB - yA))
    if interArea == 0.0:
        return 0.0
    boxAArea = float((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = float((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    return interArea / (boxAArea + boxBArea - interArea)

def convert_to_homo_matrix(translation, rotation_quat):
    """Convert translation and quaternion to 4x4 homogeneous transformation matrix"""
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(rotation_quat).as_matrix()
    matrix[:3, 3] = translation
    return matrix

def associate_detections_to_trackers(detections, trackers, iou_threshold=0.1):
    """Matches incoming 2D detections to existing Kalman Filter tracks using IoU"""
    if len(trackers) == 0 or len(detections) == 0:
        return [], list(range(len(detections))), list(range(len(trackers)))
        
    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            if trk.bbox is not None:
                iou_matrix[d, t] = bb_iou(det['bbox'], trk.bbox)
                
    matched_indices = linear_sum_assignment(-iou_matrix)
    matched_indices = np.asarray(matched_indices).T
    
    matches = []
    unmatched_detections = []
    unmatched_trackers = list(range(len(trackers)))
    
    for d in range(len(detections)):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
            
    for m in matched_indices:
        d, t = m[0], m[1]
        if iou_matrix[d, t] < iou_threshold:
            unmatched_detections.append(d)
        else:
            matches.append((d, t))
            if t in unmatched_trackers:
                unmatched_trackers.remove(t)
                
    return matches, unmatched_detections, unmatched_trackers


class ObjectKalmanFilter:
    """
    Lightweight Constant-Velocity Kalman Filter for tracking objects in the 2D real-world plane.
    State vector: [x, y, vx, vy]
    """
    def __init__(self, track_id, x, y, time_sec, max_speed=MAX_OBJECT_SPEED, initial_score=0.0, label='unknown', bbox=None):
        self.track_id = track_id
        self.state = np.array([x, y, 0.0, 0.0])
        self.P = np.eye(4) * 5.0  # High initial uncertainty
        self.last_time = time_sec
        self.first_time = time_sec
        self.latest_score = initial_score
        self.bbox = bbox
        
        self.H = np.array([[1.0, 0.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0, 0.0]])
        self.R = np.eye(2) * KF_MEASUREMENT_NOISE 
        
        self.status = 'tentative'
        self.hits = 1
        self.misses = 0
        self.age = 1
        self.is_dynamic = False 
        self.history = []
        self.max_speed = max_speed

        self.label = label
        self.labels_history = [label]

    def predict(self, current_time):
        dt = current_time - self.last_time
        if dt <= 0: return
        
        F = np.array([[1.0, 0.0,  dt, 0.0],
                      [0.0, 1.0, 0.0,  dt],
                      [0.0, 0.0, 1.0, 0.0],
                      [0.0, 0.0, 0.0, 1.0]])
        
        q = KF_PROCESS_NOISE
        Q = np.array([[(dt**4)/4, 0, (dt**3)/2, 0],
                      [0, (dt**4)/4, 0, (dt**3)/2],
                      [(dt**3)/2, 0, dt**2, 0],
                      [0, (dt**3)/2, 0, dt**2]]) * q
                      
        self.state = F @ self.state
        self.P = F @ self.P @ F.T + Q
        self.last_time = current_time
        self.age += 1
        self.misses += 1

    def update(self, z, score=None, label=None, bbox=None):
        if score is not None: self.latest_score = score
        if bbox is not None: self.bbox = bbox
            
        if label is not None and label != 'unknown':
            self.labels_history.append(label)
            if len(self.labels_history) > 15:
                self.labels_history.pop(0)
            self.label = max(set(self.labels_history), key=self.labels_history.count)

        y_res = z - (self.H @ self.state)
        S = self.H @ self.P @ self.H.T + self.R
        
        try:
            K = self.P @ self.H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return

        self.hits += 1
        self.misses = 0
        self.history.append(self.state[:2].copy())
        if len(self.history) > 15:
            self.history.pop(0)
        
        self.state = self.state + (K @ y_res)
        I = np.eye(4)
        self.P = (I - K @ self.H) @ self.P
        
        v_norm = np.linalg.norm(self.state[2:4])
        if v_norm > self.max_speed:
            self.state[2:4] = (self.state[2:4] / v_norm) * self.max_speed
            
        if self.hits > 5 and len(self.history) >= 5:
            disp = np.linalg.norm(self.history[-1] - self.history[0])
            if disp > 0.3:
                self.is_dynamic = True

    def mahalanobis_distance(self, z):
        y_res = z - (self.H @ self.state)
        S = self.H @ self.P @ self.H.T + self.R
        try:
            inv_S = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return float('inf')
        return y_res.T @ inv_S @ y_res


class TrackingNode:
    def __init__(self):
        rospy.init_node('tracking_node', anonymous=True)
        self.bridge = CvBridge()
        
        # State Management
        self.kfs = {}
        self.next_id = 1
        self.previous_object_ids = set()
        self.chi2_gate_threshold = chi2.ppf(KF_CHI2_GATE_PROBABILITY, df=2)
        
        # Publishers
        self.obj_pub = rospy.Publisher('/tracked_obj', DiagnosticArray, queue_size=1)
        self.marker_pub = rospy.Publisher('/tracked_obj_markers', MarkerArray, queue_size=1)
        self.annotated_img_pub = rospy.Publisher('/tracked_obj_annotated_img/compressed', CompressedImage, queue_size=1)
        
        # Subscribers
        det_sub = message_filters.Subscriber('/raw_det', DiagnosticArray, queue_size=20)
        scan_sub = message_filters.Subscriber('/scan', LaserScan, queue_size=100)
        odom_sub = message_filters.Subscriber('/odom', Odometry, queue_size=500)
        img_sub = message_filters.Subscriber('/raw_det_annotated_img/compressed', CompressedImage, queue_size=20, buff_size=2**24)
        
        # Increased slop slightly to account for potential timing differences from multiple sources
        self.ts = message_filters.ApproximateTimeSynchronizer([det_sub, scan_sub, odom_sub, img_sub], queue_size=100, slop=0.2)
        self.ts.registerCallback(self.det_cb)
        
        rospy.loginfo("Tracking Node Ready.")

    def get_object_depth_from_scan(self, xmin, xmax, scan_msg, img_w):
        scale_x = img_w / 1280.0
        fx = CAM_INTRINSICS[0, 0] * scale_x
        cx = CAM_INTRINSICS[0, 2] * scale_x
        
        phi_max = math.atan((cx - xmin) / fx)
        phi_min = math.atan((cx - xmax) / fx)
        
        valid_points = []
        for i, r in enumerate(scan_msg.ranges):
            if scan_msg.range_min < r < scan_msg.range_max:
                angle = scan_msg.angle_min + i * scan_msg.angle_increment
                if phi_min <= angle <= phi_max:
                    valid_points.append((r, angle))
                    
        if len(valid_points) < 2: return None, None, "lidar < 2 valid points"
            
        ranges = np.array([p[0] for p in valid_points]).reshape(-1, 1)
        db = DBSCAN(eps=0.5, min_samples=2).fit(ranges)
        labels = db.labels_
        
        unique_labels = set(labels)
        if -1 in unique_labels: unique_labels.remove(-1)
        if not unique_labels: return None, None, "DBSCAN found no clusters"
            
        clusters = [np.array(valid_points)[labels == label] for label in unique_labels]
        
        valid_sized_clusters = []
        for cluster in clusters:
            r_first, a_first = cluster[0]
            r_last, a_last = cluster[-1]
            x1, y1 = r_first * math.cos(a_first), r_first * math.sin(a_first)
            x2, y2 = r_last * math.cos(a_last), r_last * math.sin(a_last)
            width = math.hypot(x1 - x2, y1 - y2)
            
            if MIN_CLUSTER_WIDTH <= width <= MAX_CLUSTER_WIDTH:
                valid_sized_clusters.append(cluster)
                
        if valid_sized_clusters:
            closest_cluster_points = min(valid_sized_clusters, key=lambda points: np.min(points[:, 0]))
        else:
            closest_cluster_points = min(clusters, key=lambda points: np.min(points[:, 0]))
        
        median_r = np.median(closest_cluster_points[:, 0])
        mean_angle = np.mean(closest_cluster_points[:, 1])
        
        return median_r, mean_angle, None

    def process_3d_state(self, det, scan_msg, odom_msg, img_w):
        xmin, _, xmax, _ = det['bbox']
        r, angle, err = self.get_object_depth_from_scan(xmin, xmax, scan_msg, img_w)
        
        if r is not None:
            x_base = r * math.cos(angle)
            y_base = r * math.sin(angle)
            
            translation = np.array([odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y, odom_msg.pose.pose.position.z])
            rotation_quat = np.array([odom_msg.pose.pose.orientation.x, odom_msg.pose.pose.orientation.y, odom_msg.pose.pose.orientation.z, odom_msg.pose.pose.orientation.w])
            ego_homo_trans = convert_to_homo_matrix(translation, rotation_quat)
            
            p_base = np.array([x_base, y_base, 0, 1])
            p_world = ego_homo_trans @ p_base
            return np.array([p_world[0], p_world[1]]), None
        return None, err

    def det_cb(self, det_msg, scan_msg, odom_msg, img_msg):
        try:
            if not det_msg.status:
                rospy.logwarn_throttle(5, "Received empty DiagnosticArray for detections.")
                return
            detections = json.loads(det_msg.status[0].message)
            frame = self.bridge.compressed_imgmsg_to_cv2(img_msg, "bgr8")
            img_h, img_w = frame.shape[:2]
        except Exception as e:
            rospy.logwarn(f"Failed to parse messages: {e}")
            return
            
        time_sec = odom_msg.header.stamp.to_sec()
        scan_snap = scan_msg
        odom_snap = odom_msg

        # 1. Predict
        for t_id, kf in self.kfs.items():
            kf.predict(time_sec)
            
        # 2. Associate Data
        trk_list = list(self.kfs.values())
        trk_ids = list(self.kfs.keys())
        matches, unmatched_dets, _ = associate_detections_to_trackers(detections, trk_list, IOU_MATCH_THRESHOLD)
        
        det_reasons = ["Unknown"] * len(detections)

        # 3. Update Existing Tracks
        for (d, t) in matches:
            det = detections[d]
            kf = self.kfs[trk_ids[t]]
            z_meas, err = self.process_3d_state(det, scan_snap, odom_snap, img_w)
            
            if z_meas is not None:
                dist = kf.mahalanobis_distance(z_meas)
                if dist < self.chi2_gate_threshold:
                    kf.update(z_meas, score=det['score'], label=det['label'], bbox=det['bbox'])
                    if kf.status == 'tentative' and kf.hits >= CONFIRMATION_HITS:
                        kf.status = 'confirmed'
                        det_reasons[d] = "Tracked (Just Confirmed)"
                    elif kf.status == 'confirmed':
                        det_reasons[d] = "Tracked"
                    else:
                        det_reasons[d] = f"KF Tentative (Hit {kf.hits}/{CONFIRMATION_HITS})"
                else:
                    det_reasons[d] = "Rejected: KF"
            else:
                det_reasons[d] = f"Rejected: {err}"

        # 4. Create New Tracks
        for d in unmatched_dets:
            det = detections[d]
            z_meas, err = self.process_3d_state(det, scan_snap, odom_snap, img_w)
            if z_meas is not None:
                t_id = self.next_id
                self.next_id += 1
                self.kfs[t_id] = ObjectKalmanFilter(t_id, z_meas[0], z_meas[1], time_sec, MAX_OBJECT_SPEED, det['score'], label=det['label'], bbox=det['bbox'])
                det_reasons[d] = f"New Track (Tentative, Hit 1/{CONFIRMATION_HITS})"
            else:
                det_reasons[d] = f"Rejected: {err}"

        # 5. Clean up stale tracks
        dead_ids = []
        for tid, kf in self.kfs.items():
            if (kf.status == 'tentative' and kf.misses > 2) or \
               (kf.status == 'tentative' and kf.age > 10 and kf.hits < CONFIRMATION_HITS) or \
               (kf.status == 'confirmed' and kf.misses > TRACKING_COAST_FRAMES):
                dead_ids.append(tid)
        for tid in dead_ids:
            del self.kfs[tid]

        # 6. Publish
        self.publish_outputs(time_sec, odom_snap, detections, det_reasons)

        # 7. Draw tracked objects on the image and publish
        for t_id, kf in self.kfs.items():
            if kf.status == 'confirmed' and kf.misses == 0 and kf.bbox is not None:
                xmin, ymin, xmax, ymax = [int(c) for c in kf.bbox]
                
                # Draw Bbox & Label onto frame (Yellow for tracked objects)
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 255), 2)
                label_str = f"{kf.label}_{t_id}"
                cv2.putText(frame, label_str, (xmin, max(ymin - 10, 10)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3, cv2.LINE_AA)                
                cv2.putText(frame, label_str, (xmin, max(ymin - 10, 10)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

        # Draw Bag timestamp and Universal System Time
        H, W = frame.shape[:2]
        sys_time = time.time() % 100000
        cv2.rectangle(frame, (5, H - 65), (210, H - 5), (255, 255, 255), -1)
        cv2.putText(frame, f"Bag: {time_sec % 10000:.3f}s", (10, H - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Sys: {sys_time:.3f}", (10, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

        # Publish Annotated Image
        annotated_msg = self.bridge.cv2_to_compressed_imgmsg(frame)
        annotated_msg.header = img_msg.header
        self.annotated_img_pub.publish(annotated_msg)

    def publish_outputs(self, time_sec, odom_snap, detections=None, det_reasons=None):
        tracked_obj_list = []
        marker_array = MarkerArray()
        grid_z = float(odom_snap.pose.pose.position.z)
        published_ids = set()
        
        for t_id, kf in self.kfs.items():
            if kf.status != 'confirmed' or kf.misses > 0: 
                continue
                
            published_ids.add(t_id)
            state = kf.state
            
            # Publish JSON array of tracked objects
            tracked_obj_list.append({
                "id": t_id,
                "label": kf.label,
                "score": float(kf.latest_score),
                "x": float(state[0]),
                "y": float(state[1]),
                "vx": float(state[2]),
                "vy": float(state[3]),
                "is_dynamic": kf.is_dynamic,
                "bbox": [float(c) for c in kf.bbox] if kf.bbox is not None else None
            })
            
            # RViz Marker (Cylinder)
            m_cyl = Marker()
            m_cyl.header.frame_id = "odom"
            m_cyl.header.stamp = odom_snap.header.stamp
            m_cyl.ns = "tracked_centroids"
            m_cyl.id = int(t_id) * 3
            m_cyl.type = Marker.CYLINDER
            m_cyl.action = Marker.ADD
            m_cyl.pose.position.x, m_cyl.pose.position.y, m_cyl.pose.position.z = float(state[0]), float(state[1]), grid_z + 0.01
            m_cyl.pose.orientation.w = 1.0
            m_cyl.scale.x, m_cyl.scale.y, m_cyl.scale.z = 0.4, 0.4, 0.02
            m_cyl.color.a = 0.6
            m_cyl.color.r, m_cyl.color.g, m_cyl.color.b = (1.0, 0.0, 1.0) if kf.is_dynamic else (0.0, 1.0, 0.0)
            marker_array.markers.append(m_cyl)
            
            # RViz Marker (Text)
            m_text = Marker()
            m_text.header.frame_id = "odom"
            m_text.header.stamp = odom_snap.header.stamp
            m_text.ns = "tracked_labels"
            m_text.id = int(t_id) * 3 + 1
            m_text.type = Marker.TEXT_VIEW_FACING
            m_text.action = Marker.ADD
            m_text.pose.position.x, m_text.pose.position.y, m_text.pose.position.z = float(state[0]), float(state[1]), grid_z + 0.2
            m_text.pose.orientation.w = 1.0
            m_text.scale.z = 0.3
            m_text.color.r, m_text.color.g, m_text.color.b, m_text.color.a = 1.0, 1.0, 1.0, 1.0
            m_text.text = f"{kf.label}_{int(t_id)}"
            marker_array.markers.append(m_text)
            
            # RViz Marker (Velocity Arrow)
            if kf.is_dynamic:
                m_arr = Marker()
                m_arr.header.frame_id = "odom"
                m_arr.header.stamp = odom_snap.header.stamp
                m_arr.ns = "tracked_velocities"
                m_arr.id = int(t_id) * 3 + 2
                m_arr.type = Marker.ARROW
                m_arr.action = Marker.ADD
                m_arr.pose.orientation.w = 1.0
                p1, p2 = Point(), Point()
                p1.x, p1.y, p1.z = float(state[0]), float(state[1]), grid_z + 0.02
                p2.x, p2.y, p2.z = float(state[0] + state[2]*1.5), float(state[1] + state[3]*1.5), grid_z + 0.02
                m_arr.points = [p1, p2]
                m_arr.scale.x, m_arr.scale.y, m_arr.scale.z = 0.1, 0.2, 0.2
                m_arr.color.r, m_arr.color.g, m_arr.color.b, m_arr.color.a = 1.0, 0.0, 1.0, 0.8
                marker_array.markers.append(m_arr)

        # Cleanup dead markers
        disappeared_ids = self.previous_object_ids.difference(published_ids)
        for t_id in disappeared_ids:
            for offset, ns in enumerate(["tracked_centroids", "tracked_labels", "tracked_velocities"]):
                m_del = Marker()
                m_del.header.frame_id = "odom"
                m_del.header.stamp = odom_snap.header.stamp
                m_del.ns = ns
                m_del.id = int(t_id) * 3 + offset
                m_del.action = Marker.DELETE
                marker_array.markers.append(m_del)
        
        self.previous_object_ids = published_ids
        
        # Publish tracked objects as a DiagnosticArray with a header for synchronization
        obj_diag_array = DiagnosticArray()
        obj_diag_array.header = odom_snap.header
        status = DiagnosticStatus()
        status.name = "TrackedObjects"
        status.message = json.dumps(tracked_obj_list)
        obj_diag_array.status.append(status)
        
        if detections is not None and det_reasons is not None:
            debug_list = []
            for i, (det, reason) in enumerate(zip(detections, det_reasons)):
                debug_list.append({
                    "label": det.get('label', 'unknown'),
                    "index": i,
                    "reason": reason
                })
            status_debug = DiagnosticStatus()
            status_debug.name = "DetectionDebug"
            status_debug.message = json.dumps(debug_list)
            obj_diag_array.status.append(status_debug)
            
        self.obj_pub.publish(obj_diag_array)

        self.marker_pub.publish(marker_array)

if __name__ == '__main__':
    try:
        TrackingNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass