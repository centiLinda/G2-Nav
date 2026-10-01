#!/path_to_your_env/bin/python

import rospy
import numpy as np
import cv2
import math
import time
from cv_bridge import CvBridge
import message_filters

from scipy.interpolate import splprep, splev
from sensor_msgs.msg import Image, CompressedImage
from nav_msgs.msg import Odometry, MapMetaData, Path
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation

# --- Camera Projection ---
CAM_INTRINSICS = np.array([
    [608.115906, 0.0,        639.186401],
    [0.0,        607.871704, 363.069061],
    [0.0,        0.0,        1.0]
])
CAM_HEIGHT = 0.70                # Estimated camera height in meters

class GradientPlannerNode:
    def __init__(self):
        rospy.init_node('gradient_planner_node', anonymous=True)
        self.bridge = CvBridge()
        
        self.meta_data = None
        
        # --- Publishers ---
        self.path_pub = rospy.Publisher('/plan_path', Path, queue_size=1)
        self.img_pub = rospy.Publisher('/plan_viz', Image, queue_size=1)
        self.ego_img_pub = rospy.Publisher('/plan_ego_viz/compressed', CompressedImage, queue_size=1)
        
        # --- Subscribers ---
        # Metadata is latched, so a standard subscriber is fine
        rospy.Subscriber('/costmap_metadata', MapMetaData, self.meta_cb)
        
        # Synchronize continuous math costmap, visualization costmap, odometry, and raw image
        data_sub = message_filters.Subscriber('/costmap_data', Image, queue_size=20)
        viz_sub = message_filters.Subscriber('/costmap_safe_viz', Image, queue_size=20)
        odom_sub = message_filters.Subscriber('/odom', Odometry, queue_size=500)
        img_sub = message_filters.Subscriber('/image_raw/compressed', CompressedImage, queue_size=20)
        
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [data_sub, viz_sub, odom_sub, img_sub], queue_size=100, slop=0.2
        )
        self.ts.registerCallback(self.plan_callback)
        
        rospy.loginfo("Gradient Planner Node Initialized.")
        
    def meta_cb(self, msg):
        self.meta_data = msg
        
    def plan_callback(self, data_msg, viz_msg, odom_msg, img_msg):
        if self.meta_data is None:
            rospy.logwarn_throttle(2.0, "Waiting for costmap metadata...")
            return
            
        try:
            # 1. Decode math image, visualization image, and egocentric image
            cost_map = self.bridge.imgmsg_to_cv2(data_msg, desired_encoding="32FC1")
            viz_img = self.bridge.imgmsg_to_cv2(viz_msg, desired_encoding="bgr8")
            
            try:
                ego_img = self.bridge.compressed_imgmsg_to_cv2(img_msg, "bgr8")
                img_h, img_w = ego_img.shape[:2]
                
                # Dynamically scale intrinsics to match actual image resolution
                scale_x = img_w / 1280.0
                scale_y = img_h / 720.0
                fx = CAM_INTRINSICS[0, 0] * scale_x
                cx = CAM_INTRINSICS[0, 2] * scale_x
                fy = CAM_INTRINSICS[1, 1] * scale_y
                cy = CAM_INTRINSICS[1, 2] * scale_y
            except Exception as e:
                rospy.logwarn(f"Failed to decode compressed ego image: {e}")
                ego_img = None

            res = self.meta_data.resolution
            h, w = cost_map.shape
            
            # 2. Get map origin and orientation in odom frame
            ox = self.meta_data.origin.position.x
            oy = self.meta_data.origin.position.y
            oz = self.meta_data.origin.position.z
            o_odom = np.array([ox, oy, oz])
            
            oq = self.meta_data.origin.orientation
            rot = Rotation.from_quat([oq.x, oq.y, oq.z, oq.w])
            
            # 3. Get robot position and convert to map pixel coordinates
            rx = odom_msg.pose.pose.position.x
            ry = odom_msg.pose.pose.position.y
            rz = odom_msg.pose.pose.position.z
            p_odom = np.array([rx, ry, rz])
            
            rq = odom_msg.pose.pose.orientation
            R_ego = Rotation.from_quat([rq.x, rq.y, rq.z, rq.w])
            
            # Map transform: P_map = R^-1 * (P_odom - Origin)
            p_map = rot.inv().apply(p_odom - o_odom)
            start_px = p_map[0] / res
            start_py = p_map[1] / res
            
            # 4. Perform Gradient Descent
            curr_x, curr_y = float(start_px), float(start_py)
            path_pts = [(curr_x, curr_y)]
            
            step_size_px = 1.0  # 1 pixel = 0.15m per step
            max_steps = 150
            
            for _ in range(max_steps):
                ix, iy = int(round(curr_x)), int(round(curr_y))
                
                # Check bounds
                if ix < 1 or ix >= w-1 or iy < 1 or iy >= h-1:
                    break
                
                # Gradients (finite difference)
                dx = (cost_map[iy, ix+1] - cost_map[iy, ix-1]) / 2.0
                dy = (cost_map[iy+1, ix] - cost_map[iy-1, ix]) / 2.0
                
                cost_val = cost_map[iy, ix]
                desc_x = -dx
                desc_y = -dy
                
                # Apply tangential swirl only if cost is elevated (near an obstacle)
                if cost_val > 2.0:
                    swirl_weight = 1.2  # Multiplier for the sideways sliding push
                    desc_x += swirl_weight * dy
                    desc_y -= swirl_weight * dx
                                    
                desc_mag = math.hypot(desc_x, desc_y)
                if desc_mag < 1e-3:
                    break  # Reached the local minimum valley
                
                curr_x += (desc_x / desc_mag) * step_size_px
                curr_y += (desc_y / desc_mag) * step_size_px
                
                path_pts.append((curr_x, curr_y))
                
            # --- Apply B-Spline Smoothing ---
            if len(path_pts) > 3:
                # Filter points that are too close to each other (can cause splprep to fail)
                filtered_pts = [path_pts[0]]
                for pt in path_pts[1:]:
                    if math.hypot(pt[0] - filtered_pts[-1][0], pt[1] - filtered_pts[-1][1]) > 0.5:
                        filtered_pts.append(pt)
                
                # Need at least 4 points for cubic spline (k=3)
                if len(filtered_pts) > 3:
                    pts_arr = np.array(filtered_pts)
                    # s is the smoothing condition. Higher = smoother but deviates more from original points
                    tck, u = splprep([pts_arr[:,0], pts_arr[:,1]], s=len(filtered_pts)*0.5, k=3)
                    
                    # Evaluate spline at a fixed number of points to build a continuous curve
                    u_new = np.linspace(0, 1.0, len(filtered_pts))
                    smooth_x, smooth_y = splev(u_new, tck)
                    # Force the start of the spline to remain exactly at the robot's current position
                    smooth_x[0] = filtered_pts[0][0]
                    smooth_y[0] = filtered_pts[0][1]
                    path_pts = list(zip(smooth_x, smooth_y))

            # 5. Build and publish nav_msgs/Path for RViz
            path_msg = Path()
            path_msg.header.stamp = data_msg.header.stamp
            path_msg.header.frame_id = "odom"
            path_pixels = []
            
            for (px, py) in path_pts:
                # Transform back to odom frame
                pt_map = np.array([px * res, py * res, 0.0])
                pt_odom = rot.apply(pt_map) + o_odom
                
                pose = PoseStamped()
                pose.header = path_msg.header
                pose.pose.position.x = pt_odom[0]
                pose.pose.position.y = pt_odom[1]
                pose.pose.position.z = pt_odom[2]
                pose.pose.orientation.w = 1.0
                path_msg.poses.append(pose)
                
                if ego_img is not None:
                    # Project to egocentric camera
                    pt_ego = R_ego.inv().apply(pt_odom - p_odom)
                    x_local = pt_ego[0]
                    y_local = pt_ego[1]
                    
                    if x_local > 0.1:  # Only project points directly in front of the robot
                        Z_cam = x_local
                        Y_cam = CAM_HEIGHT
                        
                        u = int(fx * (-y_local / Z_cam) + cx)
                        v = int(fy * (Y_cam / Z_cam) + cy)
                        
                        path_pixels.append([u, v])
                
            self.path_pub.publish(path_msg)
            
            # --- Draw Path on Egocentric Image ---
            if ego_img is not None and len(path_pixels) > 1:
                pts = np.array(path_pixels, np.int32).reshape((-1, 1, 2))
                cv2.polylines(ego_img, [pts], isClosed=False, color=(255, 255, 255), thickness=6) # Outline
                cv2.polylines(ego_img, [pts], isClosed=False, color=(0, 0, 255), thickness=3)     # Core line
                
                ego_img_out = self.bridge.cv2_to_compressed_imgmsg(ego_img)
                ego_img_out.header = img_msg.header
                self.ego_img_pub.publish(ego_img_out)
            
            # 6. Visualize path on Image
            # Draw Path in Red with a white outline for high contrast
            for i in range(1, len(path_pts)):
                # Transform planner coordinates (x: fwd, y: left) to viz image coordinates (x: left, y: up)
                # Using round() instead of int() fixes the 2-pixel truncation drift
                pt1_x, pt1_y = int(round(w - 1 - path_pts[i-1][1])), int(round(h - 1 - path_pts[i-1][0]))
                pt2_x, pt2_y = int(round(w - 1 - path_pts[i][1])), int(round(h - 1 - path_pts[i][0]))
                cv2.line(viz_img, (pt1_x, pt1_y), (pt2_x, pt2_y), (255, 255, 255), 4) # Outline
                cv2.line(viz_img, (pt1_x, pt1_y), (pt2_x, pt2_y), (0, 0, 255), 2)     # Core line
            
            # Draw System time overlay
            sys_time = time.time() % 100000
            # We redraw the text bounding box to overwrite the old Sys time from the previous node
            cv2.rectangle(viz_img, (0, h - 15), (110, h), (255, 255, 255), -1)
            cv2.putText(viz_img, f"Sys: {sys_time:.3f}", (5, h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
                  
            img_out = self.bridge.cv2_to_imgmsg(viz_img, encoding="bgr8")
            img_out.header = data_msg.header
            self.img_pub.publish(img_out)
            
        except Exception as e:
            rospy.logerr(f"Error in planning callback: {e}")

if __name__ == '__main__':
    try:
        GradientPlannerNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass