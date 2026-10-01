#!/path_to_your_env/bin/python

import rospy
import math
import numpy as np

from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import Twist, Point
from visualization_msgs.msg import Marker

class PurePursuitTrackerNode:
    def __init__(self):
        rospy.init_node('pure_pursuit_tracker', anonymous=True)
        
        # --- Tunable Parameters ---
        self.robot_model = rospy.get_param("~robot_model", "holonomic")        # 'unicycle' or 'holonomic'
        self.lookahead_distance = rospy.get_param("~lookahead_distance", 0.8) # meters
        self.max_linear_vel = rospy.get_param("~max_linear_vel", 1.0)         # m/s
        self.max_angular_vel = rospy.get_param("~max_angular_vel", 1.5)       # rad/s
        self.k_theta = rospy.get_param("~k_theta", 2.0)                       # Proportional gain for steering
        
        self.current_path = None
        self.current_pose = None
        self.current_stamp = None
        
        # --- Publishers & Subscribers ---
        self.cmd_pub = rospy.Publisher('/g2nav/cmd_vel', Twist, queue_size=1)
        self.marker_pub = rospy.Publisher('/cmd_vel_viz', Marker, queue_size=10)
        rospy.Subscriber('/plan_path', Path, self.path_cb)
        rospy.Subscriber('/odom', Odometry, self.odom_cb)
        
        # Run control loop at 20Hz
        rospy.Timer(rospy.Duration(0.05), self.control_loop)
        
        rospy.loginfo("Pure Pursuit Path Tracker Initialized.")

    def path_cb(self, msg):
        # The path is in the 'odom' frame as published by planning.py
        self.current_path = msg
        
    def odom_cb(self, msg):
        self.current_pose = msg.pose.pose
        self.current_stamp = msg.header.stamp

    def control_loop(self, event):
        if self.current_path is None or self.current_pose is None or self.current_stamp is None or len(self.current_path.poses) == 0:
            self.stop_robot()
            return
            
        # 1. Get current robot state
        rx = self.current_pose.position.x
        ry = self.current_pose.position.y
        rq = self.current_pose.orientation
        
        siny_cosp = 2 * (rq.w * rq.z + rq.x * rq.y)
        cosy_cosp = 1 - 2 * (rq.y**2 + rq.z**2)
        current_yaw = math.atan2(siny_cosp, cosy_cosp)

        # 2. Find the lookahead point on the path
        target_pt = None
        
        # --- Robust Pure Pursuit: Find the closest point first ---
        closest_idx = 0
        min_dist = float('inf')
        for i, pose_stamped in enumerate(self.current_path.poses):
            px = pose_stamped.pose.position.x
            py = pose_stamped.pose.position.y
            dist = math.hypot(px - rx, py - ry)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
                
        # Search forward from the closest point to find the lookahead
        for i in range(closest_idx, len(self.current_path.poses)):
            px = self.current_path.poses[i].pose.position.x
            py = self.current_path.poses[i].pose.position.y
            dist = math.hypot(px - rx, py - ry)
            
            if dist >= self.lookahead_distance:
                target_pt = (px, py)
                break
                
        # If the whole path is shorter than lookahead, just aim for the last point
        if target_pt is None:
            last_pose = self.current_path.poses[-1].pose
            target_pt = (last_pose.position.x, last_pose.position.y)
            
        # 3. Calculate Control Commands based on Robot Model
        dx = target_pt[0] - rx
        dy = target_pt[1] - ry
        target_yaw = math.atan2(dy, dx)
        
        # Angle error mapped to [-pi, pi] for both models to turn towards the path
        yaw_error = target_yaw - current_yaw
        yaw_error = math.atan2(math.sin(yaw_error), math.cos(yaw_error))
        
        cmd = Twist()
        cmd.angular.z = np.clip(self.k_theta * yaw_error, -self.max_angular_vel, self.max_angular_vel)
        
        if self.robot_model == "unicycle":
            # Unicycle: Can only move forward. Slow down when making sharp turns.
            cmd.linear.x = self.max_linear_vel * max(0.1, 1.0 - abs(yaw_error)/math.pi)
            cmd.linear.y = 0.0
            
        elif self.robot_model == "holonomic":
            # Holonomic: Can translate in x and y. 
            dist = math.hypot(dx, dy)
            if dist > 0.01:
                # Global velocity vector aiming directly at the lookahead point
                v_x_global = self.max_linear_vel * (dx / dist)
                v_y_global = self.max_linear_vel * (dy / dist)
                
                # Rotate the global velocity vector into the robot's local frame (base_link)
                cmd.linear.x = v_x_global * math.cos(current_yaw) + v_y_global * math.sin(current_yaw)
                cmd.linear.y = -v_x_global * math.sin(current_yaw) + v_y_global * math.cos(current_yaw)
        else:
            rospy.logwarn_throttle(1.0, f"Unknown robot_model: {self.robot_model}. Using 0 vel.")
        
        self.cmd_pub.publish(cmd)

        # 4. Visualize cmd_vel as an arrow in RViz
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.header.stamp = self.current_stamp
        marker.ns = "cmd_vel"
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        
        v_mag = math.hypot(cmd.linear.x, cmd.linear.y)
        if v_mag > 0.01:
            p_start = Point()
            p_start.x = rx
            p_start.y = ry
            p_start.z = self.current_pose.position.z + 0.2  # Hover slightly above ground
            
            p_end = Point()
            # Twist is in the local frame. Rotate to the global odom frame for visualization.
            viz_scale = 1.0  # 1 m/s = 1 meter long arrow
            v_global_x = (cmd.linear.x * math.cos(current_yaw) - cmd.linear.y * math.sin(current_yaw)) * viz_scale
            v_global_y = (cmd.linear.x * math.sin(current_yaw) + cmd.linear.y * math.cos(current_yaw)) * viz_scale
            
            p_end.x = rx + v_global_x
            p_end.y = ry + v_global_y
            p_end.z = p_start.z
            
            marker.points = [p_start, p_end]
            marker.scale.x = 0.1   # Shaft diameter
            marker.scale.y = 0.2   # Head diameter
            marker.scale.z = 0.2   # Head length
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 1.0   # Cyan color
            marker.color.a = 1.0
        else:
            marker.action = Marker.DELETE
            
        self.marker_pub.publish(marker)
        
        # 5. Visualize angular velocity as a curved arc
        marker_ang = Marker()
        marker_ang.header.frame_id = "odom"
        marker_ang.header.stamp = self.current_stamp
        marker_ang.ns = "cmd_vel_ang"
        marker_ang.id = 1
        marker_ang.type = Marker.LINE_STRIP
        marker_ang.action = Marker.ADD
        marker_ang.pose.orientation.w = 1.0
        marker_ang.scale.x = 0.05  # Line width
        marker_ang.color.r = 1.0
        marker_ang.color.g = 1.0
        marker_ang.color.b = 0.0   # Yellow color
        marker_ang.color.a = 1.0
        
        if abs(cmd.angular.z) > 0.01:
            radius = 0.4
            num_pts = 10
            # Draw an arc indicating rotation direction and magnitude
            for i in range(num_pts):
                # Angle goes from 0 to angular.z (representing 1 second of rotation)
                angle = (i / float(num_pts - 1)) * cmd.angular.z
                lx = radius * math.cos(angle)
                ly = radius * math.sin(angle)
                
                gx = rx + lx * math.cos(current_yaw) - ly * math.sin(current_yaw)
                gy = ry + lx * math.sin(current_yaw) + ly * math.cos(current_yaw)
                marker_ang.points.append(Point(gx, gy, self.current_pose.position.z + 0.2))
                
            # Add a small arrowhead manually to the end of the line strip
            last_angle = cmd.angular.z
            arrow_len = 0.1
            # The tangent direction of the circle at the end
            tangent_dir = last_angle + (math.pi / 2.0 if cmd.angular.z > 0 else -math.pi / 2.0)
            
            def local_to_global(loc_x, loc_y):
                glob_x = rx + loc_x * math.cos(current_yaw) - loc_y * math.sin(current_yaw)
                glob_y = ry + loc_x * math.sin(current_yaw) + loc_y * math.cos(current_yaw)
                return Point(glob_x, glob_y, self.current_pose.position.z + 0.2)
            
            # Draw Left Wing -> Tip -> Right Wing to form the arrowhead
            marker_ang.points.append(local_to_global(radius * math.cos(last_angle) + arrow_len * math.cos(tangent_dir + math.pi * 0.8), radius * math.sin(last_angle) + arrow_len * math.sin(tangent_dir + math.pi * 0.8)))
            marker_ang.points.append(local_to_global(radius * math.cos(last_angle), radius * math.sin(last_angle)))
            marker_ang.points.append(local_to_global(radius * math.cos(last_angle) + arrow_len * math.cos(tangent_dir - math.pi * 0.8), radius * math.sin(last_angle) + arrow_len * math.sin(tangent_dir - math.pi * 0.8)))
        else:
            marker_ang.action = Marker.DELETE
            
        self.marker_pub.publish(marker_ang)
        
    def stop_robot(self):
        self.cmd_pub.publish(Twist())
        
        # Clear the arrow when the robot stops
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.header.stamp = self.current_stamp if self.current_stamp else rospy.Time(0)
        marker.ns = "cmd_vel"
        marker.id = 0
        marker.action = Marker.DELETE
        marker.pose.orientation.w = 1.0
        self.marker_pub.publish(marker)

        # Clear the angular arc
        marker_ang = Marker()
        marker_ang.header.frame_id = "odom"
        marker_ang.header.stamp = self.current_stamp if self.current_stamp else rospy.Time(0)
        marker_ang.ns = "cmd_vel_ang"
        marker_ang.id = 1
        marker_ang.action = Marker.DELETE
        marker_ang.pose.orientation.w = 1.0
        self.marker_pub.publish(marker_ang)

if __name__ == '__main__':
    try:
        PurePursuitTrackerNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass