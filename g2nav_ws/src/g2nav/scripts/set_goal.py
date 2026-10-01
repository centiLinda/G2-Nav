#!/path_to_your_env/bin/python

import rospy
import numpy as np
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PointStamped
from scipy.spatial.transform import Rotation

GOAL_DISTANCE = 30.0            # Distance in meters straight ahead for the global goal
INITIALIZATION_DELAY = 10.0      # Seconds to wait before setting the fixed global goal
ALWAYS_FORWARD = True          # If True, continuously updates goal to be straight ahead of current pose

class SetGoalNode:
    def __init__(self):
        rospy.init_node('set_goal_node', anonymous=True)
        
        self.start_time = None
        self.global_goal = None
        self.locked = False
        
        self.goal_pub = rospy.Publisher('/global_goal', PointStamped, queue_size=1)
        rospy.Subscriber('/odom', Odometry, self.odom_cb)
        
        rospy.loginfo("Set Goal Node Initialized.")
        
    def odom_cb(self, msg):
        translation = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        ])
        rotation_quat = [
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        ]
        R_init = Rotation.from_quat(rotation_quat).as_matrix()
        forward_vec = R_init @ np.array([1.0, 0.0, 0.0])
        
        if self.start_time is None:
            self.start_time = msg.header.stamp.to_sec()
            
        current_time = msg.header.stamp.to_sec()
        
        goal_pt = PointStamped()
        goal_pt.header.stamp = msg.header.stamp
        goal_pt.header.frame_id = msg.header.frame_id
        
        # Lock the fixed goal once the delay is reached
        if not ALWAYS_FORWARD and not self.locked and (current_time - self.start_time) >= INITIALIZATION_DELAY:
            self.global_goal = translation + GOAL_DISTANCE * forward_vec
            self.locked = True
            rospy.loginfo(f"Locked fixed global goal at x: {self.global_goal[0]:.2f}, y: {self.global_goal[1]:.2f} after {INITIALIZATION_DELAY}s delay.")
            
        target_goal = self.global_goal if self.locked else (translation + GOAL_DISTANCE * forward_vec)
            
        goal_pt.point.x = target_goal[0]
        goal_pt.point.y = target_goal[1]
        goal_pt.point.z = target_goal[2]
        
        self.goal_pub.publish(goal_pt)

if __name__ == '__main__':
    try:
        SetGoalNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass