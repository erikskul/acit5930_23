#!/usr/bin/env python3
# rosbag record /odom /scout_status /BMS_status /tf /tf_static /camera/depth/color/points /camera/color/image_raw/compressed /camera/color/camera_info /filter/free_acceleration /filter/positionlla /filter/quaternion /filter/twist /filter/velocity /gnss /imu/acceleration /imu/angular_velocity /imu/data /imu/dq /imu/dv /imu/mag /imu/time_ref
import rospy
import subprocess, shlex, psutil
from geometry_msgs.msg import Twist
from scout_msgs.msg import ScoutStatus

class DataBagger:
    def __init__(self):
        rospy.init_node('data_bagger')
        
        # Subscribers
        rospy.Subscriber('/scout_status', ScoutStatus, self.status_callback, queue_size=1)

        # Publishers
        self.vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.topics = [ '/odom',
                        '/scout_status',
                        '/BMS_status',
                        '/tf',
                        '/tf_static',
                        '/camera/depth/color/points',
                        '/camera/color/image_raw/compressed',
                        '/camera/depth/image_rect_raw',
                        '/camera/color/camera_info',
                        '/camera/depth/camera_info',
                        '/filter/free_acceleration',
                        '/filter/positionlla',
                        '/filter/quaternion',
                        '/filter/twist',
                        '/filter/velocity',
                        '/gnss',
                        '/imu/acceleration',
                        '/imu/angular_velocity',
                        '/imu/data',
                        '/imu/dq',
                        '/imu/dv',
                        '/imu/mag',
                        '/imu/time_ref',]

        # Initialize instance variables
        self.is_recording = False
        self.duration = rospy.Duration(3)  # default duration of 3 seconds, results in around 2.2sec long rosbags

    def status_callback(self, status):
        # Check if any of the toggles have been flipped (value = 2)
        if status.left_middle_toggle == 0:
            if status.left_toggle == 2:
                self.execute_sequence('left')
            elif status.right_toggle == 2:
                self.execute_sequence('right')
            elif status.right_middle_toggle == 2:
                self.execute_sequence('middle')

    def execute_sequence(self, direction):
        # Start rosbag record with filename including direction and counter

        # Move robot in the chosen direction
        twist = Twist()
        if direction == 'left':
            twist.linear.x = 0.3 # move straight at 0.3 m/s
            twist.angular.z = 0.3  # turn left at 0.5 rad/s
        elif direction == 'right':
            twist.linear.x = 0.3 # move straight at 0.3 m/s
            twist.angular.z = -0.3  # turn right at 0.5 rad/s
        elif direction == 'middle':
            twist.linear.x = 0.3  # move straight at 0.3 m/s
            twist.angular.z = 0.0  # don't turn
        self.vel_pub.publish(twist)

if __name__ == '__main__':
    try:
        databagger = DataBagger()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
