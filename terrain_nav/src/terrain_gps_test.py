#! /usr/bin/env python3

import rospy
import math
import message_filters
from geographiclib.geodesic import Geodesic
from geometry_msgs.msg import Twist, QuaternionStamped, Vector3Stamped
from terrain_characterizer.msg import ErrorNav
from dynamic_reconfigure.server import Server
from scout_msgs.msg import ScoutStatus
import subprocess, shlex, psutil


def callback(location, orientation, errors, status):
    point_a = {
        # "long" : 59.91,
        # "lat" : 10.72
        "long" : location.vector.x,
        "lat" : location.vector.y
    }

    errors = get_errors(errors)
    robot_yaw = get_yaw(orientation.quaternion)
    geodata = get_geodata(point_a, point_b)
    azimuth = geodata["azi1"]
    robot_b_heading = azimuth - (robot_yaw)

    print("____________________________________________________________________")
    print(f"azimuth: {azimuth}")
    print(f"robot_yaw: {robot_yaw}")
    print(f"heading: {robot_b_heading}")

    #adjust_twist(robot_b_heading)
    cost_twist(robot_b_heading, errors[0], errors[1])

def get_errors(msg):
    left = msg.left.MSE/msg.left.indices
    middle = msg.middle.MSE/msg.middle.indices
    right = msg.right.MSE/msg.right.indices

    left_bias = left / middle
    right_bias = right / middle

    return left_bias, right_bias

def get_yaw(q):
    
    robot_yaw_rad = math.atan2(2 * ((q.x*q.y) + (q.w * q.z)),q.w**2 + q.x**2 - q.y**2 - q.z**2)
    robot_yaw_deg = math.degrees(robot_yaw_rad)

    #make it so that turning right is positive degrees, turning left is negative, just like GPS
    if robot_yaw_deg > 0:
        return -robot_yaw_deg
    elif robot_yaw_deg < 0:
        return -robot_yaw_deg


def get_geodata(point_a, point_b):
    
    data = Geodesic.WGS84.Inverse(point_a["lat"], point_a["long"], point_b["lat"], point_b["long"])
    
    return data

def cost_twist(heading, left_bias, right_bias):
    twist.linear.x = 0.3 
    
    scale = 1
    cost_heading = heading / 100 * scale    # if negative, need to turn left
                                            # if positive, need to turn right
    cost_left = left_bias + cost_heading
    cost_right = right_bias - cost_heading

    print(f"biases: left - {round(left_bias,3)}, right - {round(right_bias,3)}")
    print(f"costs: left - {round(cost_left,3)}, right - {round(cost_right,3)}")

    if cost_left > 1 and cost_right > 1:
        twist.angular.z = 0
        rospy.loginfo("M")
    
    elif cost_left < cost_right:
        twist.angular.z = -0.3
        rospy.loginfo("L")
    elif cost_right < cost_left:
        twist.angular.z = 0.3
        rospy.loginfo("R")


if __name__ == '__main__':
    rospy.init_node('terrain_gps_test')
    #59.9091011614321, 10.720408935369424
    #59.90872, 10.71901
    point_b = {
        "long" : 59.9091011614321, 
        "lat" : 10.720408935369424
    }

    gps_sub = message_filters.Subscriber('/filter/positionlla', Vector3Stamped)
    imu_sub = message_filters.Subscriber('/filter/quaternion', QuaternionStamped)
    error_sub = message_filters.Subscriber('/terrain_nav/errors', ErrorNav)
    status_sub = message_filters.Subscriber('/scout_status', ScoutStatus)

    ts = message_filters.ApproximateTimeSynchronizer([gps_sub, imu_sub, error_sub], queue_size=10, slop=0.1, allow_headerless=True)
    ts.registerCallback(callback)

    pub = rospy.Publisher('/cmd_vel', Twist, queue_size=2) 
    rospy.Rate(2) 
    twist = Twist()

    #srv = Server(TutorialsConfig, callback)
    
    rospy.spin()