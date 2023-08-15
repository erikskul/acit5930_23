#! /usr/bin/env python3

import rospy
import math
import message_filters
from geographiclib.geodesic import Geodesic
from geometry_msgs.msg import Twist, QuaternionStamped, Vector3Stamped
from terrain_characterizer.msg import ErrorNav
from dynamic_reconfigure.server import Server
import os
import torch
from torch import FloatTensor
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.io import read_image
from torchvision import transforms
from PIL import Image
import numpy as np
from scout_msgs.msg import ScoutStatus
import subprocess, shlex, psutil
# rosbag record /odom /scout_status /BMS_status /tf /tf_static /camera/depth/color/points /camera/color/image_raw/compressed /camera/color/camera_info /filter/free_acceleration /filter/positionlla /filter/quaternion /filter/twist /filter/velocity /gnss /imu/acceleration /imu/angular_velocity /imu/data /imu/dq /imu/dv /imu/mag /imu/time_ref

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU()
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class TCNetRes(nn.Module):
    def __init__(self):
        super(TCNetRes, self).__init__()
        self.layer1 = self._make_layer(1, 16, stride=1)
        self.layer2 = self._make_layer(16, 32, stride=2)
        self.layer3 = self._make_layer(32, 64, stride=2)
        self.layer4 = self._make_layer(64, 128, stride=2)
        
        self.fc1 = nn.Linear(512, 128)
        self.fc_action = nn.Linear(1, 16)
        self.fc2 = nn.Linear(128 + 16, 64)
        self.fc3 = nn.Linear(64, 1)

    def _make_layer(self, in_channels, out_channels, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        return nn.Sequential(*layers)
        
    def forward(self, x, action):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        action = F.relu(self.fc_action(action.view(-1, 1)))
        x = torch.cat((x, action), dim=1)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def predict(msg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = TCNetRes()
    net.load_state_dict(torch.load('res_rot_dist_20230722_165815_7570'))
    net.to(device)
    net.eval()

    input_image = Image.open(msg).convert('L')
    input_image = Image.fromarray(np.array(input_image))
    transform = transforms.Compose([
            transforms.ToTensor()
    ])

    image = transform(input_image).unsqueeze(0).to(device)
    action_dict = {'left': 0, 'middle': 1, 'right': 2}

    for action in action_dict:
        action_value = torch.tensor([action_dict[action]], dtype=torch.float).to(device)

        with torch.no_grad():
            prediction = net(image, action_value)
        
        #print(f'{action}: {prediction.item()}')
    
    return prediction.item()

def callback(location, orientation, errors, status):
    point_a = {
        # "long" : 59.91,
        # "lat" : 10.72
        "long" : location.vector.x,
        "lat" : location.vector.y
    }

    errors = predict(errors)
    robot_yaw = get_yaw(orientation.quaternion)
    geodata = get_geodata(point_a, point_b)
    azimuth = geodata["azi1"]
    robot_b_heading = azimuth - (robot_yaw)

    print("____________________________________________________________________")
    print(f"azimuth: {azimuth}")
    print(f"robot_yaw: {robot_yaw}")
    print(f"heading: {robot_b_heading}")

    cost_twist(errors, robot_b_heading)


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


def cost_twist(predictions, heading):
    twist.linear.x = 0.3
    left_bias = predictions[0] / predictions[1]
    right_bias = predictions[2] / predictions[1]

    scale = 1
    cost_heading = heading / 100 * scale    # if negative, need to turn left
                                            # if positive, need to turn right
    cost_left = left_bias + cost_heading
    cost_right = right_bias - cost_heading

    #print(f"biases: left - {round(left_bias,3)}, right - {round(right_bias,3)}")
    #print(f"costs: left - {round(cost_left,3)}, right - {round(cost_right,3)}")

    if cost_left > 1.5 and cost_right > 1.5:
        twist.angular.z = 0
        rospy.loginfo("M")

    elif cost_left < cost_right:
        twist.angular.z = -0.3
        rospy.loginfo("L")
    elif cost_right < cost_left:
        twist.angular.z = 0.3
        rospy.loginfo("R")

    pub.publish(twist)


if __name__ == '__main__':
    rospy.init_node('ml_test')
    
    point_b = {
        "long" : 59.9091011614321, 
        "lat" : 10.720408935369424
    }

    gps_sub = message_filters.Subscriber('/filter/positionlla', Vector3Stamped)
    imu_sub = message_filters.Subscriber('/filter/quaternion', QuaternionStamped)
    heightmap_sub = message_filters.Subscriber('/terrain_nav/heightmap', Image)
    status_sub = message_filters.Subscriber('/scout_status', ScoutStatus)

    ts = message_filters.ApproximateTimeSynchronizer([gps_sub, imu_sub, heightmap_sub, status_sub], queue_size=10, slop=0.1, allow_headerless=True)
    ts.registerCallback(callback)

    pub = rospy.Publisher('/cmd_vel', Twist, queue_size=2) 
    rospy.Rate(2) 
    twist = Twist()

    #srv = Server(TutorialsConfig, callback)
    
    rospy.spin()