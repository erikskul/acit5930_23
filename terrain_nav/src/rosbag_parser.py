#!/usr/bin/env python

import os
import rospy
import rosbag
import subprocess
import csv
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.signal import butter, filtfilt, savgol_filter
from pykalman import KalmanFilter

home_dir = os.path.expanduser('~')
bag_dir = '/media/user/Plex/ROSBAGS/day1_1'
annotation_path = home_dir+'/scout_ws/datasetAll/annotations.csv'
bagnames_path = home_dir+'/scout_ws/datasetAll/bag_to_image.csv'
image_dir = home_dir+'/scout_ws/datasetAll/images'
image_num = 1
mirroring = True


def kalman_filter(imu_data):
    # Set initial state and initial covariance
    initial_state = imu_data[0]
    initial_covariance = np.eye(len(initial_state)) * 1e-6

    # Set observation covariance
    observation_covariance = np.eye(len(initial_state)) * 1e-6

    # Set transition covariance
    transition_covariance = np.eye(len(initial_state)) * 1e-6

    # Create the Kalman filter
    kf = KalmanFilter(initial_state_mean=initial_state,
                      initial_state_covariance=initial_covariance,
                      observation_covariance=observation_covariance,
                      transition_covariance=transition_covariance)

    # Apply the Kalman filter on the data
    filtered_state_means, _ = kf.filter(imu_data)
    return filtered_state_means

# Create directories if they don't exist
if not os.path.exists(image_dir):
    os.makedirs(image_dir)

bag_files = os.listdir(bag_dir)

# sort the list of files based on their creation date
bag_files.sort(key=lambda s: os.path.getmtime(os.path.join(bag_dir, s)))

# iterate through the sorted files
for bag_file in bag_files:
    bag_path = os.path.join(bag_dir, bag_file)

    if os.path.isfile(bag_path) and bag_file.endswith('.bag'):

        # Find which action the current file is
        action = bag_file.split('_')[0]

        # Image name
        image_name = str(image_num) + '.png'

        # Open rosbag file       
        bag = rosbag.Bag(bag_path)

        # Empty lists for extracting current and time measurements
        current_values = []
        time_stamps = []
        odom_list = []
        odom_filt_list = []
        imu_list = []
        imu_data = []

        # Iterate over all messages in the rosbag of the topic /scout_status, append the total current of all 4 motors and time measurements
        for topic, msg, t in bag.read_messages(topics='/scout_status'):
            current_values.append(msg.motor_states[0].current+msg.motor_states[1].current+msg.motor_states[2].current+msg.motor_states[3].current)
            time_stamps.append(msg.header.stamp.to_sec())

        for topic, msg, t in bag.read_messages(topics='/odom'):
            odom_list.append(msg)

        # for topic, msg, t in bag.read_messages(topics='/odometry/filtered'):
        #     odom_filt_list.append(msg)
        
        for topic, msg, t in bag.read_messages(topics='/imu/data'):
            imu_list.append(msg)
            imu_data.append([msg.linear_acceleration.x, 
                             msg.linear_acceleration.y, 
                             msg.linear_acceleration.z])

        imu_data = np.array(imu_data)   

        window_length = min(len(imu_data), 51)
        window_length = window_length if window_length % 2 == 1 else window_length - 1

        filtered_data = savgol_filter(imu_data, window_length, 3, axis=0)
        
        # Empty list for charge calculations
        charge_consumed = []

        # For-loop with index to get matching pairs from current_values and time_stamps
        for i in range(len(current_values)-1):
            time_interval = time_stamps[i+1] - time_stamps[i]
            charge = current_values[i] * time_interval
            charge_consumed.append(charge)

        # Sum up all the charge consumed in As
        total_charge = sum(charge_consumed)

        # Convert from As to mAh
        total_charge = total_charge / 3600 * 1000

        # ODOM ROBOT
        total_distance = 0
        for i in range(len(odom_list) - 1):
            x1 = odom_list[i].pose.pose.position.x
            y1 = odom_list[i].pose.pose.position.y
            x2 = odom_list[i+1].pose.pose.position.x
            y2 = odom_list[i+1].pose.pose.position.y
            distance = np.sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1))
            total_distance += distance

        # # ACCEL ONLY
        # # initialize position to (0, 0)
        # x_3 = 0
        # y_3 = 0

        # # initialize velocity to (0, 0)
        # vx_3 = 0
        # vy_3 = 0

        # # loop over all odometry messages
        # for i in range(len(imu_list) - 1):
        #     # get the timestamp of the current odometry message
        #     imu_time = imu_list[i].header.stamp.to_nsec()

        #     # get the linear acceleration values from the imu message
        #     ax_3 = imu_list[i].linear_acceleration.x
        #     ay_3 = imu_list[i].linear_acceleration.y

        #     # get the time step
        #     dt_3 = (imu_list[i].header.stamp.to_nsec() - imu_list[i-1].header.stamp.to_nsec()) * 1e-10

        #     # update the velocity and position, using equations of motion for velocity and distance
        #     vx_3 += ax_3 * dt_3
        #     vy_3 += ay_3 * dt_3
        #     x_3 += vx_3 * dt_3 + 0.5 * ax_3 * dt_3 * dt_3
        #     y_3 += vy_3 * dt_3 + 0.5 * ay_3 * dt_3 * dt_3

        # imu_distance3 = np.sqrt(x_3*x_3 + y_3*y_3) * 10

        # # Accel2
        # velocity = np.zeros_like(filtered_data)
        # position = np.zeros_like(filtered_data)

        # for i in range(1, len(filtered_data)):
        #     imu_time = imu_list[i].header.stamp.to_sec()

        #     velocity[i] = velocity[i - 1] + filtered_data[i] * imu_time
        #     position[i] = position[i - 1] + velocity[i] * imu_time

        # distance_2D = np.sqrt(position[-1, 0] ** 2 + position[-1, 1] ** 2) / 1e23
        # distance_3D = np.sqrt(position[-1, 0] ** 2 + position[-1, 1] ** 2 + position[-1, 2] ** 2) / 1e24

        # Accel3
        velocity2 = np.zeros_like(imu_data)
        position2 = np.zeros_like(imu_data)

        for i in range(1, len(imu_data)):
            imu_time2 = imu_list[i].header.stamp.to_sec()

            velocity2[i] = velocity2[i - 1] + imu_data[i] * imu_time2
            position2[i] = position2[i - 1] + velocity2[i] * imu_time2

        distance_2D_2 = np.sqrt(position2[-1, 0] ** 2 + position2[-1, 1] ** 2) / 1e23
        distance_3D_2 = np.sqrt(position2[-1, 0] ** 2 + position2[-1, 1] ** 2 + position2[-1, 2] ** 2) / 1e24
        energy_per_meter = total_charge / distance_2D_2
        # Save action and total_charge to a csv file
        with open(annotation_path, 'a', newline='') as annotations_file:
            annotations_writer = csv.writer(annotations_file)
            annotations_writer.writerow([image_name, action, total_charge, distance_2D_2, energy_per_meter])

                # # Just in case, save original bag name with its corresponding image name into a csv file
        with open(bagnames_path, 'a', newline='') as bag_to_image_file:
            bag_to_image_writer = csv.writer(bag_to_image_file)
            bag_to_image_writer.writerow([bag_file, image_name])

        image_num += 1

        if mirroring:
            image_name = str(image_num) + '.png'
            

            if action == 'left':
                with open(annotation_path, 'a', newline='') as annotations_file:
                    annotations_writer = csv.writer(annotations_file)
                    annotations_writer.writerow([image_name, 'right', total_charge, distance_2D_2, energy_per_meter])
                with open(bagnames_path, 'a', newline='') as bag_to_image_file:
                    bag_to_image_writer = csv.writer(bag_to_image_file)
                    bag_to_image_writer.writerow([bag_file, image_name, 'mirrored'])
                
                image_num += 1

            elif action == 'right':
                with open(annotation_path, 'a', newline='') as annotations_file:
                    annotations_writer = csv.writer(annotations_file)
                    annotations_writer.writerow([image_name, 'left', total_charge, distance_2D_2, energy_per_meter])
                with open(bagnames_path, 'a', newline='') as bag_to_image_file:
                    bag_to_image_writer = csv.writer(bag_to_image_file)
                    bag_to_image_writer.writerow([bag_file, image_name, 'mirrored'])

                image_num += 1

        # Show progress by print image_num of how many total rosbags there are, as fstring
        print(f'Completed rosbag {bag_file}, {image_num} of {len(os.listdir(bag_dir))} done')

        # Increment image_num
        