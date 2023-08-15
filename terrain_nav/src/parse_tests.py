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
from scipy.spatial.transform import Rotation
from pykalman import KalmanFilter


def quaternion_to_euler(x, y, z, w):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        rot = Rotation.from_quat([x, y, z, w])
        return rot.as_euler('xyz', degrees=True)

from scipy.signal import butter, filtfilt

# define a Butterworth low-pass filter
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

# apply the filter
def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

# filter parameters
order = 6
fs = 30.0       # sample rate, Hz
cutoff = 3.667  # desired cutoff frequency of the filter, Hz

###################################################################
home_dir = os.path.expanduser('~')
bag_dir = '/media/user/Plex/ROSBAGS/tests'
results_path = home_dir+'/scout_ws/testData/results_yaw.csv'

# Create folder for results if it doesn't exist
if not os.path.exists(home_dir+'/scout_ws/testData'):
    os.makedirs(home_dir+'/scout_ws/testData')

bag_files = os.listdir(bag_dir)

# sort the list of files based on their creation date
bag_files.sort(key=lambda s: os.path.getmtime(os.path.join(bag_dir, s)))

# iterate through the sorted files
for bag_file in bag_files:
    bag_path = os.path.join(bag_dir, bag_file)

    if os.path.isfile(bag_path) and bag_file.endswith('.bag'):

        # Open rosbag file       
        bag = rosbag.Bag(bag_path)

        current_values, scout_times, avg_driver_voltage, odom_list, imu_times, accel_data, orientation_data = [], [], [], [], [], [], []

        # Iterate over all messages in the rosbag of the topic /scout_status, append the total current of all 4 motors and time measurements
        for topic, msg, t in bag.read_messages(topics='/scout_status'):
            current_values.append(msg.motor_states[0].current+msg.motor_states[1].current+msg.motor_states[2].current+msg.motor_states[3].current)
            scout_times.append(msg.header.stamp.to_sec())
            avg_driver_voltage.append(sum(msg.driver_states[i].driver_voltage for i in range(4)) / 4)  # Gather average driver voltage


        for topic, msg, t in bag.read_messages(topics='/odom'):
            odom_list.append(msg)
        
        for topic, msg, t in bag.read_messages(topics='/imu/data'):
            #imu_list.append(msg)
            imu_times.append(msg.header.stamp.to_sec())
            accel_data.append([msg.linear_acceleration.x, 
                             msg.linear_acceleration.y, 
                             msg.linear_acceleration.z])
            orientation_data.append([msg.orientation.x, 
                                     msg.orientation.y, 
                                     msg.orientation.z, 
                                     msg.orientation.w])

        #accel_data = np.array(accel_data)
        orientation_data = np.array(orientation_data)   

        window_length = min(len(accel_data), 51)
        window_length = window_length if window_length % 2 == 1 else window_length - 1

        #filtered_accel = savgol_filter(accel_data, window_length, 3, axis=0)
        accel_data = np.array(accel_data)
        filtered_accel_data = []

        # assuming cutoff and fs (sampling frequency) are already defined
        for axis in range(accel_data.shape[1]):  # shape[1] gets the number of columns
            filtered_accel = butter_lowpass_filter(accel_data[:, axis], cutoff, fs, order)
            filtered_accel_data.append(filtered_accel)

        filtered_accel_data = np.array(filtered_accel_data).T  # transpose back to original shape
        filtered_orientation = savgol_filter(orientation_data, window_length, 3, axis=0)
        
        # Empty list for charge calculations
        charge_consumed, energy_consumed = [], []

        # For-loop with index to get matching pairs from current_values and scout_times
        for i in range(len(current_values)-1):
            # Calculate charge consumed
            time_interval = scout_times[i+1] - scout_times[i]
            charge = current_values[i] * time_interval
            charge_consumed.append(charge)

            # Calculate energy consumption
            power = current_values[i] * avg_driver_voltage[i]
            energy = power * time_interval
            energy_consumed.append(energy)
        
        # Sum up all the charge consumed in As
        total_charge = sum(charge_consumed)
        total_energy = sum(energy_consumed)
        # Time elapsed
        time_elapsed = scout_times[-1] - scout_times[0]
        # Convert from As to mAh
        total_charge = total_charge / 3600 * 1000
        # Energy / s
        total_power = total_energy / time_elapsed
        

        # Odometry distance
        odom_distance = 0
        for i in range(len(odom_list) - 1):
            x1 = odom_list[i].pose.pose.position.x
            y1 = odom_list[i].pose.pose.position.y
            x2 = odom_list[i+1].pose.pose.position.x
            y2 = odom_list[i+1].pose.pose.position.y
            distance = np.sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1))
            odom_distance += distance


        # Stability (std/var of roll and pitch)
        roll_angles, pitch_angles, yaw_angles = [], [], []
        for quaternion in orientation_data:
            x, y, z, w = quaternion
            roll, pitch, yaw = quaternion_to_euler(x, y, z, w)
            roll_angles.append(roll)
            pitch_angles.append(pitch)
            yaw_angles.append(yaw)

        total_yaw_rotation = 0
        for i in range(1, len(yaw_angles)):
            # Calculate the difference between successive yaw angles
            yaw_difference = abs(yaw_angles[i] - yaw_angles[i-1])
            
            # If using radians, you might want to handle wrap-around if the difference is large
            if yaw_difference > np.pi:
                yaw_difference = 2 * np.pi - yaw_difference

            total_yaw_rotation += yaw_difference
        
        # Std dev
        roll_std_dev = np.std(roll_angles)
        pitch_std_dev = np.std(pitch_angles)
        # Variance
        roll_var = np.var(roll_angles)
        pitch_var = np.var(pitch_angles)
        

        # # Distance based on acceleration
        # velocity = np.zeros_like(accel_data)
        # position = np.zeros_like(accel_data)

        # for i in range(1, len(accel_data)):
        #     imu_time = imu_list[i].header.stamp.to_sec()
        #     dt = imu_time[i] - imu_time[i - 1]
        #     velocity[i] = velocity[i - 1] + accel_data[i] * dt
        #     position[i] = position[i - 1] + velocity[i] * dt

        # accel_distance_2D = np.sqrt(position[-1, 0] ** 2 + position[-1, 1] ** 2)
        # accel_distance_3D = np.sqrt(position[-1, 0] ** 2 + position[-1, 1] ** 2 + position[-1, 2] ** 2)


        # velocity = [0]
        # distance_IMU = [0]

        # # Integrate acceleration to get velocity and distance, get total distance
        # for i in range(1, len(accel_data)):
        #     dt = imu_times[i] - imu_times[i-1]
        #     velocity.append(velocity[i-1] + accel_data[i]*dt)
        #     distance_IMU.append(distance_IMU[i-1] + velocity[i]*dt)

        # total_distance_IMU = distance_IMU[-1]

        # velocity = np.array([0, 0, 0])
        # distance_IMU = np.array([0, 0, 0])
        # total_distance_IMU = 0

        # # assume `accel_data` is your list of 3D linear acceleration data
        # # and `imu_times` is your list of timestamps
        # for i in range(1, len(accel_data)):
        #     dt = imu_times[i] - imu_times[i-1]
        #     velocity = velocity + accel_data[i]*dt
        #     displacement = velocity*dt
        #     distance_IMU = distance_IMU + displacement

        #     # calculate the magnitude of the displacement and add it to the total distance
        #     total_distance_IMU += np.linalg.norm(displacement)
        
        # total_distance_IMU = total_distance_IMU / 1000
        # charge_per_meter = total_charge / total_distance_IMU
        # energy_per_meter = total_energy / total_distance_IMU
        energy_per_odom_meter = total_energy / odom_distance

        # est_wheel_slip = odom_distance - total_distance_IMU
        
        # # Save to a csv file
        # with open(results_path, 'a', newline='') as results_file:
        #     annotations_writer = csv.writer(results_file)
        #     annotations_writer.writerow([bag_file, time_elapsed, total_charge, total_energy, power, 
        #                                  odom_distance, total_distance_IMU, est_wheel_slip, 
        #                                  charge_per_meter, energy_per_meter, energy_per_odom_meter,
        #                                  roll_std_dev, pitch_std_dev, roll_var, pitch_var])

        # Define your headers in a list
        headers = ['bag_file', 'time_elapsed', 'total_charge', 'total_energy', 'total_power', 
                'odom_distance', 'energy_per_odom_meter',
                'roll_std_dev', 'pitch_std_dev', 'roll_var', 'pitch_var', 'total_yaw_rotation']

        # Open your CSV file
        with open(results_path, 'a', newline='') as results_file:
            writer = csv.DictWriter(results_file, fieldnames=headers)
        
            # Write the headers, only if it isn't already there
            if results_file.tell() == 0:
                writer.writeheader()

            writer.writerow({
                'bag_file': bag_file, 
                'time_elapsed': time_elapsed,
                'total_charge': total_charge,
                'total_energy': total_energy,
                'total_power': total_power,
                'odom_distance': odom_distance,
                'energy_per_odom_meter': energy_per_odom_meter,
                'roll_std_dev': roll_std_dev,
                'pitch_std_dev': pitch_std_dev,
                'roll_var': roll_var,
                'pitch_var': pitch_var,
                'total_yaw_rotation': total_yaw_rotation
            })


        # Show progress by print image_num of how many total rosbags there are, as fstring
        print(f'Completed rosbag {bag_file}')
        