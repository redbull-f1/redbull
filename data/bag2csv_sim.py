#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import csv
import os
from scipy.spatial.transform import Rotation
from nav_msgs.msg import Odometry
import time
'''
python3 bag2csv_sim.py --bag /home/harry/Downloads/levine_obs.db3

정적 장애물 global 좌표를 LiDAR 좌표로 변환하여 CSV로 저장합니다.

'''



# 장애물의 전역 좌표 (예시)
OBSTACLE_GLOBAL_X = 5.2
OBSTACLE_GLOBAL_Y = 0.84

# LiDAR 오프셋 (base_link → laser, x: 전방, y: 좌우, z: 높이)
LIDAR_OFFSET_X = 0.275  # xacro에서 0.275m
LIDAR_OFFSET_Y = 0.0

timestamp_str = time.strftime("%Y%m%d_%H%M%S")
OUTPUT_CSV = f'/home/harry/ros2_ws/src/redbull/data/obstacle_lidar_{timestamp_str}.csv'

class Bag2CSV(Node):
    def __init__(self):
        super().__init__('bag2csv')
        self.csv_rows = []

    def odom_callback(self, msg):
        # Ego 차량의 전역 위치 및 자세
        ego_x = msg.pose.pose.position.x
        ego_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        ego_quat = [q.x, q.y, q.z, q.w]
        ego_yaw = Rotation.from_quat(ego_quat).as_euler('xyz')[2]

        # 장애물의 전역 좌표
        obs_global = np.array([OBSTACLE_GLOBAL_X, OBSTACLE_GLOBAL_Y])
        ego_global = np.array([ego_x, ego_y])

        # 1. 전역 → base_link 변환
        rel = obs_global - ego_global
        cos_yaw = np.cos(-ego_yaw)
        sin_yaw = np.sin(-ego_yaw)
        obs_base = np.array([
            rel[0] * cos_yaw - rel[1] * sin_yaw,
            rel[0] * sin_yaw + rel[1] * cos_yaw
        ])

        # 2. base_link → lidar 변환 (오프셋 적용)
        obs_lidar = np.array([
            obs_base[0] - LIDAR_OFFSET_X,
            obs_base[1] - LIDAR_OFFSET_Y
        ])

        # timestamp (초 단위 float)
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        # 저장
        self.csv_rows.append([f"{t:.6f}", f"{obs_lidar[0]:.6f}", f"{obs_lidar[1]:.6f}"])

    def save_csv(self):
        os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
        with open(OUTPUT_CSV, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'x_lidar', 'y_lidar'])
            writer.writerows(self.csv_rows)
        print(f"Saved: {OUTPUT_CSV}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--bag', type=str, required=True, help='Input ROS2 bag file (db3)')
    args = parser.parse_args()

    import rosbag2_py
    from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    rclpy.init()
    node = Bag2CSV()

    # bag 읽기
    storage_options = StorageOptions(uri=args.bag, storage_id='sqlite3')
    converter_options = ConverterOptions('', '')
    reader = SequentialReader()
    reader.open(storage_options, converter_options)
    topic_types = {t.name: t.type for t in reader.get_all_topics_and_types()}

    while reader.has_next():
        topic, data, timestamp = reader.read_next()
        if topic == '/ego_racecar/odom':
            msg_type = get_message(topic_types[topic])
            msg = deserialize_message(data, msg_type)
            node.odom_callback(msg)

    node.save_csv()
    rclpy.shutdown()

if __name__ == '__main__':
    main()