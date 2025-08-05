#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import sqlite3
import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import os
import argparse
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_msgs.msg import TFMessage
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import tf2_py as tf2
from builtin_interfaces.msg import Time
import math


class RosbagDatasetROS2(Node):
    """
    ROS2에서 db3 bag 파일을 처리하여 opponent 차량의 데이터를 ego 차량의 LiDAR 좌표계로 변환
    """
    
    def __init__(self, bag_file_path, output_file, start_time=0, end_time=None):
        super().__init__('rosbag_dataset_ros2')
        
        self.bag_file_path = bag_file_path
        self.output_file = output_file
        self.start_time = start_time if start_time is not None else 0
        self.end_time = end_time
        
        # Topics to process
        self.topics = ["/scan", "/car_state/odom", "/tf", "/tf_static", "/opp/car_state/odom"]
        
        # Data storage
        self.dataset = []
        self.initial_time = None
        
        # TF buffer for coordinate transformations
        self.tf_buffer = tf2.BufferCore()
        
        # Previous opponent position for velocity calculation
        self.prev_opp_pos = None
        self.prev_opp_time = None
        self.prev_opp_yaw = None
        
        # 시간 동기화 설정
        self.sync_tolerance = 0.0125  # 12.5ms 허용 오차
        self.message_buffer = {
            "/scan": [],
            "/car_state/odom": [],
            "/opp/car_state/odom": []
        }
        
        # 시간 범위 동기화를 위한 변수
        self.sync_start_time = None
        self.sync_end_time = None
        self.time_sync_calculated = False
        
    def process_bag(self):
        """
        ROS2 bag 파일(db3)을 읽어서 데이터를 처리
        """
        storage_options = StorageOptions(uri=self.bag_file_path, storage_id='sqlite3')
        converter_options = ConverterOptions('', '')
        
        reader = SequentialReader()
        
        try:
            reader.open(storage_options, converter_options)
            
            topic_types = reader.get_all_topics_and_types()
            
            # 메시지 타입 매핑
            type_map = {}
            for topic_metadata in topic_types:
                type_map[topic_metadata.name] = topic_metadata.type
            
            self.get_logger().info(f'bag 파일 처리 시작: {self.bag_file_path}')
            
            message_count = 0
            while reader.has_next():
                (topic, data, timestamp) = reader.read_next()
                
                # 타임스탬프를 초 단위로 변환
                timestamp_sec = timestamp / 1e9
                
                # 초기 시간 설정
                if self.initial_time is None:
                    self.initial_time = timestamp_sec
                
                # 상대 시간 계산
                relative_time = timestamp_sec - self.initial_time
                
                # 시작 시간 이전이면 건너뛰기
                if relative_time < self.start_time:
                    continue
                
                # 종료 시간 이후면 종료
                if self.end_time and relative_time > self.end_time:
                    break
                
                # 메시지 타입 가져오기
                if topic not in type_map:
                    continue
                
                msg_type = get_message(type_map[topic])
                msg = deserialize_message(data, msg_type)
                
                # TF 메시지 처리
                if topic in ["/tf", "/tf_static"]:
                    self.process_tf_message(msg, topic, timestamp_sec)
                    continue
                
                # 동기화 대상 토픽만 버퍼에 저장
                if topic in self.message_buffer:
                    self.message_buffer[topic].append((msg, timestamp_sec))
                    
                    # 시간 동기화 구간 계산 (처음 한 번만)
                    if not self.time_sync_calculated and len(self.message_buffer["/car_state/odom"]) > 10 and len(self.message_buffer["/opp/car_state/odom"]) > 10:
                        self.calculate_sync_time_range()
                    
                    # 동기화 구간이 계산된 후에만 처리
                    if self.time_sync_calculated:
                        # 오래된 메시지 제거 (1초 이상 된 메시지)
                        current_time = timestamp_sec
                        for topic_name in self.message_buffer:
                            self.message_buffer[topic_name] = [
                                (m, t) for m, t in self.message_buffer[topic_name]
                                if current_time - t < 1.0
                            ]
                        
                        # 동기화된 메시지 그룹 찾기 및 처리
                        synced_count = self.process_synchronized_messages()
                        message_count += synced_count
                        
                        if message_count > 0 and message_count % 100 == 0:
                            self.get_logger().info(f'처리된 메시지: {message_count}개')
            
            self.get_logger().info(f'총 {message_count}개의 데이터 포인트를 처리했습니다.')
            return True
            
        except Exception as e:
            self.get_logger().error(f'Bag 파일 읽기 중 오류 발생: {e}')
            return False
        finally:
            del reader
    
    def process_tf_message(self, msg, topic, timestamp_sec):
        """
        TF 메시지를 처리하여 tf_buffer에 저장
        """
        try:
            if topic == "/tf_static":
                for transform in msg.transforms:
                    self.tf_buffer.set_transform_static(transform, 'default_authority')
            else:  # "/tf"
                for transform in msg.transforms:
                    self.tf_buffer.set_transform(transform, 'default_authority')
        except Exception as e:
            self.get_logger().warning(f'TF 메시지 처리 중 오류: {e}')
    
    def calculate_sync_time_range(self):
        """
        두 bag 파일의 겹치는 시간 구간을 계산
        """
        try:
            if not self.message_buffer["/car_state/odom"] or not self.message_buffer["/opp/car_state/odom"]:
                return
            
            # ego와 opp의 시간 범위 계산
            ego_times = [t for _, t in self.message_buffer["/car_state/odom"]]
            opp_times = [t for _, t in self.message_buffer["/opp/car_state/odom"]]
            
            ego_start = min(ego_times)
            ego_end = max(ego_times)
            opp_start = min(opp_times)
            opp_end = max(opp_times)
            
            # 겹치는 시간 구간 계산: 더 늦게 시작하고 더 빨리 끝나는 시간
            self.sync_start_time = max(ego_start, opp_start)
            self.sync_end_time = min(ego_end, opp_end)
            
            self.time_sync_calculated = True
            
            self.get_logger().info(f'시간 동기화 구간 계산 완료:')
            self.get_logger().info(f'  Ego 시간 범위: {ego_start:.3f}s ~ {ego_end:.3f}s')
            self.get_logger().info(f'  Opp 시간 범위: {opp_start:.3f}s ~ {opp_end:.3f}s')
            self.get_logger().info(f'  동기화 구간: {self.sync_start_time:.3f}s ~ {self.sync_end_time:.3f}s')
            
            # 겹치는 시간이 있는지 확인
            if self.sync_start_time >= self.sync_end_time:
                self.get_logger().error('두 파일의 시간이 겹치지 않습니다!')
                self.time_sync_calculated = False
                
        except Exception as e:
            self.get_logger().warning(f'시간 동기화 구간 계산 중 오류: {e}')
            self.time_sync_calculated = False
    
    def process_synchronized_messages(self):
        """
        시간 동기화된 메시지 그룹을 찾아 처리
        """
        processed_count = 0
        
        # ego odom 메시지를 기준으로 동기화
        if not self.message_buffer["/car_state/odom"]:
            return 0
        
        # 처리되지 않은 ego odom 메시지들을 순서대로 처리
        ego_messages = self.message_buffer["/car_state/odom"].copy()
        
        for ego_msg, ego_time in ego_messages:
            # 동기화 구간 내의 메시지만 처리
            if ego_time < self.sync_start_time or ego_time > self.sync_end_time:
                continue
                
            # ego 시간 기준으로 동기화된 메시지 찾기
            synced_msgs = self.find_synchronized_messages(ego_time)
            
            if synced_msgs:
                scan_msg, scan_time = synced_msgs["/scan"]
                opp_msg, opp_time = synced_msgs["/opp/car_state/odom"]
                
                # 상대 시간 계산
                relative_time = ego_time - self.initial_time if self.initial_time else 0
                
                if self.process_synchronized_data_new(scan_msg, ego_msg, opp_msg, relative_time):
                    processed_count += 1
                
                # 처리된 메시지들을 버퍼에서 제거
                self.remove_processed_messages(ego_time)
        
        return processed_count
    
    def find_synchronized_messages(self, reference_time):
        """
        기준 시간에서 동기화 허용 오차 내의 메시지들을 찾기
        """
        synced_messages = {}
        
        # scan 메시지 찾기 (동기화 구간 내에서)
        scan_msg = self.find_closest_message("/scan", reference_time)
        if not scan_msg:
            return None
        synced_messages["/scan"] = scan_msg
        
        # opp odom 메시지 찾기 (동기화 구간 내에서)
        opp_msg = self.find_closest_message("/opp/car_state/odom", reference_time)
        if not opp_msg:
            return None
        synced_messages["/opp/car_state/odom"] = opp_msg
        
        return synced_messages
    
    def find_closest_message(self, topic, reference_time):
        """
        기준 시간에 가장 가까운 메시지 찾기 (동기화 허용 오차 및 동기화 구간 내에서)
        """
        if not self.message_buffer[topic]:
            return None
        
        closest_msg = None
        min_time_diff = float('inf')
        
        for msg, timestamp in self.message_buffer[topic]:
            # 동기화 구간 내의 메시지만 고려
            if timestamp < self.sync_start_time or timestamp > self.sync_end_time:
                continue
                
            time_diff = abs(timestamp - reference_time)
            
            if time_diff <= self.sync_tolerance and time_diff < min_time_diff:
                min_time_diff = time_diff
                closest_msg = (msg, timestamp)
        
        return closest_msg
    
    def remove_processed_messages(self, processed_time):
        """
        처리된 시간 이전의 메시지들을 버퍼에서 제거
        """
        for topic in self.message_buffer:
            self.message_buffer[topic] = [
                (msg, timestamp) for msg, timestamp in self.message_buffer[topic]
                if timestamp > processed_time
            ]
    
    def process_synchronized_data_new(self, scan_msg, ego_odom_msg, opp_odom_msg, current_time):
        """
        동기화된 데이터를 처리하여 데이터셋 생성 (새로운 버전)
        """
        try:
            # LiDAR 데이터 처리
            ranges = np.array(scan_msg.ranges, dtype=np.float32)
            intensities = np.array(scan_msg.intensities, dtype=np.float32) if scan_msg.intensities else np.full_like(ranges, 0.5)
            
            # 유효한 범위 필터링
            valid_mask = np.isfinite(ranges) & (ranges >= scan_msg.range_min) & (ranges <= scan_msg.range_max)
            valid_ranges = ranges[valid_mask]
            valid_intensities = intensities[valid_mask]
            
            # Opponent 위치를 LiDAR 좌표계로 변환
            opp_pos_lidar, opp_vel_lidar, opp_yaw_lidar = self.transform_opponent_to_lidar(
                opp_odom_msg, ego_odom_msg, current_time
            )
            
            if opp_pos_lidar is not None:
                # 데이터셋 엔트리 생성
                dataset_entry = {
                    'time': current_time,
                    'lidar': valid_ranges.tolist(),
                    'intensities': valid_intensities.tolist(),
                    'x': float(opp_pos_lidar[0]),
                    'y': float(opp_pos_lidar[1]),
                    'vx': float(opp_vel_lidar[0]),
                    'vy': float(opp_vel_lidar[1]),
                    'yaw': float(opp_yaw_lidar)
                }
                
                self.dataset.append(dataset_entry)
                return True
        
        except Exception as e:
            self.get_logger().warning(f'데이터 처리 중 오류: {e}')
        
        return False
    
    def transform_opponent_to_lidar(self, opp_odom_msg, ego_odom_msg, current_time):
        """
        Opponent의 global 좌표를 ego 차량의 LiDAR 좌표계로 변환
        """
        try:
            # Opponent global 위치 및 방향
            opp_pos_global = np.array([
                opp_odom_msg.pose.pose.position.x,
                opp_odom_msg.pose.pose.position.y
            ])
            
            opp_quat = np.array([
                opp_odom_msg.pose.pose.orientation.x,
                opp_odom_msg.pose.pose.orientation.y,
                opp_odom_msg.pose.pose.orientation.z,
                opp_odom_msg.pose.pose.orientation.w
            ])
            opp_rot_global = Rotation.from_quat(opp_quat)
            opp_yaw_global = opp_rot_global.as_euler('xyz')[2]
            
            # Ego global 위치 및 방향
            ego_pos_global = np.array([
                ego_odom_msg.pose.pose.position.x,
                ego_odom_msg.pose.pose.position.y
            ])
            
            ego_quat = np.array([
                ego_odom_msg.pose.pose.orientation.x,
                ego_odom_msg.pose.pose.orientation.y,
                ego_odom_msg.pose.pose.orientation.z,
                ego_odom_msg.pose.pose.orientation.w
            ])
            ego_rot_global = Rotation.from_quat(ego_quat)
            ego_yaw_global = ego_rot_global.as_euler('xyz')[2]
            
            # TF를 통해 map -> laser 변환 시도
            try:
                # 현재 시간으로 TF 조회 (Time(0)은 가장 최근 변환 사용)
                transform = self.tf_buffer.lookup_transform_core("laser", "map", Time())
                
                # map 좌표의 opponent 위치를 laser 좌표계로 변환
                T_trans = np.array([
                    transform.transform.translation.x,
                    transform.transform.translation.y,
                    transform.transform.translation.z
                ])
                
                T_quat = np.array([
                    transform.transform.rotation.x,
                    transform.transform.rotation.y,
                    transform.transform.rotation.z,
                    transform.transform.rotation.w
                ])
                T_rot = Rotation.from_quat(T_quat)
                
                # Opponent 위치를 LiDAR 좌표계로 변환
                opp_pos_3d = np.array([opp_pos_global[0], opp_pos_global[1], 0.0])
                opp_pos_lidar_3d = T_rot.apply(opp_pos_3d) + T_trans
                opp_pos_lidar = opp_pos_lidar_3d[:2]  # x, y만 사용
                
                # Yaw를 LiDAR 좌표계로 변환
                opp_yaw_lidar = opp_yaw_global - ego_yaw_global
                # Yaw를 [-π, π] 범위로 정규화
                opp_yaw_lidar = ((opp_yaw_lidar + math.pi) % (2 * math.pi)) - math.pi
                
            except Exception as tf_error:
                # TF 조회 실패 시 수동 계산
                self.get_logger().warning(f'TF 조회 실패, 수동 계산 사용: {tf_error}')
                
                # Ego 좌표계로 변환 (간단한 2D 변환)
                relative_pos = opp_pos_global - ego_pos_global
                
                # Ego의 yaw만큼 회전하여 LiDAR 좌표계로 변환
                cos_yaw = math.cos(-ego_yaw_global)
                sin_yaw = math.sin(-ego_yaw_global)
                
                opp_pos_lidar = np.array([
                    relative_pos[0] * cos_yaw - relative_pos[1] * sin_yaw,
                    relative_pos[0] * sin_yaw + relative_pos[1] * cos_yaw
                ])
                
                # Yaw를 LiDAR 좌표계로 변환
                opp_yaw_lidar = opp_yaw_global - ego_yaw_global
                opp_yaw_lidar = ((opp_yaw_lidar + math.pi) % (2 * math.pi)) - math.pi
            
            # 속도 계산 (시간 차분 기반)
            opp_vel_lidar = self.calculate_opponent_velocity(
                opp_pos_lidar, opp_yaw_lidar, current_time
            )
            
            return opp_pos_lidar, opp_vel_lidar, opp_yaw_lidar
            
        except Exception as e:
            self.get_logger().error(f'좌표 변환 중 오류: {e}')
            return None, None, None
    
    def calculate_opponent_velocity(self, opp_pos_lidar, opp_yaw_lidar, current_time):
        """
        시간 차분을 이용한 opponent 속도 계산
        """
        try:
            if self.prev_opp_pos is not None and self.prev_opp_time is not None:
                dt = current_time - self.prev_opp_time
                
                if 0.01 < dt < 0.1:  # 합리적인 시간 간격
                    # 위치 변화량 계산
                    dx = opp_pos_lidar[0] - self.prev_opp_pos[0]
                    dy = opp_pos_lidar[1] - self.prev_opp_pos[1]
                    
                    # 속도 계산
                    vx = dx / dt
                    vy = dy / dt
                    
                    # 속도 크기 제한 (20 m/s 이상은 비현실적)
                    velocity_magnitude = math.sqrt(vx**2 + vy**2)
                    if velocity_magnitude > 20.0:
                        vx, vy = 0.0, 0.0
                    
                    # 이전 값 업데이트
                    self.prev_opp_pos = opp_pos_lidar.copy()
                    self.prev_opp_time = current_time
                    self.prev_opp_yaw = opp_yaw_lidar
                    
                    return np.array([vx, vy])
            
            # 첫 번째 계산이거나 시간 간격이 부적절한 경우
            self.prev_opp_pos = opp_pos_lidar.copy()
            self.prev_opp_time = current_time
            self.prev_opp_yaw = opp_yaw_lidar
            
            return np.array([0.0, 0.0])
            
        except Exception as e:
            self.get_logger().warning(f'속도 계산 중 오류: {e}')
            return np.array([0.0, 0.0])
    
    def save_to_csv(self):
        """
        데이터셋을 CSV 파일로 저장
        """
        if not self.dataset:
            self.get_logger().error('저장할 데이터가 없습니다!')
            return False
        
        # 출력 디렉토리 생성
        output_dir = os.path.dirname(self.output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        headers = ['time', 'lidar', 'intensities', 'x', 'y', 'vx', 'vy', 'yaw']
        
        try:
            with open(self.output_file, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=headers)
                writer.writeheader()
                
                for entry in self.dataset:
                    writer.writerow(entry)
            
            self.get_logger().info(f'CSV 파일 저장 완료: {self.output_file}')
            self.get_logger().info(f'총 {len(self.dataset)}개의 엔트리가 저장되었습니다.')
            return True
            
        except Exception as e:
            self.get_logger().error(f'CSV 저장 중 오류: {e}')
            return False
    
    def visualize_data(self, show_count=10):
        """
        데이터 시각화 (처음 몇 개의 데이터만)
        """
        if not self.dataset:
            self.get_logger().warning('시각화할 데이터가 없습니다!')
            return
        
        plt.figure(figsize=(12, 8))
        
        for i in range(min(show_count, len(self.dataset))):
            entry = self.dataset[i]
            
            # LiDAR 데이터 시각화
            ranges = np.array(entry['lidar'])
            if len(ranges) > 0:
                # 각도 계산 (일반적인 LiDAR 파라미터 사용)
                angles = np.linspace(-2.356194496154785, 2.356194496154785, len(ranges))
                
                # 극좌표를 직교좌표로 변환
                x_points = ranges * np.cos(angles)
                y_points = ranges * np.sin(angles)
                
                plt.subplot(2, 5, i + 1)
                plt.scatter(x_points, y_points, c='lightblue', s=1, alpha=0.6)
                
                # Opponent 위치 표시
                if entry['x'] != 0.0 or entry['y'] != 0.0:
                    plt.scatter(entry['x'], entry['y'], c='red', s=50, marker='o')
                    
                    # 속도 벡터 표시
                    if entry['vx'] != 0.0 or entry['vy'] != 0.0:
                        plt.arrow(entry['x'], entry['y'], 
                                entry['vx'], entry['vy'],
                                head_width=0.1, head_length=0.1, 
                                fc='blue', ec='blue')
                
                # LiDAR 위치 (원점)
                plt.scatter(0, 0, c='black', s=30, marker='s')
                
                plt.axis('equal')
                plt.title(f'Data Point {i+1}')
                plt.xlabel('X (m)')
                plt.ylabel('Y (m)')
                plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='ROS2 bag 파일에서 opponent 차량 데이터를 추출하여 CSV로 저장')
    parser.add_argument('--bag', type=str, required=True, help='ROS2 bag 파일 경로 (db3)')
    parser.add_argument('--output', type=str, help='출력 CSV 파일 경로')
    parser.add_argument('--start', type=float, default=0, help='시작 시간 (초)')
    parser.add_argument('--end', type=float, help='종료 시간 (초)')
    parser.add_argument('--visualize', action='store_true', help='데이터 시각화')
    
    args = parser.parse_args()
    
    # 출력 경로 설정
    if args.output is None:
        bag_name = os.path.basename(args.bag).split('.')[0]
        args.output = f'/home/harry/ros2_ws/src/redbull/{bag_name}_opponent_dataset.csv'
    
    # bag 파일 존재 확인
    if not os.path.exists(args.bag):
        print(f'Error: bag 파일을 찾을 수 없습니다: {args.bag}')
        return
    
    # ROS2 초기화
    rclpy.init()
    
    try:
        # 데이터셋 생성기 생성 및 실행
        dataset_generator = RosbagDatasetROS2(args.bag, args.output, args.start, args.end)
        
        print(f'ROS2 bag 파일 처리 시작: {args.bag}')
        if dataset_generator.process_bag():
            print('데이터 처리 완료')
            
            print('CSV 파일 저장 중...')
            if dataset_generator.save_to_csv():
                print(f'성공적으로 완료되었습니다: {args.output}')
                
                # 시각화 옵션
                if args.visualize:
                    print('데이터 시각화 중...')
                    dataset_generator.visualize_data()
            else:
                print('CSV 파일 저장에 실패했습니다.')
        else:
            print('bag 파일 처리에 실패했습니다.')
    
    except Exception as e:
        print(f'오류 발생: {e}')
    
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
