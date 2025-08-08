#!/usr/bin/env python3
"""
ROS2 bag 파일에서 opponent 차량 데이터를 추출하여 
ego 차량의 LiDAR 좌표계로 변환 후 CSV 파일로 저장

    csv 파일 형식: (x,y,vx,vy,yaw)
    - LiDAR 데이터: /scan
    - Ego 차량 상태: /car_state/odom
    - Opponent 차량 상태: /opp/car_state/odom
    - TF 메시지: /tf, /tf_static
    - 시간 동기화: 두 bag 파일의 겹치는 시간 구간만 병합
    - 데이터셋 형식: CSV 파일로 저장

Usage:
    # 기본 사용
    python3 parse_bag_csv.py input.db3

    # 옵션 사용
    python3 parse_bag_csv.py input.db3 --output result.csv --start 10.0 --end 60.0

    # LiDAR 오프셋 조정 (기본값: x=0.28, y=0.0, z=0.11)
    python3 parse_bag_csv.py input.db3 --lidar-x 0.28 --lidar-y 0.0 --lidar-z 0.11
"""

import rclpy
from rclpy.node import Node
import numpy as np
import csv
import os
import argparse
import math
from scipy.spatial.transform import Rotation

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from tf2_msgs.msg import TFMessage
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import tf2_py as tf2
from builtin_interfaces.msg import Time


class BagToCSV(Node):
    """ROS2 bag 파일을 읽어서 opponent 차량 데이터를 CSV로 변환"""
    
    def __init__(self, bag_path, output_csv, start_time=0, end_time=None, 
                 lidar_offset_x=0.287, lidar_offset_y=0.0, lidar_offset_z=0.115):
        super().__init__('bag_to_csv')
        
        self.bag_path = bag_path
        self.output_csv = output_csv
        self.start_time = start_time
        self.end_time = end_time
        
        # LiDAR 센서 오프셋 (base_link -> laser 프레임)
        self.lidar_offset_x = lidar_offset_x
        self.lidar_offset_y = lidar_offset_y
        self.lidar_offset_z = lidar_offset_z
        
        # 처리할 토픽들
        self.topics = ["/scan", "/car_state/odom", "/opp/car_state/odom", "/tf"]
        
        # 데이터 저장
        self.dataset = []
        self.initial_time = None
        
        # TF 버퍼
        self.tf_buffer = tf2.BufferCore()
        
        # 속도 계산용 이전 값들
        self.prev_opp_pos = None
        self.prev_opp_time = None
        
        # 메시지 버퍼 (시간 동기화용)
        self.buffers = {
            "/scan": [],
            "/car_state/odom": [],
            "/opp/car_state/odom": []
        }
        
        # 동기화 설정
        self.sync_tolerance = 0.02  # 20ms 허용오차
        self.min_interval = 1.0 / 70.0  # 70Hz 기준
        self.max_buffer_size = 50
        
        # 동기화 시간 범위
        self.sync_start = None
        self.sync_end = None
        self.sync_ready = False
    
    def process_bag(self):
        """bag 파일 처리 메인 함수"""
        self.get_logger().info(f'Processing bag: {self.bag_path}')
        
        storage_options = StorageOptions(uri=self.bag_path, storage_id='sqlite3')
        converter_options = ConverterOptions('', '')
        reader = SequentialReader()
        
        try:
            reader.open(storage_options, converter_options)
            topic_types = reader.get_all_topics_and_types()
            
            # 토픽 타입 매핑
            type_map = {topic.name: topic.type for topic in topic_types}
            
            message_count = 0
            processed_count = 0
            
            while reader.has_next():
                topic, data, timestamp = reader.read_next()
                timestamp_sec = timestamp / 1e9
                
                # 초기 시간 설정
                if self.initial_time is None:
                    self.initial_time = timestamp_sec
                
                relative_time = timestamp_sec - self.initial_time
                
                # 시간 필터링
                if relative_time < self.start_time:
                    continue
                if self.end_time and relative_time > self.end_time:
                    break
                
                # 메시지 처리
                if topic in type_map and topic in self.topics:
                    msg_type = get_message(type_map[topic])
                    msg = deserialize_message(data, msg_type)
                    
                    if topic == "/tf":
                        self._process_tf(msg)
                        continue
                    
                    if topic in self.buffers:
                        self.buffers[topic].append((msg, timestamp_sec))
                        message_count += 1
                        
                        # 버퍼 정리
                        self._cleanup_buffers(timestamp_sec)
                        
                        # 동기화 범위 계산
                        if not self.sync_ready and self._has_enough_data():
                            self._calculate_sync_range()
                        
                        # 데이터 처리
                        if self.sync_ready:
                            batch_count = self._process_batch()
                            processed_count += batch_count
                            
                            if processed_count % 100 == 0 and processed_count > 0:
                                self.get_logger().info(f'Processed: {processed_count} points')
            
            # 마지막 배치 처리
            if self.sync_ready:
                processed_count += self._process_batch()
            
            self.get_logger().info(f'Total processed: {processed_count} data points')
            return processed_count > 0
            
        except Exception as e:
            self.get_logger().error(f'Error processing bag: {e}')
            return False
        finally:
            del reader
    
    def _process_tf(self, tf_msg):
        """TF 메시지 처리"""
        try:
            for transform in tf_msg.transforms:
                self.tf_buffer.set_transform(transform, 'default_authority')
        except Exception:
            pass
    
    def _has_enough_data(self):
        """동기화를 위한 충분한 데이터 확인"""
        return (len(self.buffers["/car_state/odom"]) > 5 and 
                len(self.buffers["/opp/car_state/odom"]) > 5)
    
    def _calculate_sync_range(self):
        """ego와 opponent 데이터의 겹치는 시간 범위 계산"""
        ego_times = [t for _, t in self.buffers["/car_state/odom"]]
        opp_times = [t for _, t in self.buffers["/opp/car_state/odom"]]
        
        if ego_times and opp_times:
            self.sync_start = max(min(ego_times), min(opp_times))
            self.sync_end = min(max(ego_times), max(opp_times))
            
            if self.sync_start < self.sync_end:
                self.sync_ready = True
                duration = self.sync_end - self.sync_start
                self.get_logger().info(f'Sync range: {duration:.2f} seconds')
    
    def _cleanup_buffers(self, current_time):
        """오래된 메시지 정리"""
        cutoff = current_time - 3.0
        for topic in self.buffers:
            if len(self.buffers[topic]) > self.max_buffer_size:
                self.buffers[topic] = [
                    (msg, t) for msg, t in self.buffers[topic] if t > cutoff
                ]
    
    def _process_batch(self):
        """동기화된 메시지 배치 처리"""
        processed = 0
        
        # ego odom을 기준으로 처리
        ego_msgs = [(msg, t) for msg, t in self.buffers["/car_state/odom"]
                   if self.sync_start <= t <= self.sync_end]
        
        if not ego_msgs:
            return 0
        
        ego_msgs.sort(key=lambda x: x[1])
        
        # 마지막 처리 시간 확인
        last_time = self.dataset[-1]['time'] if self.dataset else 0
        
        for ego_msg, ego_time in ego_msgs:
            ego_header_time = ego_msg.header.stamp.sec + ego_msg.header.stamp.nanosec * 1e-9
            
            # 중복 방지 및 최소 간격 확인
            if ego_header_time <= last_time:
                continue
            if self.dataset and (ego_header_time - self.dataset[-1]['time']) < self.min_interval * 0.8:
                continue
            
            # 동기화된 메시지 찾기
            synced = self._find_synced_messages(ego_time)
            
            if synced:
                scan_msg, _ = synced["/scan"]
                opp_msg, _ = synced["/opp/car_state/odom"]
                
                if self._process_data(scan_msg, ego_msg, opp_msg, ego_header_time):
                    processed += 1
        
        return processed
    
    def _find_synced_messages(self, ref_time):
        """기준 시간에 가장 가까운 동기화된 메시지들 찾기"""
        synced = {}
        
        for topic in ["/scan", "/opp/car_state/odom"]:
            closest = self._find_closest(topic, ref_time)
            if closest is None:
                return None
            synced[topic] = closest
        
        return synced
    
    def _find_closest(self, topic, ref_time):
        """특정 토픽에서 기준 시간에 가장 가까운 메시지 찾기"""
        closest = None
        min_diff = float('inf')
        
        for msg, timestamp in self.buffers[topic]:
            if timestamp < self.sync_start or timestamp > self.sync_end:
                continue
            
            diff = abs(timestamp - ref_time)
            if diff <= self.sync_tolerance and diff < min_diff:
                min_diff = diff
                closest = (msg, timestamp)
        
        return closest
    
    def _process_data(self, scan_msg, ego_msg, opp_msg, ego_time):
        """동기화된 데이터 처리"""
        try:
            # opponent 위치를 LiDAR 좌표계로 변환
            opp_pos, opp_vel, opp_yaw = self._transform_to_lidar(opp_msg, ego_msg, ego_time)
            
            if opp_pos is not None:
                entry = {
                    'time': ego_time,
                    'x': float(opp_pos[0]),
                    'y': float(opp_pos[1]),
                    'vx': float(opp_vel[0]),
                    'vy': float(opp_vel[1]),
                    'yaw': float(opp_yaw)
                }
                self.dataset.append(entry)
                return True
        except Exception as e:
            self.get_logger().warning(f'Data processing error: {e}')
        
        return False
    
    def _transform_to_lidar(self, opp_msg, ego_msg, ego_time):
        """opponent을 LiDAR 좌표계로 변환"""
        try:
            # opponent 글로벌 위치/방향
            opp_pos_global = np.array([
                opp_msg.pose.pose.position.x,
                opp_msg.pose.pose.position.y
            ])
            
            opp_quat = np.array([
                opp_msg.pose.pose.orientation.x,
                opp_msg.pose.pose.orientation.y,
                opp_msg.pose.pose.orientation.z,
                opp_msg.pose.pose.orientation.w
            ])
            opp_yaw_global = Rotation.from_quat(opp_quat).as_euler('xyz')[2]
            
            # ego 글로벌 위치/방향
            ego_pos_global = np.array([
                ego_msg.pose.pose.position.x,
                ego_msg.pose.pose.position.y
            ])
            
            ego_quat = np.array([
                ego_msg.pose.pose.orientation.x,
                ego_msg.pose.pose.orientation.y,
                ego_msg.pose.pose.orientation.z,
                ego_msg.pose.pose.orientation.w
            ])
            ego_yaw_global = Rotation.from_quat(ego_quat).as_euler('xyz')[2]
            
            # 좌표 변환: 글로벌 -> base_link -> LiDAR
            relative_pos = opp_pos_global - ego_pos_global
            
            # ego 방향만큼 회전 (글로벌 -> base_link)
            cos_yaw = math.cos(-ego_yaw_global)
            sin_yaw = math.sin(-ego_yaw_global)
            
            opp_pos_baselink = np.array([
                relative_pos[0] * cos_yaw - relative_pos[1] * sin_yaw,
                relative_pos[0] * sin_yaw + relative_pos[1] * cos_yaw
            ])
            
            # base_link -> LiDAR (오프셋 적용)
            opp_pos_lidar = np.array([
                opp_pos_baselink[0] - self.lidar_offset_x,
                opp_pos_baselink[1] - self.lidar_offset_y
            ])
            
            # yaw 변환
            opp_yaw_lidar = opp_yaw_global - ego_yaw_global
            opp_yaw_lidar = ((opp_yaw_lidar + math.pi) % (2 * math.pi)) - math.pi
            
            # 속도 계산
            opp_vel_lidar = self._calculate_velocity(opp_pos_lidar, ego_time)
            
            return opp_pos_lidar, opp_vel_lidar, opp_yaw_lidar
            
        except Exception as e:
            self.get_logger().error(f'Transform error: {e}')
            return None, None, None
    
    def _calculate_velocity(self, current_pos, current_time):
        """시간 차분을 이용한 속도 계산"""
        try:
            if self.prev_opp_pos is not None and self.prev_opp_time is not None:
                dt = current_time - self.prev_opp_time
                
                if 0.01 < dt < 0.1:  # 적절한 시간 간격
                    vel = (current_pos - self.prev_opp_pos) / dt
                    
                    # 속도 제한 (20 m/s)
                    if np.linalg.norm(vel) > 20.0:
                        vel = np.array([0.0, 0.0])
                    
                    self.prev_opp_pos = current_pos.copy()
                    self.prev_opp_time = current_time
                    return vel
            
            # 첫 번째 계산이거나 부적절한 시간 간격
            self.prev_opp_pos = current_pos.copy()
            self.prev_opp_time = current_time
            return np.array([0.0, 0.0])
            
        except Exception:
            return np.array([0.0, 0.0])
    
    def save_csv(self):
        """CSV 파일로 저장"""
        if not self.dataset:
            self.get_logger().error('No data to save')
            return False
        
        # 출력 디렉토리 생성
        output_dir = os.path.dirname(self.output_csv)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        try:
            with open(self.output_csv, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['time', 'x', 'y', 'vx', 'vy', 'yaw'])
                
                for entry in self.dataset:
                    writer.writerow([
                        f"{entry['time']:.6f}",
                        f"{entry['x']:.6f}",
                        f"{entry['y']:.6f}",
                        f"{entry['vx']:.6f}",
                        f"{entry['vy']:.6f}",
                        f"{entry['yaw']:.6f}"
                    ])
            
            self.get_logger().info(f'CSV saved: {self.output_csv}')
            self.get_logger().info(f'Total entries: {len(self.dataset)}')
            
            # 통계 출력
            self._print_stats()
            return True
            
        except Exception as e:
            self.get_logger().error(f'Save error: {e}')
            return False
    
    def _print_stats(self):
        """데이터 통계 출력"""
        if len(self.dataset) < 2:
            return
        
        # 시간 간격 통계
        intervals = [self.dataset[i]['time'] - self.dataset[i-1]['time'] 
                    for i in range(1, len(self.dataset))]
        avg_interval = np.mean(intervals)
        avg_freq = 1.0 / avg_interval if avg_interval > 0 else 0
        
        # 거리/속도 통계
        distances = [math.sqrt(e['x']**2 + e['y']**2) for e in self.dataset]
        velocities = [math.sqrt(e['vx']**2 + e['vy']**2) for e in self.dataset]
        
        self.get_logger().info(f'Average frequency: {avg_freq:.1f} Hz')
        self.get_logger().info(f'Distance range: {min(distances):.2f} - {max(distances):.2f} m')
        self.get_logger().info(f'Velocity range: {min(velocities):.2f} - {max(velocities):.2f} m/s')


def main():
    parser = argparse.ArgumentParser(description='Convert ROS2 bag to CSV')
    parser.add_argument('bag_file', help='Input bag file (.db3)')
    parser.add_argument('--output', help='Output CSV file')
    parser.add_argument('--start', type=float, default=0, help='Start time (sec)')
    parser.add_argument('--end', type=float, help='End time (sec)')
    parser.add_argument('--lidar-x', type=float, default=0.287, help='LiDAR X offset')
    parser.add_argument('--lidar-y', type=float, default=0.0, help='LiDAR Y offset')
    parser.add_argument('--lidar-z', type=float, default=0.115, help='LiDAR Z offset')
    
    args = parser.parse_args()
    
    # 출력 파일명 설정
    if not args.output:
        bag_name = os.path.basename(args.bag_file).split('.')[0]
        args.output = f'{bag_name}_opponent_data.csv'
    
    # bag 파일 존재 확인
    if not os.path.exists(args.bag_file):
        print(f'Error: Bag file not found: {args.bag_file}')
        return
    
    print(f'Input: {args.bag_file}')
    print(f'Output: {args.output}')
    print(f'Time range: {args.start} - {args.end if args.end else "end"}')
    print(f'LiDAR offset: ({args.lidar_x}, {args.lidar_y}, {args.lidar_z})')
    
    # ROS2 초기화
    rclpy.init()
    
    try:
        converter = BagToCSV(
            bag_path=args.bag_file,
            output_csv=args.output,
            start_time=args.start,
            end_time=args.end,
            lidar_offset_x=args.lidar_x,
            lidar_offset_y=args.lidar_y,
            lidar_offset_z=args.lidar_z
        )
        
        if converter.process_bag():
            print('Bag processing completed')
            if converter.save_csv():
                print('CSV conversion successful')
            else:
                print('CSV save failed')
        else:
            print('Bag processing failed')
    
    except KeyboardInterrupt:
        print('Interrupted by user')
    except Exception as e:
        print(f'Error: {e}')
    
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
