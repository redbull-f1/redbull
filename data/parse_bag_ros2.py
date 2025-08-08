#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import sqlite3
import numpy as np
import csv
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.spatial.transform import Rotation
import os
import argparse

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans' 
plt.rcParams['axes.unicode_minus'] = False
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_msgs.msg import TFMessage
from visualization_msgs.msg import Marker, MarkerArray
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions, SequentialWriter
from rclpy.serialization import deserialize_message, serialize_message
from rosidl_runtime_py.utilities import get_message
import tf2_py as tf2
from builtin_interfaces.msg import Time
import math


class RosbagDatasetROS2(Node):
    """
    ROS2에서 db3 bag 파일을 처리하여 opponent 차량의 데이터를 ego 차량의 LiDAR 좌표계로 변환해서 csv 파일로 저장하는 클래스
    csv 파일 형식: (x,y,vx,vy,yaw)
    - LiDAR 데이터: /scan
    - Ego 차량 상태: /car_state/odom
    - Opponent 차량 상태: /opp/car_state/odom
    - TF 메시지: /tf, /tf_static
    - 시간 동기화: 두 bag 파일의 겹치는 시간 구간만 병합
    - 데이터셋 형식: CSV 파일로 저장
    - 데이터 시각화: matplotlib을 이용하여 LiDAR 데이터와 opponent 차량 위치를 시각화
    - 속도 계산: 시간 차분을 이용하여 opponent 차량의 속도를 계산   
    
    src/redbull/example_single_opp_converted_ros2_bag_with_visualization_1754531942/example_single_opp_converted_ros2_bag_with_visualization_1754531942_0.db3
    이 bag파일에서 좌표계 변환 잘 되는거 확인함 

    실행 코드 
    python3 parse_bag_ros2.py --bag-file /path/to/your.bag --output-file output.csv --start-time 0 --end-time 10 --create_bag
    python3 parse_bag_ros2.py /path/to/your.bag --create_bag
    """
    
    def __init__(self, bag_file_path, output_file, start_time=0, end_time=None, 
                 lidar_offset_x=0.287, lidar_offset_y=0.0, lidar_offset_z=0.115):
        super().__init__('rosbag_dataset_ros2')
        
        self.bag_file_path = bag_file_path
        self.output_file = output_file
        self.start_time = start_time if start_time is not None else 0
        self.end_time = end_time
        
        # LiDAR 센서 오프셋 (base_link → laser 프레임, tf_static에서 확인된 실제 값)
        self.lidar_offset_x = lidar_offset_x  # 전방 거리: 0.28m (tf_static 확인)
        self.lidar_offset_y = lidar_offset_y  # 좌우 거리: 0.0m (중앙)
        self.lidar_offset_z = lidar_offset_z  # 높이: 0.11m (tf_static 확인)
        
        # Topics to process (bag 파일에 실제로 있는 토픽들만)
        self.topics = ["/scan", "/car_state/odom", "/tf", "/opp/car_state/odom"]
        
        # 처리하지 않을 메시지 타입들 (문제가 있는 타입들)
        self.skip_message_types = [
            'rosgraph_msgs/msg/Log',
            'rcl_interfaces/msg/Log',
            'visualization_msgs/msg/MarkerArray'  # 역직렬화 문제가 있는 MarkerArray
        ]
        
        # Data storage
        self.dataset = []
        self.initial_time = None
        
        # TF buffer for coordinate transformations
        self.tf_buffer = tf2.BufferCore()
        
        # Previous opponent position for velocity calculation
        self.prev_opp_pos = None
        self.prev_opp_time = None
        self.prev_opp_yaw = None
        
        # 시간 동기화 설정 (80Hz 기준) - 메모리 최적화
        self.sync_tolerance = 0.02  # 20ms 허용 오차
        self.hz_80_interval = 1.0 / 80.0  # 80Hz 간격 (약 14.3ms)
        self.message_buffer = {
            "/scan": [],
            "/car_state/odom": [],
            "/opp/car_state/odom": []
        }
        
        # 메모리 최적화를 위한 설정
        self.max_buffer_size = 100  # 최대 버퍼 크기 제한
        self.cleanup_interval = 3.0  # 3초마다 정리
        self.last_cleanup_time = None
        
        # 시간 범위 동기화를 위한 변수
        self.sync_start_time = None
        self.sync_end_time = None
        self.time_sync_calculated = False
        
        # RViz2 시각화를 위한 새 bag 파일 생성 설정
        self.create_visualization_bag = True
        self.output_bag_path = None
        self.bag_writer = None
        self.marker_id_counter = 0
        
    def process_bag(self):
        """
        ROS2 bag 파일(db3)을 읽어서 데이터를 처리하고 시각화용 새 bag 파일 생성 (메모리 최적화 버전)
        """
        # 출력 bag 파일 경로 설정
        if self.create_visualization_bag:
            bag_name = os.path.basename(self.bag_file_path).split('.')[0]
            import time
            timestamp = int(time.time())
            self.output_bag_path = f'/home/harry/ros2_ws/src/redbull/bag/{bag_name}_with_visualization_{timestamp}'
            self.setup_output_bag()
        
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
            if self.create_visualization_bag:
                self.get_logger().info(f'시각화 bag 파일 생성: {self.output_bag_path}')
            
            message_count = 0
            processed_count = 0
            
            while reader.has_next():
                (topic, data, timestamp) = reader.read_next()
                
                # 원본 메시지를 새 bag 파일에 복사
                if self.create_visualization_bag:
                    self.copy_message_to_output_bag(topic, data, timestamp, type_map)
                
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
                
                # 문제가 있는 메시지 타입들은 건너뛰기
                if type_map[topic] in self.skip_message_types:
                    self.get_logger().debug(f'문제가 있는 메시지 타입 건너뛰기: {topic} ({type_map[topic]})')
                    continue
                
                try:
                    msg_type = get_message(type_map[topic])
                    msg = deserialize_message(data, msg_type)
                except (AttributeError, ImportError, ValueError) as e:
                    # 알려지지 않은 메시지 타입이나 지원되지 않는 타입은 건너뛰기
                    self.get_logger().debug(f'지원되지 않는 메시지 타입 건너뛰기: {topic} ({type_map[topic]}): {e}')
                    continue
                except Exception as e:
                    # 역직렬화 오류나 기타 메시지 처리 오류는 건너뛰기
                    self.get_logger().debug(f'메시지 역직렬화 오류 건너뛰기: {topic} ({type_map[topic]}): {e}')
                    continue
                
                # TF 메시지 처리
                if topic == "/tf":
                    self.process_tf_message(msg, topic, timestamp_sec)
                    continue
                
                # 동기화 대상 토픽만 버퍼에 저장
                if topic in self.message_buffer:
                    self.message_buffer[topic].append((msg, timestamp_sec))
                    message_count += 1
                    
                    # 메모리 관리: 주기적으로 버퍼 정리
                    self.cleanup_buffers_if_needed(timestamp_sec)
                    
                    # 시간 동기화 구간 계산 (처음 한 번만)
                    if not self.time_sync_calculated and len(self.message_buffer["/car_state/odom"]) > 5 and len(self.message_buffer["/opp/car_state/odom"]) > 5:
                        self.calculate_sync_time_range()
                    
                    # 동기화 구간이 계산된 후에만 처리
                    if self.time_sync_calculated:
                        # 배치 처리로 메모리 효율화
                        if message_count % 20 == 0:  # 20개마다 처리
                            batch_processed = self.process_synchronized_messages_batch()
                            processed_count += batch_processed
                            
                            if processed_count > 0 and processed_count % 50 == 0:
                                self.get_logger().info(f'처리된 메시지: {processed_count}개, 시간: {relative_time:.3f}s')
            
            # 남은 메시지들 마지막 처리
            if self.time_sync_calculated:
                final_processed = self.process_synchronized_messages_batch()
                processed_count += final_processed
            
            self.get_logger().info(f'총 {processed_count}개의 데이터 포인트를 처리했습니다.')
            
            # 새 bag 파일 닫기
            if self.create_visualization_bag and self.bag_writer:
                del self.bag_writer
                self.get_logger().info(f'시각화 bag 파일 생성 완료: {self.output_bag_path}')
            
            return processed_count > 0
            
        except Exception as e:
            self.get_logger().error(f'Bag 파일 읽기 중 오류 발생: {e}')
            import traceback
            self.get_logger().error(f'상세 오류: {traceback.format_exc()}')
            return False
        finally:
            del reader
    
    def setup_output_bag(self):
        """
        시각화용 출력 bag 파일 설정
        """
        try:
            # 기존 디렉토리가 있으면 완전히 삭제
            if os.path.exists(self.output_bag_path):
                import shutil
                shutil.rmtree(self.output_bag_path)
                self.get_logger().info(f'기존 출력 디렉토리 삭제: {self.output_bag_path}')
                
                # 삭제 후 잠시 대기
                import time
                time.sleep(0.1)
            
            # bag writer 설정 (디렉토리는 자동 생성됨)
            storage_options = StorageOptions(uri=self.output_bag_path, storage_id='sqlite3')
            converter_options = ConverterOptions('', '')
            
            self.bag_writer = SequentialWriter()
            self.bag_writer.open(storage_options, converter_options)
            
            # MarkerArray 토픽 생성
            from rosbag2_py import TopicMetadata
            marker_topic = TopicMetadata(
                name='/opponent_markers',
                type='visualization_msgs/msg/MarkerArray',
                serialization_format='cdr'
            )
            self.bag_writer.create_topic(marker_topic)
            
            self.get_logger().info(f'출력 bag 파일 설정 완료: {self.output_bag_path}')
            
        except Exception as e:
            self.get_logger().error(f'출력 bag 파일 설정 실패: {e}')
            import traceback
            self.get_logger().error(f'상세 오류: {traceback.format_exc()}')
            self.create_visualization_bag = False
    
    def copy_message_to_output_bag(self, topic, data, timestamp, type_map):
        """
        원본 메시지를 새 bag 파일에 복사
        """
        try:
            if self.bag_writer and topic in type_map:
                # 문제가 있는 메시지 타입들은 복사하지 않음
                if type_map[topic] in self.skip_message_types:
                    return
                
                # 지원되지 않는 메시지 타입 건너뛰기
                try:
                    # 메시지 타입 확인
                    msg_type = get_message(type_map[topic])
                except (AttributeError, ImportError, ValueError):
                    # 지원되지 않는 메시지 타입은 복사하지 않음
                    return
                
                # 토픽이 아직 생성되지 않았다면 생성
                if not hasattr(self, '_created_topics'):
                    self._created_topics = set()
                
                if topic not in self._created_topics:
                    from rosbag2_py import TopicMetadata
                    topic_meta = TopicMetadata(
                        name=topic,
                        type=type_map[topic],
                        serialization_format='cdr'
                    )
                    self.bag_writer.create_topic(topic_meta)
                    self._created_topics.add(topic)
                
                # 메시지 쓰기
                self.bag_writer.write(topic, data, timestamp)
                
        except Exception as e:
            self.get_logger().debug(f'메시지 복사 중 오류: {e}')
    
    def process_tf_message(self, msg, topic, timestamp_sec):
        """
        TF 메시지를 처리하여 tf_buffer에 저장 (tf_static은 merged bag에 없으므로 tf만 처리)
        """
        try:
            for transform in msg.transforms:
                self.tf_buffer.set_transform(transform, 'default_authority')
        except Exception as e:
            self.get_logger().warning(f'TF 메시지 처리 중 오류: {e}')
    
    def write_marker_array_to_bag(self, marker_array, timestamp):
        """
        MarkerArray를 새 bag 파일에 쓰기
        """
        try:
            if self.bag_writer and marker_array:
                serialized_data = serialize_message(marker_array)
                self.bag_writer.write('/opponent_markers', serialized_data, timestamp)
                
        except Exception as e:
            self.get_logger().warning(f'MarkerArray 쓰기 중 오류: {e}')
    
    def calculate_sync_time_range(self):
        """
        전체 bag 파일을 스캔하여 겹치는 시간 구간을 계산
        """
        try:
            # 전체 bag 파일을 다시 스캔하여 시간 범위 계산
            storage_options = StorageOptions(uri=self.bag_file_path, storage_id='sqlite3')
            converter_options = ConverterOptions('', '')
            reader = SequentialReader()
            
            ego_times = []
            opp_times = []
            
            try:
                reader.open(storage_options, converter_options)
                topic_types = reader.get_all_topics_and_types()
                type_map = {}
                for topic_metadata in topic_types:
                    type_map[topic_metadata.name] = topic_metadata.type
                
                while reader.has_next():
                    (topic, data, timestamp) = reader.read_next()
                    timestamp_sec = timestamp / 1e9
                    
                    if topic == "/car_state/odom":
                        ego_times.append(timestamp_sec)
                    elif topic == "/opp/car_state/odom":
                        opp_times.append(timestamp_sec)
                        
            finally:
                del reader
            
            if not ego_times or not opp_times:
                self.get_logger().error('ego 또는 opp 메시지를 찾을 수 없습니다!')
                return
            
            ego_start = min(ego_times)
            ego_end = max(ego_times)
            opp_start = min(opp_times)
            opp_end = max(opp_times)
            
            # 겹치는 시간 구간 계산: 더 늦게 시작하고 더 빨리 끝나는 시간
            self.sync_start_time = max(ego_start, opp_start)
            self.sync_end_time = min(ego_end, opp_end)
            
            self.time_sync_calculated = True
            
            self.get_logger().info(f'전체 파일 시간 동기화 구간 계산 완료:')
            self.get_logger().info(f'  Ego 시간 범위: {ego_start:.3f}s ~ {ego_end:.3f}s (총 {len(ego_times)}개 메시지)')
            self.get_logger().info(f'  Opp 시간 범위: {opp_start:.3f}s ~ {opp_end:.3f}s (총 {len(opp_times)}개 메시지)')
            self.get_logger().info(f'  동기화 구간: {self.sync_start_time:.3f}s ~ {self.sync_end_time:.3f}s')
            self.get_logger().info(f'  동기화 구간 길이: {self.sync_end_time - self.sync_start_time:.3f}초')
            
            # 겹치는 시간이 있는지 확인
            if self.sync_start_time >= self.sync_end_time:
                self.get_logger().error('두 파일의 시간이 겹치지 않습니다!')
                self.time_sync_calculated = False
                
        except Exception as e:
            self.get_logger().warning(f'시간 동기화 구간 계산 중 오류: {e}')
            self.time_sync_calculated = False
    
    def cleanup_buffers_if_needed(self, current_time):
        """
        메모리 최적화를 위한 버퍼 정리
        """
        # 정리가 필요한 조건: 시간이 지났거나 버퍼가 너무 클 때
        need_cleanup = (
            self.last_cleanup_time is None or 
            current_time - self.last_cleanup_time > self.cleanup_interval or
            any(len(buffer) > self.max_buffer_size for buffer in self.message_buffer.values())
        )
        
        if need_cleanup:
            # 오래된 메시지 제거 (2초 이상 된 메시지)
            cutoff_time = current_time - 2.0
            
            for topic_name in self.message_buffer:
                old_size = len(self.message_buffer[topic_name])
                self.message_buffer[topic_name] = [
                    (msg, timestamp) for msg, timestamp in self.message_buffer[topic_name]
                    if timestamp > cutoff_time
                ]
                new_size = len(self.message_buffer[topic_name])
                
                if old_size > new_size:
                    self.get_logger().debug(f'{topic_name}: {old_size} → {new_size} 메시지 정리')
            
            self.last_cleanup_time = current_time

    def process_synchronized_messages_batch(self):
        """
        배치 처리로 메모리 효율화된 동기화 메시지 처리
        """
        processed_count = 0
        
        # ego odom 메시지를 기준으로 동기화
        if not self.message_buffer["/car_state/odom"]:
            return 0
        
        # 동기화 구간 내의 ego odom 메시지들만 처리
        ego_messages = [(msg, time) for msg, time in self.message_buffer["/car_state/odom"]
                       if self.sync_start_time <= time <= self.sync_end_time]
        
        if not ego_messages:
            return 0
        
        # 시간순 정렬
        ego_messages.sort(key=lambda x: x[1])
        
        # 마지막 처리 시간 계산 (/car_state/odom 헤더 타임스탬프 기반)
        last_processed_header_time = None
        if self.dataset:
            last_processed_header_time = self.dataset[-1]['time']
        
        for ego_msg, ego_time in ego_messages:
            # /car_state/odom 헤더 타임스탬프 계산
            ego_header_time = ego_msg.header.stamp.sec + ego_msg.header.stamp.nanosec * 1e-9
            
            # 이미 처리된 메시지 건너뛰기 (/car_state/odom 헤더 타임스탬프 기반)
            if last_processed_header_time and ego_header_time <= last_processed_header_time:
                continue
            
            # 80Hz 간격 체크 (/car_state/odom 헤더 타임스탬프 기반)
            if self.dataset:
                last_entry_header_time = self.dataset[-1]['time']
                time_diff = ego_header_time - last_entry_header_time
                if time_diff < self.hz_80_interval * 0.7:  # 80% 여유분
                    continue
            
            # ego 시간 기준으로 동기화된 메시지 찾기
            synced_msgs = self.find_synchronized_messages_80hz(ego_time)
            
            if synced_msgs:
                scan_msg, scan_time = synced_msgs["/scan"]
                opp_msg, opp_time = synced_msgs["/opp/car_state/odom"]
                
                # 토픽 타임스탬프 사용 (ego_time이 기준 시간)
                topic_timestamp = ego_time
                
                if self.process_synchronized_data_optimized(scan_msg, ego_msg, opp_msg, topic_timestamp):
                    processed_count += 1
                    
                    if processed_count % 20 == 0:
                        relative_time = ego_time - self.initial_time if self.initial_time else 0
                        self.get_logger().info(f'배치 처리: {processed_count}개 완료, 시간: {relative_time:.3f}s')
        
        return processed_count
    
    def find_synchronized_messages_80hz(self, reference_time):
        """
        80Hz 기준으로 기준 시간에서 동기화 허용 오차 내의 메시지들을 찾기
        """
        synced_messages = {}
        
        # scan 메시지 찾기 (80Hz 동기화 구간 내에서)
        scan_msg = self.find_closest_message_80hz("/scan", reference_time)
        if not scan_msg:
            return None
        synced_messages["/scan"] = scan_msg
        
        # opp odom 메시지 찾기 (80Hz 동기화 구간 내에서)
        opp_msg = self.find_closest_message_80hz("/opp/car_state/odom", reference_time)
        if not opp_msg:
            return None
        synced_messages["/opp/car_state/odom"] = opp_msg
        
        return synced_messages
    
    def find_closest_message_80hz(self, topic, reference_time):
        """
        80Hz 기준으로 기준 시간에 가장 가까운 메시지 찾기 (동기화 허용 오차 및 동기화 구간 내에서)
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
            
            # 80Hz 기준 동기화 허용 오차 내에서 가장 가까운 메시지 선택
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
    
    def process_synchronized_data_optimized(self, scan_msg, ego_odom_msg, opp_odom_msg, topic_timestamp):
        """
        동기화된 데이터를 처리하여 데이터셋 생성 및 MarkerArray 생성 (메모리 최적화 버전)
        """
        try:
            # LiDAR 데이터 처리 (메모리 효율적으로)
            ranges = np.array(scan_msg.ranges, dtype=np.float32)
            
            # 유효한 범위 필터링 먼저 수행 (메모리 절약)
            valid_mask = np.isfinite(ranges) & (ranges >= scan_msg.range_min) & (ranges <= scan_msg.range_max)
            
            # /car_state/odom 헤더 타임스탬프 사용
            ego_header_time = ego_odom_msg.header.stamp.sec + ego_odom_msg.header.stamp.nanosec * 1e-9
            
            # Opponent 위치를 LiDAR 좌표계로 변환
            opp_pos_lidar, opp_vel_lidar, opp_yaw_lidar = self.transform_opponent_to_lidar(
                opp_odom_msg, ego_odom_msg, ego_header_time
            )
            
            if opp_pos_lidar is not None:
                # CSV 데이터셋 엔트리 생성
                dataset_entry = {
                    'time': ego_header_time,  # /car_state/odom의 헤더 타임스탬프 사용
                    'timestamp': topic_timestamp,  # 실제 토픽 타임스탬프도 보관
                    'x': float(opp_pos_lidar[0]),
                    'y': float(opp_pos_lidar[1]),
                    'vx': float(opp_vel_lidar[0]),
                    'vy': float(opp_vel_lidar[1]),
                    'yaw': float(opp_yaw_lidar)
                }
                self.dataset.append(dataset_entry)
                
                # RViz2 시각화를 위한 MarkerArray 생성 및 저장
                if self.create_visualization_bag:
                    marker_array = self.create_marker_array(
                        opp_pos_lidar, opp_vel_lidar, opp_yaw_lidar, ego_header_time
                    )
                    
                    # bag 파일 타임스탬프: /car_state/odom의 헤더 타임스탬프와 정확히 동일하게 설정
                    bag_timestamp = int(ego_odom_msg.header.stamp.sec) * int(1e9) + int(ego_odom_msg.header.stamp.nanosec)
                    self.write_marker_array_to_bag(marker_array, bag_timestamp)
                
                return True
        
        except Exception as e:
            self.get_logger().warning(f'데이터 처리 중 오류: {e}')
        
        return False
    
    def transform_opponent_to_lidar(self, opp_odom_msg, ego_odom_msg, ego_header_time):
        """
        Opponent의 global 좌표를 ego 차량의 LiDAR 좌표계로 변환
        ego_header_time: /car_state/odom의 헤더 타임스탬프 사용
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
            
            # TF를 통한 좌표 변환 시도 (merged bag에는 tf_static이 부족할 수 있음)
            # tf_static에서 확인된 프레임: base_link → laser (x=0.28, y=0.0, z=0.11)
            try:
                # laser 프레임으로 조회 시도 (tf_static에서 확인된 정확한 프레임명)
                transform = self.tf_buffer.lookup_transform_core("laser", "map", Time())
                
                # map 좌표의 opponent 위치를 laser 좌표계로 직접 변환
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
                
                # Opponent 위치를 laser 좌표계로 변환
                opp_pos_3d = np.array([opp_pos_global[0], opp_pos_global[1], 0.0])
                opp_pos_lidar_3d = T_rot.apply(opp_pos_3d) + T_trans
                opp_pos_lidar = opp_pos_lidar_3d[:2]  # x, y만 사용
                
                # Yaw를 laser 좌표계로 변환
                opp_yaw_lidar = opp_yaw_global - ego_yaw_global
                # Yaw를 [-π, π] 범위로 정규화
                opp_yaw_lidar = ((opp_yaw_lidar + math.pi) % (2 * math.pi)) - math.pi
                
                self.get_logger().debug(f'TF 변환 성공: laser 프레임 사용 (tf_static 기반)')
                
            except Exception as tf_error:
                # TF 조회 실패 시 수동 계산 (tf_static 기반 정확한 오프셋 사용)
                self.get_logger().debug(f'TF 조회 실패, tf_static 기반 수동 계산 사용: {tf_error}')
                
                # 1단계: Global → base_link 변환
                relative_pos = opp_pos_global - ego_pos_global
                
                # Ego의 yaw만큼 회전하여 base_link 좌표계로 변환
                cos_yaw = math.cos(-ego_yaw_global)
                sin_yaw = math.sin(-ego_yaw_global)
                
                opp_pos_baselink = np.array([
                    relative_pos[0] * cos_yaw - relative_pos[1] * sin_yaw,
                    relative_pos[0] * sin_yaw + relative_pos[1] * cos_yaw
                ])
                
                # 2단계: base_link → laser 프레임 변환 (tf_static에서 확인된 정확한 오프셋)
                # tf_static: base_link → laser 변환은 x=0.28, y=0.0, z=0.11
                opp_pos_lidar = np.array([
                    opp_pos_baselink[0] - self.lidar_offset_x,  # LiDAR가 0.28m 앞에 있음
                    opp_pos_baselink[1] - self.lidar_offset_y   # 좌우 오프셋 (0.0m, 중앙)
                ])
                
                self.get_logger().debug(f'tf_static 기반 변환: baselink({opp_pos_baselink[0]:.3f}, {opp_pos_baselink[1]:.3f}) → laser({opp_pos_lidar[0]:.3f}, {opp_pos_lidar[1]:.3f})')
                
                # Yaw를 LiDAR 좌표계로 변환
                opp_yaw_lidar = opp_yaw_global - ego_yaw_global
                opp_yaw_lidar = ((opp_yaw_lidar + math.pi) % (2 * math.pi)) - math.pi
            
            # 속도 계산 (시간 차분 기반, /car_state/odom 헤더 타임스탬프 사용)
            opp_vel_lidar = self.calculate_opponent_velocity(
                opp_pos_lidar, opp_yaw_lidar, ego_header_time
            )
            
            return opp_pos_lidar, opp_vel_lidar, opp_yaw_lidar
            
        except Exception as e:
            self.get_logger().error(f'좌표 변환 중 오류: {e}')
            return None, None, None
    
    def calculate_opponent_velocity(self, opp_pos_lidar, opp_yaw_lidar, ego_header_time):
        """
        시간 차분을 이용한 opponent 속도 계산 (/car_state/odom 헤더 타임스탬프 기반)
        """
        try:
            if self.prev_opp_pos is not None and self.prev_opp_time is not None:
                dt = ego_header_time - self.prev_opp_time
                
                if 0.01 < dt < 0.1:  # 합리적인 시간 간격 (10ms ~ 100ms)
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
                    
                    # 이전 값 업데이트 (/car_state/odom 헤더 타임스탬프 사용)
                    self.prev_opp_pos = opp_pos_lidar.copy()
                    self.prev_opp_time = ego_header_time
                    self.prev_opp_yaw = opp_yaw_lidar
                    
                    return np.array([vx, vy])
            
            # 첫 번째 계산이거나 시간 간격이 부적절한 경우
            self.prev_opp_pos = opp_pos_lidar.copy()
            self.prev_opp_time = ego_header_time
            self.prev_opp_yaw = opp_yaw_lidar
            
            return np.array([0.0, 0.0])
            
        except Exception as e:
            self.get_logger().warning(f'속도 계산 중 오류: {e}')
            return np.array([0.0, 0.0])
    
    def save_to_csv(self):
        """
        데이터셋을 간단한 CSV 파일로 저장 (time,x,y,vx,vy,yaw 형식)
        """
        if not self.dataset:
            self.get_logger().error('저장할 데이터가 없습니다!')
            return False
        
        # 출력 디렉토리 생성
        output_dir = os.path.dirname(self.output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        try:
            with open(self.output_file, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                
                # 헤더 작성
                writer.writerow(['time', 'x', 'y', 'vx', 'vy', 'yaw'])
                
                # 80Hz로 동기화된 데이터 저장 (메모리 효율적)
                for entry in self.dataset:
                    writer.writerow([
                        f"{entry['time']:.6f}",
                        f"{entry['x']:.6f}",
                        f"{entry['y']:.6f}",
                        f"{entry['vx']:.6f}",
                        f"{entry['vy']:.6f}",
                        f"{entry['yaw']:.6f}"
                    ])
            
            self.get_logger().info(f'80Hz 동기화 CSV 파일 저장 완료: {self.output_file}')
            self.get_logger().info(f'총 {len(self.dataset)}개의 80Hz 동기화 엔트리가 저장되었습니다.')
            
            # 시간 간격 통계 출력
            if len(self.dataset) > 1:
                time_diffs = []
                for i in range(1, len(self.dataset)):
                    dt = self.dataset[i]['time'] - self.dataset[i-1]['time']
                    time_diffs.append(dt)
                
                avg_dt = np.mean(time_diffs)
                avg_hz = 1.0 / avg_dt if avg_dt > 0 else 0
                
                self.get_logger().info(f'평균 시간 간격: {avg_dt:.4f}s (약 {avg_hz:.1f}Hz)')
                self.get_logger().info(f'목표 80Hz 간격: {self.hz_80_interval:.4f}s')
            
            return True
            
        except Exception as e:
            self.get_logger().error(f'CSV 저장 중 오류: {e}')
            return False
    
    def create_opponent_marker(self, opp_pos_lidar, opp_vel_lidar, opp_yaw_lidar, ego_header_time):
        """
        Opponent 차량을 시각화하기 위한 Marker 생성
        """
        marker = Marker()
        marker.header.frame_id = "laser"  # LiDAR 프레임 기준
        marker.header.stamp.sec = int(ego_header_time)
        marker.header.stamp.nanosec = int((ego_header_time - int(ego_header_time)) * 1e9)
        
        marker.ns = "opponent_vehicle"
        marker.id = self.marker_id_counter
        self.marker_id_counter += 1
        
        marker.type = Marker.CUBE  # 박스 형태로 차량 표현
        marker.action = Marker.ADD
        
        # 위치 설정 (LiDAR 좌표계 기준)
        marker.pose.position.x = float(opp_pos_lidar[0])
        marker.pose.position.y = float(opp_pos_lidar[1])
        marker.pose.position.z = 0.0
        
        # 방향 설정 (yaw 각도를 quaternion으로 변환)
        from scipy.spatial.transform import Rotation as R
        rot = R.from_euler('z', opp_yaw_lidar)
        quat = rot.as_quat()  # [x, y, z, w]
        marker.pose.orientation.x = quat[0]
        marker.pose.orientation.y = quat[1]
        marker.pose.orientation.z = quat[2]
        marker.pose.orientation.w = quat[3]
        
        # 크기 설정 (일반적인 차량 크기)
        marker.scale.x = 0.3  # 길이
        marker.scale.y = 0.2  # 폭
        marker.scale.z = 0.3  # 높이
        
        # 색상 설정 (빨간색)
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 0.5  # 투명도
        
        # 지속 시간 설정
        marker.lifetime.sec = 0
        marker.lifetime.nanosec = 100000000  # 0.1초
        
        return marker
    
    def create_velocity_arrow_marker(self, opp_pos_lidar, opp_vel_lidar, ego_header_time):
        """
        Opponent 차량의 속도를 화살표로 시각화하기 위한 Marker 생성
        """
        velocity_magnitude = math.sqrt(opp_vel_lidar[0]**2 + opp_vel_lidar[1]**2)
        
        if velocity_magnitude < 0.1:  # 속도가 너무 작으면 화살표 표시 안함
            return None
        
        marker = Marker()
        marker.header.frame_id = "laser"
        marker.header.stamp.sec = int(ego_header_time)
        marker.header.stamp.nanosec = int((ego_header_time - int(ego_header_time)) * 1e9)
        
        marker.ns = "opponent_velocity"
        marker.id = self.marker_id_counter
        self.marker_id_counter += 1
        
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        
        # 화살표 시작점 (차량 위치)
        marker.pose.position.x = float(opp_pos_lidar[0])
        marker.pose.position.y = float(opp_pos_lidar[1])
        marker.pose.position.z = 0.5  # 차량 위쪽에 표시
        
        # 속도 방향으로 화살표 방향 설정
        velocity_angle = math.atan2(opp_vel_lidar[1], opp_vel_lidar[0])
        from scipy.spatial.transform import Rotation as R
        rot = R.from_euler('z', velocity_angle)
        quat = rot.as_quat()
        marker.pose.orientation.x = quat[0]
        marker.pose.orientation.y = quat[1]
        marker.pose.orientation.z = quat[2]
        marker.pose.orientation.w = quat[3]
        
        # 화살표 크기 (속도에 비례)
        arrow_length = min(velocity_magnitude * 0.5, 3.0)  # 최대 3m
        marker.scale.x = arrow_length  # 길이
        marker.scale.y = 0.2  # 폭
        marker.scale.z = 0.2  # 높이
        
        # 색상 설정 (파란색)
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = 0.8
        
        marker.lifetime.sec = 0
        marker.lifetime.nanosec = 100000000  # 0.1초
        
        return marker
    
    def create_marker_array(self, opp_pos_lidar, opp_vel_lidar, opp_yaw_lidar, ego_header_time):
        """
        Opponent 차량과 속도 벡터를 포함한 MarkerArray 생성
        """
        marker_array = MarkerArray()
        
        # Opponent 차량 마커
        vehicle_marker = self.create_opponent_marker(opp_pos_lidar, opp_vel_lidar, opp_yaw_lidar, ego_header_time)
        if vehicle_marker is not None:
            marker_array.markers.append(vehicle_marker)
        
        # 속도 화살표 마커
        velocity_marker = self.create_velocity_arrow_marker(opp_pos_lidar, opp_vel_lidar, ego_header_time)
        if velocity_marker is not None:
            marker_array.markers.append(velocity_marker)
        
        return marker_array
    
    def show_rviz_instructions(self):
        """
        RViz2에서 시각화하는 방법 안내
        """
        if not self.output_bag_path:
            self.get_logger().warning('시각화 bag 파일이 생성되지 않았습니다!')
            return
        
        print("\n" + "="*60)
        print("🚗 RViz2에서 Opponent 차량 시각화하기")
        print("="*60)
        print(f"1. 생성된 bag 파일: {self.output_bag_path}")
        print()
        print("2. RViz2에서 시각화하는 방법:")
        print("   # 터미널 1: bag 파일 재생")
        print(f"   ros2 bag play {self.output_bag_path}")
        print()
        print("   # 터미널 2: RViz2 실행")
        print("   rviz2")
        print()
        print("3. RViz2 설정:")
        print("   - Fixed Frame: 'laser'로 설정")
        print("   - Add → By topic → /scan → LaserScan (LiDAR 데이터)")
        print("   - Add → By topic → /opponent_markers → MarkerArray (상대방 차량)")
        print()
        print("4. 시각화 요소:")
        print("   - 🔴 빨간 박스: Opponent 차량 위치 및 방향")
        print("   - 🔵 파란 화살표: Opponent 차량 속도 벡터")
        print("   - ⚪ 점들: LiDAR 스캔 데이터")
        print()
        print("5. 추가 정보:")
        print(f"   - 총 처리된 데이터 포인트: {len(self.dataset)}개")
        if len(self.dataset) > 0:
            start_time = self.dataset[0]['time']
            end_time = self.dataset[-1]['time']
            print(f"   - 시간 범위: {start_time:.3f}s ~ {end_time:.3f}s")
            print(f"   - 총 길이: {end_time - start_time:.3f}초")
        print("="*60)
        print()
        
        # 간단한 통계 출력
        if len(self.dataset) > 1:
            velocities = [math.sqrt(entry['vx']**2 + entry['vy']**2) for entry in self.dataset]
            distances = [math.sqrt(entry['x']**2 + entry['y']**2) for entry in self.dataset]
            
            print("📊 데이터 통계:")
            print(f"   - 평균 거리: {np.mean(distances):.2f}m")
            print(f"   - 최소/최대 거리: {np.min(distances):.2f}m / {np.max(distances):.2f}m")
            print(f"   - 평균 속도: {np.mean(velocities):.2f}m/s")
            print(f"   - 최대 속도: {np.max(velocities):.2f}m/s")
            print()
import time

def main():
    parser = argparse.ArgumentParser(description='ROS2 bag 파일에서 opponent 차량 데이터를 추출하여 CSV로 저장하고 RViz2 시각화용 bag 파일 생성')
    parser.add_argument('bag_file', nargs='?', type=str, help='ROS2 bag 파일 경로 (db3) - positional argument')
    parser.add_argument('--bag', type=str, help='ROS2 bag 파일 경로 (db3) - optional argument')
    parser.add_argument('--output', type=str, help='출력 CSV 파일 경로')
    parser.add_argument('--start', type=float, default=0, help='시작 시간 (초)')
    parser.add_argument('--end', type=float, help='종료 시간 (초)')
    parser.add_argument('--create-bag', '--create_bag', action='store_true', help='RViz2 시각화용 bag 파일 생성 및 방법 안내')
    parser.add_argument('--visualize', action='store_true', help='RViz2 시각화 방법 안내')
    parser.add_argument('--no-bag-output', action='store_true', help='시각화용 bag 파일 생성 비활성화')
    
    # LiDAR 센서 오프셋 파라미터 (tf_static에서 확인된 실제 값)
    parser.add_argument('--lidar-x', type=float, default=0.287, help='LiDAR X 오프셋 (base_link 기준, 미터) - tf_static 확인값')
    parser.add_argument('--lidar-y', type=float, default=0.0, help='LiDAR Y 오프셋 (base_link 기준, 미터)')
    parser.add_argument('--lidar-z', type=float, default=0.115, help='LiDAR Z 오프셋 (base_link 기준, 미터) - tf_static 확인값')

    args = parser.parse_args()
    
    # bag 파일 경로 결정 (positional argument 우선, 없으면 --bag 옵션 사용)
    bag_file_path = args.bag_file or args.bag
    if not bag_file_path:
        parser.error('bag 파일 경로가 필요합니다. positional argument 또는 --bag 옵션을 사용하세요.')
    
    # --create-bag 옵션이 사용되면 시각화 활성화
    if args.create_bag:
        args.visualize = True
    
    # 출력 경로 설정
    if args.output is None:
        bag_name = os.path.basename(bag_file_path).split('.')[0]
        args.output = f'/home/harry/ros2_ws/src/redbull/csv/{bag_name}_opponent_local_gt{time.time()}.csv'
    
    # bag 파일 존재 확인
    if not os.path.exists(bag_file_path):
        print(f'Error: bag 파일을 찾을 수 없습니다: {bag_file_path}')
        return
    
    # ROS2 초기화   
    rclpy.init()
    
    try:
        # 데이터셋 생성기 생성 및 실행
        dataset_generator = RosbagDatasetROS2(
            bag_file_path, args.output, args.start, args.end,
            args.lidar_x, args.lidar_y, args.lidar_z
        )
        
        # 시각화용 bag 파일 생성 비활성화 옵션
        if args.no_bag_output:
            dataset_generator.create_visualization_bag = False
        
        print(f'🚗 ROS2 bag 파일 처리 시작: {bag_file_path}')
        print(f'📍 LiDAR 오프셋 (tf_static 기반): x={args.lidar_x}m, y={args.lidar_y}m, z={args.lidar_z}m')
        
        if dataset_generator.process_bag():
            print('✅ 데이터 처리 완료')
            
            print('💾 CSV 파일 저장 중...')
            if dataset_generator.save_to_csv():
                print(f'✅ CSV 파일 저장 완료: {args.output}')
                
                # 시각화 옵션
                if args.visualize:
                    print('🎯 RViz2 시각화 방법 안내:')
                    dataset_generator.show_rviz_instructions()
                else:
                    print('💡 RViz2에서 시각화하려면 --visualize 옵션을 사용하세요')
            else:
                print('❌ CSV 파일 저장에 실패했습니다.')
        else:
            print('❌ bag 파일 처리에 실패했습니다.')
    
    except Exception as e:
        print(f'❌ 오류 발생: {e}')
        import traceback
        print(f'상세 오류: {traceback.format_exc()}')
    
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
