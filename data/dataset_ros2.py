#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import csv
from sensor_msgs.msg import LaserScan
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import math


class StaticBoxDetector(Node):
    """
    ROS2 클래스로 db3 파일에서 /scan 토픽을 읽어 정적 박스 데이터를 디텍하고 추출해서 CSV로 저장
    """
    
    def __init__(self, bag_path, output_path):
        super().__init__('static_box_detector')
        
        self.bag_path = bag_path
        self.output_path = output_path
        
        # LiDAR 파라미터 (일반적인 값들, 필요시 수정)
        self.angle_min = -2.356194496154785  # -135도
        self.angle_max = 2.356194496154785   # 135도
        self.angle_increment = 0.004363323096185923
        self.range_min = 0.0
        self.range_max = 30.0
        
        # 정적 박스 검출을 위한 파라미터
        self.lambda_angle = 10  # degree를 radian으로 변환 필요
        self.sigma = 0.03
        self.min_obs_size = 5
        self.max_obs_size = 50
        self.min_2_points_dist = 0.1
        
        # 데이터 저장용
        self.scan_data = []
        self.detected_boxes = []
        self.previous_box_position = None
        self.previous_timestamp = None
        
        # 속도 계산을 위한 파라미터
        self.dt_threshold = 0.5  # 시간 차이가 너무 크면 속도 계산 안함 (초)
        self.position_threshold = 2.0  # 위치 변화가 너무 크면 잘못된 검출로 간주 (미터)
        
    def read_bag(self):
        """
        ROS2 bag 파일(db3)을 읽어서 /scan 토픽 데이터를 추출
        """
        storage_options = StorageOptions(uri=self.bag_path, storage_id='sqlite3')
        converter_options = ConverterOptions('', '')
        
        reader = SequentialReader()
        
        try:
            reader.open(storage_options, converter_options)
            
            topic_types = reader.get_all_topics_and_types()
            
            # /scan 토픽의 타입 찾기
            scan_type = None
            for topic_metadata in topic_types:
                if topic_metadata.name == '/scan':
                    scan_type = topic_metadata.type
                    break
            
            if scan_type is None:
                self.get_logger().error('/scan 토픽을 찾을 수 없습니다!')
                return False
            
            # 메시지 타입 가져오기
            msg_type = get_message(scan_type)
            
            scan_count = 0
            while reader.has_next():
                (topic, data, timestamp) = reader.read_next()
                
                if topic == '/scan':
                    msg = deserialize_message(data, msg_type)
                    # timestamp를 초 단위로 변환 (nanoseconds -> seconds)
                    timestamp_sec = timestamp / 1e9
                    self.process_scan(msg, scan_count, timestamp_sec)
                    scan_count += 1
                    
                    if scan_count % 100 == 0:
                        self.get_logger().info(f'처리된 스캔: {scan_count}개')
            
            self.get_logger().info(f'총 {scan_count}개의 스캔을 처리했습니다.')
            return True
            
        except Exception as e:
            self.get_logger().error(f'Bag 파일 읽기 중 오류 발생: {e}')
            return False
        finally:
            # ROS2의 SequentialReader는 close() 메서드가 없을 수 있음
            # del을 사용하여 객체를 정리
            del reader
    
    def process_scan(self, scan_msg, scan_index, timestamp):
        """
        개별 LaserScan 메시지를 처리하여 정적 박스를 검출
        
        Args:
            scan_msg: LaserScan 메시지
            scan_index: 스캔 인덱스
            timestamp: 타임스탬프 (초 단위)
        """
        ranges = np.array(scan_msg.ranges)
        intensities = np.array(scan_msg.intensities) if scan_msg.intensities else np.full_like(ranges, 0.5)
        
        # 유효하지 않은 값들 필터링
        valid_indices = np.isfinite(ranges) & (ranges >= scan_msg.range_min) & (ranges <= scan_msg.range_max)
        valid_ranges = ranges[valid_indices]
        valid_intensities = intensities[valid_indices]
        
        # 각도 계산
        angles = np.linspace(scan_msg.angle_min, scan_msg.angle_max, len(ranges))
        valid_angles = angles[valid_indices]
        
        # 극좌표를 직교좌표로 변환
        x_points = valid_ranges * np.cos(valid_angles)
        y_points = valid_ranges * np.sin(valid_angles)
        
        # ROI 필터링: LiDAR 위치에서 x 방향으로 -0.1m 뒤쪽 데이터 제거
        roi_mask = x_points >= -0.1  # x >= -0.1m 조건
        x_points_roi = x_points[roi_mask]
        y_points_roi = y_points[roi_mask]
        valid_ranges_roi = valid_ranges[roi_mask]
        valid_intensities_roi = valid_intensities[roi_mask]
        valid_angles_roi = valid_angles[roi_mask]
        
        # ROI 필터링 후 포인트가 충분한지 확인
        if len(x_points_roi) < self.min_obs_size:
            # ROI 내에 충분한 포인트가 없으면 박스 검출 안함
            csv_entry = {
                'lidar': valid_ranges.tolist(),
                'intensities': [0.5] * len(valid_ranges),
                'x': 0.0,
                'y': 0.0,
                'vx': 0.0,
                'vy': 0.0,
                'yaw': 0.0
            }
            self.scan_data.append(csv_entry)
            return
        
        # 포인트 클라우드 생성 (ROI 적용)
        laser_points = np.vstack([x_points_roi, y_points_roi])
        
        # 정적 박스 검출
        detected_box = self.detect_static_box(laser_points, valid_ranges_roi, valid_intensities_roi, valid_angles_roi)
        
        # 속도 계산
        vx, vy = self.calculate_velocity(detected_box, timestamp)
        
        # 위치 변화 기반 yaw 계산
        if not hasattr(self, 'prev_x'):
            self.prev_x = None
            self.prev_y = None

        if detected_box is not None and self.prev_x is not None and self.prev_y is not None:
            # 위치 변화 계산
            dx = detected_box['x'] - self.prev_x
            dy = detected_box['y'] - self.prev_y
            
            if dx == 0.0 and dy == 0.0:
                yaw = 0.0
            else:
                yaw = math.atan2(dy, dx)
        else:
            yaw = 0.0

        # 다음 프레임을 위해 현재 위치 저장
        if detected_box is not None:
            self.prev_x = detected_box['x']
            self.prev_y = detected_box['y']

        if detected_box is not None:
            csv_entry = {
                'lidar': valid_ranges.tolist(),
                'intensities': [0.5] * len(valid_ranges),
                'x': detected_box['x'],
                'y': detected_box['y'],
                'vx': vx,
                'vy': vy,
                'yaw': yaw
            }
            self.scan_data.append(csv_entry)
            self.detected_boxes.append(detected_box)
        else:
            # 박스가 검출되지 않은 경우에도 스캔 데이터는 저장 (위치와 속도는 0으로)
            csv_entry = {
                'lidar': valid_ranges.tolist(),
                'intensities': [0.5] * len(valid_ranges),
                'x': 0.0,
                'y': 0.0,
                'vx': 0.0,
                'vy': 0.0,
                'yaw': 0.0
            }
            self.scan_data.append(csv_entry)
    
    def calculate_velocity(self, detected_box, timestamp):
        """
        이전 박스 위치와 현재 박스 위치를 비교하여 속도를 계산
        
        Args:
            detected_box: 현재 검출된 박스 정보 (None일 수 있음)
            timestamp: 현재 타임스탬프 (초 단위)
        
        Returns:
            tuple: (vx, vy) LiDAR 좌표계 기준 속도 (m/s)
        """
        vx, vy = 0.0, 0.0
        
        if detected_box is not None and self.previous_box_position is not None and self.previous_timestamp is not None:
            # 시간 차이 계산
            dt = timestamp - self.previous_timestamp
            
            # 시간 차이가 너무 크거나 작으면 속도 계산 안함
            if 0.01 < dt < self.dt_threshold:
                # 위치 변화 계산
                dx = detected_box['x'] - self.previous_box_position['x']
                dy = detected_box['y'] - self.previous_box_position['y']
                
                # 위치 변화가 합리적인 범위 내에 있는지 확인
                position_change = np.sqrt(dx**2 + dy**2)
                if position_change < self.position_threshold:
                    # 속도 계산 (m/s)
                    vx = dx / dt
                    vy = dy / dt
                    
                    # 속도가 너무 크면 잘못된 검출로 간주하고 0으로 설정
                    velocity_magnitude = np.sqrt(vx**2 + vy**2)
                    if velocity_magnitude > 20.0:  # 20 m/s 이상은 비현실적
                        vx, vy = 0.0, 0.0
        
        # 현재 박스 위치와 타임스탬프를 다음 계산을 위해 저장
        if detected_box is not None:
            self.previous_box_position = {
                'x': detected_box['x'],
                'y': detected_box['y']
            }
        self.previous_timestamp = timestamp
        
        return vx, vy
    
    def detect_static_box(self, laser_points, ranges, intensities, angles):
        """
        ABD(Adaptive Breakpoint Detection) 알고리즘을 사용하여 정적 박스를 검출
        
        Args:
            laser_points: 2D numpy array [x, y] 좌표
            ranges: 거리 값들
            intensities: 강도 값들
            angles: 각도 값들
        
        Returns:
            dict: 검출된 박스 정보 또는 None
        """
        if len(ranges) < self.min_obs_size:
            return None
        
        # 포인트 클라우드를 리스트로 변환
        cloud_points = []
        for i in range(laser_points.shape[1]):
            cloud_points.append((laser_points[0, i], laser_points[1, i]))
        
        # ABD 알고리즘으로 객체 분할
        objects_pointcloud_list = self.segment_objects(cloud_points, ranges, angles)
        
        if not objects_pointcloud_list:
            return None
        
        # 가장 큰 객체를 정적 박스로 간주
        largest_object = max(objects_pointcloud_list, key=len)
        
        if len(largest_object) < self.min_obs_size:
            return None
        
        # 박스의 중심점과 방향 계산
        box_info = self.calculate_box_properties(largest_object)
        
        return box_info
    
    def segment_objects(self, cloud_points, ranges, angles):
        """
        ABD 알고리즘을 사용하여 포인트 클라우드를 객체별로 분할
        """
        if len(cloud_points) == 0:
            return []
        
        objects_pointcloud_list = [[cloud_points[0]]]
        
        for idx in range(1, len(cloud_points)):
            point = cloud_points[idx]
            prev_point = cloud_points[idx-1]
            
            # 거리 기반 분할 임계값 계산
            dist = math.sqrt(point[0]**2 + point[1]**2)
            d_phi = self.angle_increment
            l = self.lambda_angle * math.pi / 180  # degree to radian
            d_max = (dist * math.sin(d_phi) / math.sin(l - d_phi) + 3 * self.sigma) / 2
            
            # 이전 점과의 거리 계산
            point_dist = math.sqrt((point[0] - prev_point[0])**2 + (point[1] - prev_point[1])**2)
            
            if point_dist > d_max:
                # 새로운 객체 시작
                objects_pointcloud_list.append([point])
            else:
                # 기존 객체에 추가
                objects_pointcloud_list[-1].append(point)
        
        # 너무 작은 객체들 제거
        filtered_objects = []
        for obj in objects_pointcloud_list:
            if len(obj) >= self.min_obs_size and len(obj) <= self.max_obs_size:
                filtered_objects.append(obj)
        
        return filtered_objects
    
    def calculate_box_properties(self, object_points):
        """
        객체 포인트들로부터 박스의 중심점과 방향을 계산
        
        Args:
            object_points: 객체를 구성하는 포인트들의 리스트
        
        Returns:
            dict: 박스의 속성 (x, y, yaw)
        """
        points_array = np.array(object_points)
        
        # 중심점 계산 (단순 평균)
        center_x = np.mean(points_array[:, 0])
        center_y = np.mean(points_array[:, 1])
        
        # PCA를 사용하여 주축 방향 계산
        centered_points = points_array - np.array([center_x, center_y])
        cov_matrix = np.cov(centered_points.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        
        # 가장 큰 고유값에 해당하는 고유벡터가 주축 방향
        main_axis_idx = np.argmax(eigenvalues)
        main_axis = eigenvectors[:, main_axis_idx]
        
        # 방향각 계산 (라디안)
        yaw = math.atan2(main_axis[1], main_axis[0])
        
        return {
            'x': center_x,
            'y': center_y,
            'yaw': yaw,
            'size': len(object_points)
        }
    
    def save_to_csv(self):
        """
        검출된 데이터를 CSV 파일로 저장
        """
        if not self.scan_data:
            self.get_logger().error('저장할 데이터가 없습니다!')
            return False
        
        # 출력 디렉토리 생성
        output_dir = os.path.dirname(self.output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        headers = ['lidar', 'intensities', 'x', 'y', 'vx', 'vy', 'yaw']
        
        with open(self.output_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            
            for entry in self.scan_data:
                writer.writerow(entry)
        
        self.get_logger().info(f'CSV 파일 저장 완료: {self.output_path}')
        self.get_logger().info(f'총 {len(self.scan_data)}개의 엔트리가 저장되었습니다.')
        self.get_logger().info(f'박스가 검출된 스캔: {len(self.detected_boxes)}개')
        
        return True
    
    def manual_verify_data(self, verify_all=False):
        """
        수동 검증 모드: 저장된 데이터를 시각화하고 사용자가 수정/삭제할 수 있게 함
        
        Args:
            verify_all: True면 모든 데이터 검증, False면 의심스러운 데이터만 검증
        """
        if not self.scan_data:
            print('검증할 데이터가 없습니다!')
            return False
        
        if verify_all:
            # 모든 데이터 검증
            target_indices = list(range(len(self.scan_data)))
            print(f'=== 전체 검증 모드 ===')
            print(f'총 {len(self.scan_data)}개의 모든 데이터를 검증합니다.')
        else:
            # 의심스러운 데이터만 검증
            target_indices = self._find_suspicious_data()
            print(f'=== 스마트 검증 모드 ===')
            print(f'총 {len(self.scan_data)}개 중 {len(target_indices)}개의 의심스러운 데이터를 검증합니다.')
            if len(target_indices) == 0:
                print('의심스러운 데이터가 없습니다. 검증을 건너뜁니다.')
                return True
        
        print('사용법:')
        print('  → (오른쪽 화살표): 다음 데이터')
        print('  ← (왼쪽 화살표): 이전 데이터')
        print('  e: 수정 모드 (마우스로 위치 클릭)')
        print('  d: 현재 데이터 삭제')
        print('  s: 검증 완료 후 저장')
        print('  q: 검증 종료 (저장 안함)')
        print('  창을 닫으면 다음 데이터로 이동')
        print('-' * 50)
        
        # matplotlib 설정
        plt.ion()  # 인터랙티브 모드 활성화
        
        # 검증 결과 저장 (원본 인덱스 기준)
        verification_results = {}  # {original_index: 'keep'/'delete'/modified_entry}
        current_target_idx = 0
        
        # 키보드 이벤트를 위한 변수
        self.current_action = None
        self.clicked_position = None
        
        try:
            while current_target_idx < len(target_indices):
                original_idx = target_indices[current_target_idx]
                entry = self.scan_data[original_idx]
                
                print(f'\n검증 진행률: {current_target_idx + 1}/{len(target_indices)} (전체 데이터 {original_idx + 1}번)')
                
                # 데이터 시각화
                fig = self._visualize_entry(entry, original_idx)
                
                # 키보드 이벤트 핸들러 연결
                def on_key(event):
                    if event.key == 'right':
                        self.current_action = 'next'
                        plt.close(fig)
                    elif event.key == 'left':
                        self.current_action = 'prev'
                        plt.close(fig)
                    elif event.key == 'd':
                        self.current_action = 'delete'
                        plt.close(fig)
                    elif event.key == 'e':
                        self.current_action = 'edit'
                        plt.close(fig)
                    elif event.key == 's':
                        self.current_action = 'save'
                        plt.close(fig)
                    elif event.key == 'q':
                        self.current_action = 'quit'
                        plt.close(fig)
                
                fig.canvas.mpl_connect('key_press_event', on_key)
                
                # 그래프 표시 및 대기
                plt.show()
                
                # 이벤트가 발생할 때까지 대기
                while plt.get_fignums() and self.current_action is None:
                    plt.pause(0.1)
                
                # 액션 처리
                if self.current_action == 'quit':
                    print('검증을 종료합니다.')
                    return False
                elif self.current_action == 'save':
                    print('검증을 완료하고 저장합니다.')
                    break
                elif self.current_action == 'delete':
                    print(f'데이터 {original_idx+1} 삭제됨')
                    verification_results[original_idx] = 'delete'
                    current_target_idx += 1
                elif self.current_action == 'edit':
                    # 데이터 수정 - 마우스 클릭 모드
                    modified_entry = self._edit_entry_with_mouse(entry, original_idx)
                    if modified_entry:
                        verification_results[original_idx] = modified_entry
                        print('데이터가 수정되었습니다.')
                    else:
                        verification_results[original_idx] = 'keep'
                        print('수정이 취소되었습니다.')
                    current_target_idx += 1
                elif self.current_action == 'prev':
                    # 이전 프레임으로
                    if current_target_idx > 0:
                        current_target_idx -= 1
                        # 이전 결과 제거
                        prev_idx = target_indices[current_target_idx]
                        if prev_idx in verification_results:
                            del verification_results[prev_idx]
                    continue
                elif self.current_action == 'next' or self.current_action is None:
                    # 다음 프레임으로 (기본 동작 - 유지)
                    verification_results[original_idx] = 'keep'
                    current_target_idx += 1
                
                # 액션 초기화
                self.current_action = None
        
        finally:
            plt.ioff()  # 인터랙티브 모드 비활성화
            plt.close('all')  # 모든 그래프 닫기
        
        # 검증 결과 적용
        self._apply_verification_results(verification_results)
        
        return True
    
    def _find_suspicious_data(self):
        """
        의심스러운 데이터 인덱스 찾기 (이전 위치로부터 0.6m 이상 떨어진 경우)
        
        Returns:
            list: 의심스러운 데이터의 인덱스 리스트
        """
        suspicious_indices = []
        distance_threshold = 0.6  # 0.6m
        
        for i in range(1, len(self.scan_data)):
            prev_entry = self.scan_data[i-1]
            curr_entry = self.scan_data[i]
            
            # 이전 데이터나 현재 데이터가 (0,0)이면 건너뛰기
            if (prev_entry["x"] == 0.0 and prev_entry["y"] == 0.0) or \
               (curr_entry["x"] == 0.0 and curr_entry["y"] == 0.0):
                continue
            
            # 거리 계산
            dx = curr_entry["x"] - prev_entry["x"]
            dy = curr_entry["y"] - prev_entry["y"]
            distance = math.sqrt(dx**2 + dy**2)
            
            if distance > distance_threshold:
                suspicious_indices.append(i)
                print(f'의심스러운 데이터 발견: 인덱스 {i+1}, 이전 위치로부터 {distance:.2f}m 떨어짐')
        
        return suspicious_indices
    
    def _apply_verification_results(self, verification_results):
        """
        검증 결과를 원본 데이터에 적용
        
        Args:
            verification_results: {index: 'keep'/'delete'/modified_entry} 형태의 딕셔너리
        """
        new_scan_data = []
        deleted_count = 0
        modified_count = 0
        
        for i, entry in enumerate(self.scan_data):
            if i in verification_results:
                result = verification_results[i]
                if result == 'delete':
                    deleted_count += 1
                    continue  # 삭제된 데이터는 추가하지 않음
                elif result == 'keep':
                    new_scan_data.append(entry)
                else:
                    # 수정된 데이터
                    new_scan_data.append(result)
                    modified_count += 1
            else:
                # 검증하지 않은 데이터는 그대로 유지
                new_scan_data.append(entry)
        
        self.scan_data = new_scan_data
        print(f'검증 완료: {len(new_scan_data)}개 데이터 유지, {modified_count}개 수정, {deleted_count}개 삭제')
    
    def _visualize_entry(self, entry, index, edit_mode=False):
        """
        개별 데이터 엔트리를 시각화
        """
        print(f'\n--- 데이터 {index+1} ---')
        print(f'위치: x={entry["x"]:.3f}, y={entry["y"]:.3f}')
        print(f'속도: vx={entry["vx"]:.3f}, vy={entry["vy"]:.3f}')
        print(f'방향: yaw={entry["yaw"]:.3f} rad ({math.degrees(entry["yaw"]):.1f}°)')
        print(f'LiDAR 포인트 수: {len(entry["lidar"])}개')
        
        # matplotlib을 사용한 시각화
        fig = plt.figure(figsize=(10, 8))
        
        # LiDAR 포인트 플롯 - 좌표 변환 (+X 위쪽, +Y 왼쪽)
        if len(entry["lidar"]) > 0:
            ranges = np.array(entry["lidar"])
            # 각도 계산 (기본 LiDAR 파라미터 사용)
            angles = np.linspace(-2.356194496154785, 2.356194496154785, len(ranges))
            
            # 극좌표를 직교좌표로 변환
            x_points = ranges * np.cos(angles)
            y_points = ranges * np.sin(angles)
            
            # 좌표 변환: (x,y) -> (-y, x) - 시계 반대방향 90도 회전
            x_rot = -y_points
            y_rot = x_points
            
            # LiDAR 포인트들 플롯
            plt.scatter(x_rot, y_rot, c='lightblue', s=10, alpha=0.6, label='LiDAR Points')
        
        # LiDAR 센서 위치 표시 (원점)
        plt.scatter(0, 0, c='black', s=100, marker='s', label='LiDAR Sensor', zorder=5)
        
        # 좌표축 그리기
        ax = plt.gca()
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)  # x축
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)  # y축
        
        # +X, +Y 방향 표시 (회전된 좌표계)
        ax.annotate('+X', xy=(0.5, max(5, abs(entry["x"])+2)), fontsize=12, color='red', weight='bold')
        ax.annotate('+Y', xy=(-max(5, abs(entry["y"])+2), 0.5), fontsize=12, color='red', weight='bold')
        
        # 객체 위치 표시 - 좌표 변환
        if entry["x"] != 0.0 or entry["y"] != 0.0:
            # 좌표 변환: (x,y) -> (-y, x)
            obj_x_rot = -entry["y"]
            obj_y_rot = entry["x"]
            
            # 객체 중심점
            plt.scatter(obj_x_rot, obj_y_rot, c='red', s=100, marker='o', label='Object Center')
            
            # 속도 벡터 (vx, vy) - 크기 1/3로 축소하고 좌표 변환
            if entry["vx"] != 0.0 or entry["vy"] != 0.0:
                vx_rot = -entry["vy"] / 3
                vy_rot = entry["vx"] / 3
                plt.arrow(obj_x_rot, obj_y_rot, vx_rot, vy_rot, 
                         head_width=0.07, head_length=0.1, fc='blue', ec='blue', 
                         label='Velocity Vector', width=0.02)
            
            # yaw 방향 벡터 - 크기 1/3로 축소하고 좌표 변환
            if entry["yaw"] != 0.0:
                yaw_length = 0.67  # 2.0 / 3 = 0.67
                yaw_x = yaw_length * math.cos(entry["yaw"])
                yaw_y = yaw_length * math.sin(entry["yaw"])
                # 좌표 변환
                yaw_x_rot = -yaw_y
                yaw_y_rot = yaw_x
                plt.arrow(obj_x_rot, obj_y_rot, yaw_x_rot, yaw_y_rot, 
                         head_width=0.05, head_length=0.07, fc='green', ec='green', 
                         label='Yaw Direction', width=0.015)
            
            # 텍스트 정보 표시 - 위치도 변환
            info_text = f'x: {entry["x"]:.2f}\ny: {entry["y"]:.2f}\nvx: {entry["vx"]:.2f}\nvy: {entry["vy"]:.2f}\nyaw: {entry["yaw"]:.2f} rad\n({math.degrees(entry["yaw"]):.1f}°)'
            plt.text(obj_x_rot + 1, obj_y_rot + 1, info_text, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8),
                    fontsize=10)
        
        # 그리드와 축 설정
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        # 축 범위 설정 (회전된 좌표계 기준)
        if entry["x"] != 0.0 or entry["y"] != 0.0:
            margin = 5
            # 원래 x,y를 회전된 좌표계로 변환
            obj_x_rot = -entry["y"]
            obj_y_rot = entry["x"]
            x_min = min(-margin, obj_x_rot - margin)
            x_max = max(margin, obj_x_rot + margin)
            y_min = min(-margin, obj_y_rot - margin)
            y_max = max(margin, obj_y_rot + margin)
        else:
            x_min, x_max = -10, 10
            y_min, y_max = -10, 10
        
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        
        # 제목 설정 (수정 모드에 따라 다르게)
        if edit_mode:
            title = f'EDIT MODE - Data Entry {index+1}\nClick to Set New Position | Enter: Confirm | ESC/q: Cancel'
        else:
            title = f'Data Entry {index+1} - Static Box Detection\nLeft Click: Move Object | ← → : Navigate | d: Delete | e: Edit | s: Save | q: Quit'
        
        # 레이블과 제목
        plt.xlabel('Y (m)', fontsize=12)  # 회전된 좌표계에서 가로축은 Y
        plt.ylabel('X (m)', fontsize=12)  # 회전된 좌표계에서 세로축은 X
        plt.title(title, fontsize=12)
        plt.legend()
        
        # 그래프 표시
        plt.tight_layout()
        
        return fig
    
    def _edit_entry_with_mouse(self, entry, index):
        """
        마우스 클릭으로 데이터 엔트리 수정
        """
        print('\n=== 수정 모드 ===')
        print('원하는 위치를 마우스로 클릭하세요.')
        print('또는 다음 키를 사용하세요:')
        print('  Enter: 현재 값 유지하고 완료')
        print('  ESC 또는 q: 수정 취소')
        print('-' * 30)
        
        # 수정용 변수 초기화
        self.edit_clicked_position = None
        self.edit_action = None
        
        # 시각화 창 생성
        fig = self._visualize_entry(entry, index, edit_mode=True)
        
        # 수정 모드용 이벤트 핸들러
        def on_edit_click(event):
            if event.inaxes is not None and event.button == 1:  # 왼쪽 마우스 버튼
                # 회전된 좌표계에서 원래 좌표계로 변환
                clicked_x_rot = event.xdata
                clicked_y_rot = event.ydata
                
                # 좌표 역변환: (x_rot, y_rot) -> (y, -x_rot)
                actual_x = clicked_y_rot
                actual_y = -clicked_x_rot
                
                self.edit_clicked_position = (actual_x, actual_y)
                self.edit_action = 'position_updated'
                print(f'새 위치 선택: x={actual_x:.3f}, y={actual_y:.3f}')
                
                # 새 위치를 시각적으로 표시
                ax = plt.gca()
                # 기존 "New Position" 마커 제거
                for artist in ax.get_children():
                    if hasattr(artist, 'get_label') and artist.get_label() == 'New Position':
                        artist.remove()
                
                # 새 위치 표시 (회전된 좌표계로)
                new_x_rot = -actual_y
                new_y_rot = actual_x
                ax.scatter(new_x_rot, new_y_rot, c='magenta', s=150, marker='x', 
                          linewidths=3, label='New Position', zorder=10)
                ax.text(new_x_rot + 0.5, new_y_rot + 0.5, 'NEW', 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="magenta", alpha=0.8),
                       fontsize=12, color='white', weight='bold')
                ax.legend()
                plt.draw()
        
        def on_edit_key(event):
            if event.key == 'enter':
                self.edit_action = 'confirm'
                plt.close(fig)
            elif event.key == 'escape' or event.key == 'q':
                self.edit_action = 'cancel'
                plt.close(fig)
        
        # 이벤트 연결
        fig.canvas.mpl_connect('button_press_event', on_edit_click)
        fig.canvas.mpl_connect('key_press_event', on_edit_key)
        
        # 창 표시 및 대기
        plt.show()
        
        # 이벤트 대기
        while plt.get_fignums() and self.edit_action is None:
            plt.pause(0.1)
        
        # 결과 처리
        if self.edit_action == 'confirm' and self.edit_clicked_position:
            # 새 위치로 수정
            modified_entry = entry.copy()
            modified_entry["x"] = self.edit_clicked_position[0]
            modified_entry["y"] = self.edit_clicked_position[1]
            print(f'위치 수정 완료: x={self.edit_clicked_position[0]:.3f}, y={self.edit_clicked_position[1]:.3f}')
            return modified_entry
        elif self.edit_action == 'confirm':
            # 위치 변경 없이 유지
            print('위치 변경 없이 유지됩니다.')
            return entry
        else:
            # 수정 취소
            print('수정이 취소되었습니다.')
            return None
    
    def _edit_entry(self, entry):
        """
        데이터 엔트리 수정
        """
        print('\n현재 값:')
        print(f'x: {entry["x"]:.3f}')
        print(f'y: {entry["y"]:.3f}')
        print(f'vx: {entry["vx"]:.3f}')
        print(f'vy: {entry["vy"]:.3f}')
        print(f'yaw: {entry["yaw"]:.3f}')
        
        try:
            new_x = input(f'새 x 값 (현재: {entry["x"]:.3f}, Enter로 유지): ')
            new_y = input(f'새 y 값 (현재: {entry["y"]:.3f}, Enter로 유지): ')
            new_vx = input(f'새 vx 값 (현재: {entry["vx"]:.3f}, Enter로 유지): ')
            new_vy = input(f'새 vy 값 (현재: {entry["vy"]:.3f}, Enter로 유지): ')
            new_yaw = input(f'새 yaw 값 (현재: {entry["yaw"]:.3f}, Enter로 유지): ')
            
            modified_entry = entry.copy()
            
            if new_x.strip():
                modified_entry["x"] = float(new_x)
            if new_y.strip():
                modified_entry["y"] = float(new_y)
            if new_vx.strip():
                modified_entry["vx"] = float(new_vx)
            if new_vy.strip():
                modified_entry["vy"] = float(new_vy)
            if new_yaw.strip():
                modified_entry["yaw"] = float(new_yaw)
            
            return modified_entry
            
        except ValueError:
            print('잘못된 값입니다. 수정이 취소됩니다.')
            return None
    
    def filter_quality_data(self):
        """
        데이터 품질 관리: 불량 데이터 자동 폐기
        """
        if not self.scan_data:
            return
        
        original_count = len(self.scan_data)
        filtered_data = []
        
        for entry in self.scan_data:
            # 품질 검사 기준
            is_valid = True
            
            # 1. 위치가 합리적인 범위 내에 있는지 확인
            if abs(entry["x"]) > 5 or abs(entry["y"]) > 5:  # 5m 이상은 비현실적
                is_valid = False
            
            # 2. 속도가 합리적인 범위 내에 있는지 확인
            velocity_magnitude = math.sqrt(entry["vx"]**2 + entry["vy"]**2)
            if velocity_magnitude > 5.56:  # 시속20 -> 5.56m/s
                is_valid = False
            
            # 3. LiDAR 데이터가 충분한지 확인
            if len(entry["lidar"]) < 5:  # 너무 적은 포인트
                is_valid = False
            
            # 4. yaw 값이 유효한지 확인
            if not (-math.pi <= entry["yaw"] <= math.pi):
                is_valid = False
            
            if is_valid:
                filtered_data.append(entry)
        
        self.scan_data = filtered_data
        removed_count = original_count - len(filtered_data)
        
        print(f'데이터 품질 필터링 완료:')
        print(f'  원본: {original_count}개')
        print(f'  유지: {len(filtered_data)}개')
        print(f'  제거: {removed_count}개')
        
        return True

def main():
    parser = argparse.ArgumentParser(description='ROS2 bag 파일에서 정적 박스 데이터를 추출하여 CSV로 저장')
    parser.add_argument('--bag', type=str, required=True, help='ROS2 bag 파일 경로 (db3)')
    parser.add_argument('--output', type=str, help='출력 CSV 파일 경로')
    parser.add_argument('--manual-verify', action='store_true', help='스마트 검증 모드 (의심스러운 데이터만 확인)')
    parser.add_argument('--manual-verify-all', action='store_true', help='전체 검증 모드 (모든 데이터 확인)')
    
    args = parser.parse_args()
    
    # 출력 경로 설정
    if args.output is None:
        bag_name = os.path.basename(args.bag).split('.')[0]
        args.output = f'/home/harry/ros2_ws/src/redbull/{bag_name}_static_boxes.csv'
    
    # bag 파일 존재 확인
    if not os.path.exists(args.bag):
        print(f'Error: bag 파일을 찾을 수 없습니다: {args.bag}')
        return
    
    # ROS2 초기화
    rclpy.init()
    
    try:
        # 검출기 생성 및 실행
        detector = StaticBoxDetector(args.bag, args.output)
        
        print(f'ROS2 bag 파일 읽기 시작: {args.bag}')
        if detector.read_bag():
            print('데이터 처리 완료')
            
            # 데이터 품질 필터링
            detector.filter_quality_data()
            
            # 수동 검증 모드
            if args.manual_verify_all:
                print('\n=== 전체 수동 검증 모드 ===')
                if not detector.manual_verify_data(verify_all=True):
                    print('검증이 취소되었습니다.')
                    return
            elif args.manual_verify:
                print('\n=== 스마트 수동 검증 모드 ===')
                if not detector.manual_verify_data(verify_all=False):
                    print('검증이 취소되었습니다.')
                    return
            
            print('CSV 파일 저장 중...')
            if detector.save_to_csv():
                print(f'성공적으로 완료되었습니다: {args.output}')
            else:
                print('CSV 파일 저장에 실패했습니다.')
        else:
            print('bag 파일 읽기에 실패했습니다.')
    
    except Exception as e:
        print(f'오류 발생: {e}')
    
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()

