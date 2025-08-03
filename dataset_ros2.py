#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.widgets import Button
import os
import argparse
import csv
from sensor_msgs.msg import LaserScan
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import math

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# 한글 폰트 찾기 및 설정
def setup_korean_font():
    """한글 폰트를 설정합니다."""
    try:
        # 시스템에서 사용 가능한 한글 폰트 찾기
        font_candidates = [
            'NanumGothic', 'Noto Sans CJK KR', 'Malgun Gothic', 
            'AppleGothic', 'Gulim', 'Dotum', 'NanumBarunGothic'
        ]
        
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        
        korean_font = None
        for font in font_candidates:
            if font in available_fonts:
                korean_font = font
                break
        
        if korean_font:
            plt.rcParams['font.family'] = korean_font
            print(f"한글 폰트 설정 완료: {korean_font}")
        else:
            # 한글 폰트가 없으면 영어로 표시
            plt.rcParams['font.family'] = 'DejaVu Sans'
            print("한글 폰트를 찾을 수 없어 영어로 표시됩니다.")
            
    except Exception as e:
        print(f"폰트 설정 중 오류: {e}")
        plt.rcParams['font.family'] = 'DejaVu Sans'

# 폰트 설정 실행
setup_korean_font()


class StaticBoxDetector(Node):
    """
    ROS2 클래스로 db3 파일에서 /scan 토픽을 읽어 정적 박스 데이터를 추출하고 CSV로 저장
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
        
        # 포인트 클라우드 생성
        laser_points = np.vstack([x_points, y_points])
        
        # 정적 박스 검출
        detected_box = self.detect_static_box(laser_points, valid_ranges, valid_intensities, valid_angles)
        
        # 속도 계산
        vx, vy = self.calculate_velocity(detected_box, timestamp)
        
        if detected_box is not None:
            # CSV 형식으로 데이터 저장: [lidar_ranges], [intensities], x, y, vx, vy, yaw
            csv_entry = {
                'lidar': valid_ranges.tolist(),
                'intensities': [0.5] * len(valid_ranges),  # 모든 intensities를 0.5로 설정
                'x': detected_box['x'],
                'y': detected_box['y'],
                'vx': vx,
                'vy': vy,
                'yaw': detected_box['yaw']
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
    
    def visualize_sample(self, sample_index=0):
        """
        샘플 데이터를 시각화
        
        Args:
            sample_index: 시각화할 샘플의 인덱스
        """
        if sample_index >= len(self.scan_data):
            self.get_logger().error(f'인덱스 {sample_index}는 유효하지 않습니다. 최대: {len(self.scan_data)-1}')
            return
        
        entry = self.scan_data[sample_index]
        ranges = np.array(entry['lidar'])
        
        # 각도 재계산
        angles = np.linspace(self.angle_min, self.angle_max, len(ranges))
        
        # 직교좌표로 변환
        x_points = ranges * np.cos(angles)
        y_points = ranges * np.sin(angles)
        
        # 플롯
        plt.figure(figsize=(12, 10))
        plt.scatter(x_points, y_points, s=1, alpha=0.6, label='LiDAR Scan')
        plt.scatter(0, 0, color='blue', s=100, label='LiDAR Sensor', marker='^')
        
        # 검출된 박스 위치 표시
        if entry['x'] != 0.0 or entry['y'] != 0.0:
            plt.scatter(entry['x'], entry['y'], color='red', s=100, label='Detected Box', marker='s')
            plt.text(entry['x'], entry['y'], f"({entry['x']:.2f}, {entry['y']:.2f})", 
                    fontsize=10, ha='left', va='bottom')
            
            # 속도 벡터 표시 (화살표)
            if abs(entry['vx']) > 0.01 or abs(entry['vy']) > 0.01:
                plt.arrow(entry['x'], entry['y'], entry['vx']*0.5, entry['vy']*0.5, 
                         head_width=0.1, head_length=0.1, fc='orange', ec='orange', 
                         label=f"Velocity: ({entry['vx']:.2f}, {entry['vy']:.2f}) m/s")
        
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.title(f'Static Box Detection Result - Sample {sample_index}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        # 축 범위 설정 (검출된 박스 중심으로)
        if entry['x'] != 0.0 or entry['y'] != 0.0:
            margin = 3.0
            plt.xlim(entry['x'] - margin, entry['x'] + margin)
            plt.ylim(entry['y'] - margin, entry['y'] + margin)
        else:
            plt.xlim(-5, 5)
            plt.ylim(-5, 5)
        
        plt.tight_layout()
        plt.show()
        
    def visualize_sample_interactive(self, sample_index=0):
        """
        인터랙티브용 샘플 데이터 시각화 (non-blocking)
        
        Args:
            sample_index: 시각화할 샘플의 인덱스
        """
        if sample_index >= len(self.scan_data):
            self.get_logger().error(f'인덱스 {sample_index}는 유효하지 않습니다. 최대: {len(self.scan_data)-1}')
            return
        
        entry = self.scan_data[sample_index]
        ranges = np.array(entry['lidar'])
        
        # 각도 재계산
        angles = np.linspace(self.angle_min, self.angle_max, len(ranges))
        
        # 직교좌표로 변환
        x_points = ranges * np.cos(angles)
        y_points = ranges * np.sin(angles)
        
        # 기존 모든 창 닫기
        plt.close('all')
        
        # 새로운 플롯 생성
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(x_points, y_points, s=1, alpha=0.6, label='LiDAR Scan')
        ax.scatter(0, 0, color='blue', s=100, label='LiDAR Sensor', marker='^')
        
        # 검출된 박스 위치 표시
        if entry['x'] != 0.0 or entry['y'] != 0.0:
            ax.scatter(entry['x'], entry['y'], color='red', s=100, label='Detected Box', marker='s')
            ax.text(entry['x'], entry['y'], f"({entry['x']:.2f}, {entry['y']:.2f})", 
                    fontsize=10, ha='left', va='bottom')
            
            # 속도 벡터 표시 (화살표)
            if abs(entry['vx']) > 0.01 or abs(entry['vy']) > 0.01:
                ax.arrow(entry['x'], entry['y'], entry['vx']*0.5, entry['vy']*0.5, 
                         head_width=0.1, head_length=0.1, fc='orange', ec='orange', 
                         label=f"Velocity: ({entry['vx']:.2f}, {entry['vy']:.2f}) m/s")
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f'Frame {sample_index}/{len(self.scan_data)-1} - Press any key in terminal to continue')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # 축 범위 설정 (검출된 박스 중심으로)
        if entry['x'] != 0.0 or entry['y'] != 0.0:
            margin = 3.0
            ax.set_xlim(entry['x'] - margin, entry['x'] + margin)
            ax.set_ylim(entry['y'] - margin, entry['y'] + margin)
        else:
            ax.set_xlim(-5, 5)
            ax.set_ylim(-5, 5)
        
        plt.tight_layout()
        # non-blocking으로 표시
        plt.show(block=False)
        plt.pause(0.1)  # 화면 업데이트를 위한 짧은 pause
    
    def visualize_interactive(self, start_index=0):
        """
        인터랙티브 시각화 - Enter로 다음 프레임, Ctrl+C로 종료
        
        Args:
            start_index: 시작할 샘플 인덱스
        """
        print("\n=== 인터랙티브 시각화 모드 ===")
        print("사용법:")
        print("- Enter 키: 다음 프레임으로 이동")
        print("- 숫자 입력 + Enter: 특정 프레임으로 이동")
        print("- 'q' + Enter: 종료")
        print("- Ctrl + C: 강제 종료")
        print(f"총 {len(self.scan_data)}개의 프레임이 있습니다.")
        print("="*40)
        print("\n⚠️  주의: matplotlib 창은 참고용입니다. 터미널에서 키를 입력하세요!")
        print("="*40)
        
        current_index = start_index
        
        try:
            while current_index < len(self.scan_data):
                print(f"\n현재 프레임: {current_index}/{len(self.scan_data)-1}")
                
                # 현재 프레임 시각화 (non-blocking)
                self.visualize_sample_interactive(current_index)
                
                # 사용자 입력 대기
                try:
                    user_input = input(">>> 다음 프레임으로 이동하려면 Enter를 누르세요 (숫자/q): ").strip()
                    
                    if user_input.lower() == 'q':
                        print("시각화를 종료합니다.")
                        break
                    elif user_input == '':
                        # Enter만 눌렀을 경우 다음 프레임으로
                        current_index += 1
                    elif user_input.isdigit():
                        # 숫자를 입력한 경우 해당 프레임으로 이동
                        target_index = int(user_input)
                        if 0 <= target_index < len(self.scan_data):
                            current_index = target_index
                        else:
                            print(f"❌ 잘못된 프레임 번호입니다. 0-{len(self.scan_data)-1} 사이의 숫자를 입력하세요.")
                            continue
                    else:
                        print("❌ 잘못된 입력입니다. Enter, 숫자, 또는 'q'를 입력하세요.")
                        continue
                        
                except EOFError:
                    # Ctrl+D가 입력된 경우
                    print("\n시각화를 종료합니다.")
                    break
                
            # 마지막에 도달한 경우
            if current_index >= len(self.scan_data):
                print(f"\n🎉 모든 프레임을 확인했습니다! (총 {len(self.scan_data)}개)")
                
        except KeyboardInterrupt:
            print("\n\n⚠️  Ctrl+C가 감지되었습니다. 시각화를 종료합니다.")
        finally:
            plt.close('all')
        
        print("시각화 완료!")


def main():
    parser = argparse.ArgumentParser(description='ROS2 bag 파일에서 정적 박스 데이터를 추출하여 CSV로 저장')
    parser.add_argument('--bag', type=str, required=True, help='ROS2 bag 파일 경로 (db3)')
    parser.add_argument('--output', type=str, help='출력 CSV 파일 경로')
    parser.add_argument('--visualize', nargs='?', const='interactive', default=None,
                       help='시각화 옵션: 숫자(특정 프레임), 값 없음(인터랙티브 모드)')
    
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
            print('CSV 파일 저장 중...')
            if detector.save_to_csv():
                print(f'성공적으로 완료되었습니다: {args.output}')
                
                # 시각화 옵션 처리
                if args.visualize is not None:
                    if args.visualize == 'interactive':
                        # --visualize만 사용된 경우 (인터랙티브 모드)
                        detector.visualize_interactive(start_index=0)
                    else:
                        try:
                            # 숫자가 입력된 경우
                            frame_index = int(args.visualize)
                            detector.visualize_sample(frame_index)
                        except ValueError:
                            print(f"잘못된 프레임 번호입니다: {args.visualize}")
                            print("숫자를 입력하거나 --visualize만 사용하세요.")
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
