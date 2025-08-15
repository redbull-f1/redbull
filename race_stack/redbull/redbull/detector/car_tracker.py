#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
import math
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist

"""
칼만 필터를 사용하여 차량의 위치와 속도를 추적하는 코드입니다.
"""

class KalmanFilter:
    def __init__(self, initial_x, initial_y):
        """칼만 필터 초기화: [x, y, vx, vy]"""
        self.state = np.array([initial_x, initial_y, 0.0, 0.0])
        
        # 상태 전이 행렬 (등속 모델)
        dt = 0.1  # 가정된 시간 간격
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # 관측 행렬 (위치만 관측)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # 공분산 행렬
        self.P = np.eye(4) * 100  # 초기 불확실성
        
        # 프로세스 노이즈
        self.Q = np.array([
            [0.1, 0, 0, 0],
            [0, 0.1, 0, 0],
            [0, 0, 0.5, 0],
            [0, 0, 0, 0.5]
        ])
        
        # 관측 노이즈
        self.R = np.eye(2) * 0.1
        
    def predict(self):
        """예측 단계"""
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        
    def update(self, measurement):
        """업데이트 단계"""
        measurement = np.array(measurement)
        
        # 혁신 (innovation)
        y = measurement - self.H @ self.state
        
        # 혁신 공분산
        S = self.H @ self.P @ self.H.T + self.R
        
        # 칼만 게인
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # 상태 업데이트
        self.state = self.state + K @ y
        
        # 공분산 업데이트
        I = np.eye(4)
        self.P = (I - K @ self.H) @ self.P
        
    def get_position(self):
        """현재 위치 반환"""
        return self.state[0], self.state[1]
        
    def get_velocity(self):
        """현재 속도 반환"""
        return self.state[2], self.state[3]

class CarTracker:
    def __init__(self, initial_x, initial_y, tracker_id):
        """개별 차량 추적기"""
        self.id = tracker_id
        self.kalman = KalmanFilter(initial_x, initial_y)
        self.age = 0  # 추적기 나이
        self.hits = 1  # 검출 횟수
        self.time_since_update = 0  # 마지막 업데이트로부터 시간
        
    def predict(self):
        """예측 수행"""
        self.kalman.predict()
        self.age += 1
        self.time_since_update += 1
        
    def update(self, detection):
        """검출로 업데이트"""
        self.kalman.update(detection)
        self.hits += 1
        self.time_since_update = 0
        
    def get_predicted_position(self):
        """예측된 위치 반환"""
        return self.kalman.get_position()
        
    def get_velocity(self):
        """속도 반환"""
        return self.kalman.get_velocity()

class LidarProcessor(Node):
    def __init__(self):
        super().__init__('lidar_processor')
        
        # 구독자와 발행자 설정
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            10
        )
        
        self.cluster_marker_pub = self.create_publisher(
            MarkerArray,
            '/cluster_markers',
            10
        )
        
        # 클러스터링 매개변수 (F1 tenth에 최적화)
        self.eps = 0.3  # F1 tenth 크기에 맞게 조정
        self.min_samples = 2  # 최소 포인트 수
        
        # 마커 관리용 변수
        self.previous_marker_count = 0
        
        # 차량 추적 변수
        self.trackers = []
        self.next_id = 0
        self.max_association_distance = 1.0  # 연관 최대 거리
        
        self.get_logger().info('LiDAR 전처리 노드가 시작되었습니다.')

    def lidar_callback(self, msg):
        """LiDAR 데이터 콜백 함수"""
        try:
            # LiDAR 데이터를 직교 좌표로 변환
            points = self.convert_scan_to_points(msg)
            
            if len(points) < 5:
                self.get_logger().debug('포인트가 너무 적습니다.')
                return
            
            # 클러스터링 수행
            clusters = self.perform_clustering(points)
            
            if len(clusters) == 0:
                # 추적만 예측 수행
                self.predict_tracked_cars()
                tracked_markers = [tracker for tracker in self.trackers if tracker.time_since_update < 5]
                self.publish_tracked_markers(tracked_markers, msg.header.stamp)
                return
            
            # 자동차로 분류된 클러스터만 필터링
            car_clusters = self.filter_car_clusters(clusters)
            
            if len(car_clusters) > 0:
                # 차량 추적 수행
                tracked_cars = self.track_cars(car_clusters)
                
                # 추적된 차량 마커 발행
                self.publish_tracked_markers(tracked_cars, msg.header.stamp)
                
                self.get_logger().info(f'추적 중인 차량: {len(tracked_cars)}개')
            else:
                # 검출된 차량이 없을 때도 예측만 수행
                self.predict_tracked_cars()
                tracked_markers = [tracker for tracker in self.trackers if tracker.time_since_update < 5]
                self.publish_tracked_markers(tracked_markers, msg.header.stamp)
                
        except Exception as e:
            self.get_logger().error(f'LiDAR 콜백 오류: {str(e)}')

    def convert_scan_to_points(self, scan_msg):
        """LaserScan 메시지를 2D 포인트 배열로 변환"""
        points = []
        
        for i, range_val in enumerate(scan_msg.ranges):
            # 유효한 거리 값만 처리
            if scan_msg.range_min <= range_val <= scan_msg.range_max:
                angle = scan_msg.angle_min + i * scan_msg.angle_increment
                x = range_val * math.cos(angle)
                y = range_val * math.sin(angle)
                points.append([x, y])
        
        return np.array(points)

    def perform_clustering(self, points):
        """DBSCAN을 사용한 클러스터링"""
        if len(points) < self.min_samples:
            return []
        
        # DBSCAN 클러스터링 수행
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        cluster_labels = dbscan.fit_predict(points)
        
        clusters = []
        unique_labels = set(cluster_labels)
        
        for label in unique_labels:
            if label == -1:  # 노이즈 제외
                continue
            
            # 해당 라벨의 포인트들 추출
            cluster_points = points[cluster_labels == label]
            
            # 클러스터 중심점 계산
            center = np.mean(cluster_points, axis=0)
            
            clusters.append({
                'points': cluster_points,
                'center': center,
                'size': len(cluster_points)
            })
        
        return clusters

    def analyze_shape_features(self, points):
        """클러스터의 형태적 특징 분석 (ㄷ 모양 검출)"""
        if len(points) < 5:
            return False, 0.0
        
        # 센서 원점을 기준으로 각도별로 포인트 정렬
        center = np.mean(points, axis=0)
        
        # 각 포인트의 센서 원점으로부터의 각도 계산
        angles = []
        distances = []
        for point in points:
            angle = math.atan2(point[1], point[0])
            distance = np.linalg.norm(point)
            angles.append(angle)
            distances.append(distance)
        
        # 각도 순으로 정렬
        sorted_indices = np.argsort(angles)
        sorted_points = points[sorted_indices]
        sorted_distances = np.array(distances)[sorted_indices]
        
        # 연속된 포인트 간의 거리 차이 분석
        distance_diffs = []
        for i in range(1, len(sorted_distances)):
            diff = abs(sorted_distances[i] - sorted_distances[i-1])
            distance_diffs.append(diff)
        
        # ㄷ 모양 특징: 가장자리에서 중간으로 갈 때 거리가 급격히 증가
        if len(distance_diffs) < 3:
            return False, 0.0
        
        # 거리 변화의 표준편차 (변화가 클수록 ㄷ 모양일 가능성)
        distance_variation = np.std(distance_diffs)
        
        # 최대 거리 차이
        max_distance_jump = max(distance_diffs) if distance_diffs else 0
        
        # ㄷ 모양 점수 계산
        u_shape_score = distance_variation * 10 + max_distance_jump * 5
        
        # 벽은 보통 일직선이므로 거리 변화가 적음
        is_u_shape = u_shape_score > 0.3 and max_distance_jump > 0.1
        
        return is_u_shape, u_shape_score

    def analyze_linearity(self, points):
        """포인트들이 직선에 가까운지 분석 (벽 검출용)"""
        if len(points) < 3:
            return True, 1.0  # 포인트가 적으면 직선으로 간주
        
        # 첫 번째와 마지막 포인트를 연결한 직선
        start_point = points[0]
        end_point = points[-1]
        
        # 각 포인트와 직선 간의 거리 계산
        line_vec = end_point - start_point
        line_length = np.linalg.norm(line_vec)
        
        if line_length < 0.015:  # 너무 짧은 선분
            return True, 1.0
        
        line_unit = line_vec / line_length
        
        deviations = []
        for point in points[1:-1]:  # 첫 번째와 마지막 포인트 제외
            vec_to_point = point - start_point
            # 직선에 수직인 성분의 크기
            perpendicular = vec_to_point - np.dot(vec_to_point, line_unit) * line_unit
            deviation = np.linalg.norm(perpendicular)
            deviations.append(deviation)
        
        if not deviations:
            return True, 1.0
        
        mean_deviation = np.mean(deviations)
        max_deviation = max(deviations)
        
        # 직선성 점수 (낮을수록 직선에 가까움)
        linearity_score = mean_deviation + max_deviation * 0.5
        
        # 벽으로 판단되는 임계값
        is_linear = linearity_score < 0.05  # 5cm 이내 편차
        
        return is_linear, linearity_score

    def filter_car_clusters(self, clusters):
        """자동차로 분류되는 클러스터만 필터링 (형태 분석 포함) - 가장 좋은 후보 1개만 선택"""
        car_candidates = []
        
        for cluster in clusters:
            points = cluster['points']
            
            # 기본 크기 및 거리 필터링
            min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
            min_y, max_y = np.min(points[:, 1]), np.max(points[:, 1])
            
            width = max_x - min_x
            height = max_y - min_y
            
            # F1 tenth 차량 크기 기준
            min_width, max_width = 0.1, 0.4  # 최소/최대 폭 (m)
            min_height, max_height = 0.1, 0.6  # 최소/최대 길이 (m)
            min_points = 3  # 최소 포인트 수
            max_points = 60  # 최대 포인트 수
            
            # 거리 필터
            distance = np.linalg.norm(cluster['center'])
            max_distance = 10.0  # 최대 10m까지만
            
            # 기본 조건 확인
            size_ok = (min_width <= width <= max_width and 
                      min_height <= height <= max_height)
            point_count_ok = min_points <= cluster['size'] <= max_points
            distance_ok = distance <= max_distance
            
            if not (size_ok and point_count_ok and distance_ok):
                continue
            
            # 형태 분석 수행
            is_u_shape, u_shape_score = self.analyze_shape_features(points)
            is_linear, linearity_score = self.analyze_linearity(points)
            
            # 차량 점수 계산 (높을수록 차량일 가능성 높음)
            car_score = 0.0
            
            if is_u_shape:
                car_score += u_shape_score * 2.0  # ㄷ 모양 가중치
            
            if not is_linear:
                car_score += 1.0  # 직선이 아닌 경우 보너스
            else:
                car_score -= 2.0  # 직선인 경우 페널티
            
            # 거리 점수 (가까울수록 좋음)
            distance_score = max(0, 5.0 - distance) / 5.0
            car_score += distance_score
            
            # 크기 점수 (F1 tenth 표준 크기에 가까울수록 좋음)
            ideal_width = 0.22  # F1 tenth 표준 폭
            ideal_height = 0.33  # F1 tenth 표준 길이
            size_score = 2.0 - (abs(width - ideal_width) + abs(height - ideal_height))
            car_score += max(0, size_score)
            
            self.get_logger().debug(
                f'클러스터 분석 - 크기: {width:.2f}x{height:.2f}, '
                f'포인트: {cluster["size"]}, 거리: {distance:.2f}, '
                f'ㄷ모양: {is_u_shape}({u_shape_score:.3f}), '
                f'직선: {is_linear}({linearity_score:.3f}), '
                f'차량점수: {car_score:.3f}'
            )
            
            # 차량 후보로 추가 (최소 점수 기준)
            if car_score > 0.5:  # 최소 점수 임계값
                car_candidates.append({
                    'cluster': cluster,
                    'score': car_score
                })
        
        # 점수 순으로 정렬하여 가장 좋은 후보 1개만 선택
        if car_candidates:
            car_candidates.sort(key=lambda x: x['score'], reverse=True)
            best_candidate = car_candidates[0]
            
            self.get_logger().info(
                f'전체 클러스터: {len(clusters)}, 후보: {len(car_candidates)}개, '
                f'선택된 차량 점수: {best_candidate["score"]:.3f}'
            )
            
            return [best_candidate['cluster']]
        else:
            self.get_logger().info(f'전체 클러스터: {len(clusters)}, 자동차 후보 없음')
            return []

    def track_cars(self, detections):
        """차량 추적 수행 (단일 차량)"""
        # 모든 추적기에 대해 예측 수행
        for tracker in self.trackers:
            tracker.predict()
        
        # 단일 차량 가정하에 추적
        if len(detections) == 0:
            return self.trackers
        
        # 가장 좋은 검출 하나만 사용
        detection = detections[0]
        center = detection['center']
        
        if len(self.trackers) == 0:
            # 첫 번째 추적기 생성
            new_tracker = CarTracker(center[0], center[1], self.next_id)
            self.trackers.append(new_tracker)
            self.next_id += 1
        else:
            # 기존 추적기와 연관
            best_tracker = None
            min_distance = float('inf')
            
            for tracker in self.trackers:
                pred_pos = tracker.get_predicted_position()
                distance = np.linalg.norm(np.array(center) - np.array(pred_pos))
                
                if distance < min_distance and distance < self.max_association_distance:
                    min_distance = distance
                    best_tracker = tracker
            
            if best_tracker:
                # 기존 추적기 업데이트
                best_tracker.update(center)
                # 다른 추적기들은 제거 (단일 차량이므로)
                self.trackers = [best_tracker]
            else:
                # 새 추적기 생성하고 기존 것들 제거
                new_tracker = CarTracker(center[0], center[1], self.next_id)
                self.trackers = [new_tracker]
                self.next_id += 1
        
        # 오래된 추적기 제거 (5프레임 이상 업데이트 안됨)
        self.trackers = [t for t in self.trackers if t.time_since_update < 5]
        
        # 단일 차량이므로 최대 1개만 유지
        if len(self.trackers) > 1:
            # 가장 최근에 업데이트된 추적기만 유지
            self.trackers.sort(key=lambda x: x.time_since_update)
            self.trackers = [self.trackers[0]]
        
        return self.trackers

    def associate_detections_to_trackers(self, detections):
        """검출과 추적기를 연관시키는 간단한 알고리즘"""
        if len(self.trackers) == 0:
            return [], list(range(len(detections)))
        
        # 추적기 예측 위치
        tracker_positions = np.array([list(tracker.get_predicted_position()) for tracker in self.trackers])
        
        # 검출 위치
        detection_positions = np.array([detection['center'] for detection in detections])
        
        # 거리 행렬 계산
        distance_matrix = cdist(tracker_positions, detection_positions)
        
        # 간단한 탐욕적 매칭
        matched_pairs = []
        used_trackers = set()
        used_detections = set()
        
        # 거리 순으로 정렬하여 매칭
        tracker_det_pairs = []
        for t_idx in range(len(self.trackers)):
            for d_idx in range(len(detections)):
                dist = distance_matrix[t_idx, d_idx]
                if dist < self.max_association_distance:
                    tracker_det_pairs.append((dist, t_idx, d_idx))
        
        # 거리 순으로 정렬
        tracker_det_pairs.sort(key=lambda x: x[0])
        
        # 매칭 수행
        for _, t_idx, d_idx in tracker_det_pairs:
            if t_idx not in used_trackers and d_idx not in used_detections:
                matched_pairs.append((t_idx, d_idx))
                used_trackers.add(t_idx)
                used_detections.add(d_idx)
        
        # 매칭되지 않은 검출 찾기
        unmatched_detections = [i for i in range(len(detections)) if i not in used_detections]
        
        return matched_pairs, unmatched_detections

    def predict_tracked_cars(self):
        """검출 없이 추적만 수행"""
        for tracker in self.trackers:
            tracker.predict()

    def publish_tracked_markers(self, tracked_cars, timestamp):
        """추적된 차량들을 RViz 마커로 발행"""
        marker_array = MarkerArray()
        
        # 먼저 모든 기존 마커를 삭제하는 DELETEALL 마커 추가
        if self.previous_marker_count > 0:
            delete_all_marker = Marker()
            delete_all_marker.header.frame_id = 'laser'
            delete_all_marker.header.stamp = timestamp
            delete_all_marker.ns = 'tracked_cars'
            delete_all_marker.action = Marker.DELETEALL
            marker_array.markers.append(delete_all_marker)
        
        # 추적된 차량들에 대한 마커 생성
        for tracker in tracked_cars:
            pos_x, pos_y = tracker.get_predicted_position()
            vel_x, vel_y = tracker.get_velocity()
            speed = math.sqrt(vel_x**2 + vel_y**2)
            
            # 위치 마커 (구)
            position_marker = Marker()
            position_marker.header.frame_id = 'laser'
            position_marker.header.stamp = timestamp
            position_marker.ns = 'tracked_cars'
            position_marker.id = tracker.id * 10  # ID 충돌 방지
            position_marker.type = Marker.SPHERE
            position_marker.action = Marker.ADD
            
            position_marker.pose.position.x = float(pos_x)
            position_marker.pose.position.y = float(pos_y)
            position_marker.pose.position.z = 0.0
            position_marker.pose.orientation.w = 1.0
            
            # 차량 ID에 따라 색상 변경
            colors = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), 
                     (1.0, 1.0, 0.0), (1.0, 0.0, 1.0), (0.0, 1.0, 1.0)]
            color = colors[tracker.id % len(colors)]
            
            position_marker.color.r = color[0]
            position_marker.color.g = color[1]
            position_marker.color.b = color[2]
            position_marker.color.a = 1.0
            
            position_marker.scale.x = 0.3
            position_marker.scale.y = 0.3
            position_marker.scale.z = 0.3
            
            position_marker.lifetime.sec = 1
            position_marker.lifetime.nanosec = 0
            marker_array.markers.append(position_marker)
            
            # 속도 벡터 마커 (화살표)
            if speed > 0.1:  # 최소 속도 임계값
                velocity_marker = Marker()
                velocity_marker.header.frame_id = 'laser'
                velocity_marker.header.stamp = timestamp
                velocity_marker.ns = 'tracked_cars'
                velocity_marker.id = tracker.id * 10 + 1  # ID 충돌 방지
                velocity_marker.type = Marker.ARROW
                velocity_marker.action = Marker.ADD
                
                # 화살표 시작점
                velocity_marker.pose.position.x = float(pos_x)
                velocity_marker.pose.position.y = float(pos_y)
                velocity_marker.pose.position.z = 0.1
                
                # 화살표 방향 (속도 벡터)
                velocity_angle = math.atan2(vel_y, vel_x)
                velocity_marker.pose.orientation.z = math.sin(velocity_angle / 2)
                velocity_marker.pose.orientation.w = math.cos(velocity_angle / 2)
                
                # 화살표 크기 (속도에 비례)
                arrow_length = min(speed * 0.5, 1.0)  # 최대 1m
                velocity_marker.scale.x = arrow_length
                velocity_marker.scale.y = 0.05
                velocity_marker.scale.z = 0.05
                
                velocity_marker.color.r = color[0]
                velocity_marker.color.g = color[1]
                velocity_marker.color.b = color[2]
                velocity_marker.color.a = 0.7
                
                velocity_marker.lifetime.sec = 1
                velocity_marker.lifetime.nanosec = 0
                marker_array.markers.append(velocity_marker)
            
            # ID 텍스트 마커
            text_marker = Marker()
            text_marker.header.frame_id = 'laser'
            text_marker.header.stamp = timestamp
            text_marker.ns = 'tracked_cars'
            text_marker.id = tracker.id * 10 + 2  # ID 충돌 방지
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            
            text_marker.pose.position.x = float(pos_x)
            text_marker.pose.position.y = float(pos_y)
            text_marker.pose.position.z = 0.3
            text_marker.pose.orientation.w = 1.0
            
            text_marker.text = f"ID:{tracker.id}\n{speed:.1f}m/s"
            text_marker.scale.z = 0.2
            
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            text_marker.color.a = 1.0
            
            text_marker.lifetime.sec = 1
            text_marker.lifetime.nanosec = 0
            marker_array.markers.append(text_marker)
        
        # 현재 마커 수 업데이트
        self.previous_marker_count = len(tracked_cars) * 3  # 각 차량당 3개 마커
        
        self.cluster_marker_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    node = LidarProcessor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
