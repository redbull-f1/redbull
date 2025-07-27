import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import numpy as np
from sklearn.cluster import DBSCAN
import math


// 현재 이 코드는 lidar 데이터를 처리하고 F1 tenth 차를 detect 하는 기능을 포함하고 있습니다.

class LidarProcessor(Node):
    def __init__(self):
        super().__init__('lidar_processor')
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )
        self.object_scan_pub = self.create_publisher(LaserScan, '/object_scan', 10)
        self.cluster_marker_pub = self.create_publisher(MarkerArray, '/cluster_markers', 10)
        
        # 클러스터링 파라미터
        self.eps = 0.3  # 최대 거리 (미터)
        self.min_samples = 3  # 최소 포인트 수
        
        # 자동차 감지 파라미터 (더 관대하게 설정)
        self.min_cluster_size = 5  # 자동차로 인식하기 위한 최소 포인트 수 (완화)
        self.max_cluster_size = 80  # 자동차로 인식하기 위한 최대 포인트 수 (증가)
        self.min_car_width = 0.2  # 자동차 최소 폭 (완화)
        self.max_car_width = 2.0  # 자동차 최대 폭 (증가)
        self.min_car_length = 0.3  # 자동차 최소 길이 (완화)
        self.max_car_length = 5.0  # 자동차 최대 길이 (증가)
        self.min_aspect_ratio = 1.0  # 최소 길이/폭 비율 (완화)
        self.max_aspect_ratio = 10.0  # 최대 길이/폭 비율 (증가)
        
        # 마커 관리
        self.previous_marker_count = 0

    def scan_callback(self, msg):
        ranges = np.array(msg.ranges)
        # 3m 초과하는 값은 inf로 처리
        filtered_ranges = np.where(ranges <= 3.0, ranges, float('inf'))

        # 유효한 점들만 추출하여 클러스터링
        valid_points = self.extract_valid_points(msg, filtered_ranges)
        
        if len(valid_points) > 0:
            clusters = self.perform_clustering(valid_points)
            car_clusters = self.filter_car_clusters(clusters)
            self.publish_cluster_markers(car_clusters, msg.header.stamp)
            
            # 감지 결과 로그 (더 상세하게)
            if len(clusters) > 0:
                self.get_logger().info(f'총 {len(clusters)}개 클러스터 중 {len(car_clusters)}개가 자동차로 인식됨')
                for i, cluster in enumerate(car_clusters):
                    points = cluster['points']
                    min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
                    min_y, max_y = np.min(points[:, 1]), np.max(points[:, 1])
                    width, length = max_y - min_y, max_x - min_x
                    distance = np.sqrt(cluster['center'][0]**2 + cluster['center'][1]**2)
                    self.get_logger().info(f'자동차 {i+1}: 크기({length:.2f}m x {width:.2f}m), 거리({distance:.2f}m)')
        else:
            # 유효한 포인트가 없으면 빈 마커 배열 발행
            self.publish_cluster_markers([], msg.header.stamp)

        # 기존 필터링된 스캔 데이터 발행
        object_scan = LaserScan()
        object_scan.header.frame_id = 'laser'
        object_scan.header.stamp = self.get_clock().now().to_msg()
        object_scan.angle_min = msg.angle_min
        object_scan.angle_max = msg.angle_max
        object_scan.angle_increment = msg.angle_increment
        object_scan.time_increment = msg.time_increment
        object_scan.scan_time = msg.scan_time
        object_scan.range_min = msg.range_min
        object_scan.range_max = msg.range_max
        object_scan.ranges = filtered_ranges.tolist()
        object_scan.intensities = [1.0]*len(filtered_ranges)
        self.object_scan_pub.publish(object_scan)

    def extract_valid_points(self, msg, ranges):
        """유효한 LiDAR 포인트들을 cartesian 좌표로 변환"""
        valid_points = []
        for i, range_val in enumerate(ranges):
            if not math.isinf(range_val) and range_val > msg.range_min:
                angle = msg.angle_min + i * msg.angle_increment
                x = range_val * math.cos(angle)
                y = range_val * math.sin(angle)
                valid_points.append([x, y])
        return np.array(valid_points)

    def perform_clustering(self, points):
        """DBSCAN을 사용하여 포인트 클러스터링"""
        if len(points) < self.min_samples:
            return []
        
        clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        cluster_labels = clustering.fit_predict(points)
        
        clusters = []
        unique_labels = set(cluster_labels)
        
        for label in unique_labels:
            if label == -1:  # 노이즈 포인트 제외
                continue
            
            cluster_points = points[cluster_labels == label]
            if len(cluster_points) >= self.min_samples:
                # 클러스터 중심점 계산
                center_x = float(np.mean(cluster_points[:, 0]))
                center_y = float(np.mean(cluster_points[:, 1]))
                clusters.append({'center': [center_x, center_y], 'points': cluster_points})
        
        return clusters

    def filter_car_clusters(self, clusters):
        """자동차로 추정되는 클러스터만 필터링"""
        car_clusters = []
        
        self.get_logger().info(f'필터링 시작: {len(clusters)}개 클러스터 검사')
        
        for i, cluster in enumerate(clusters):
            points = cluster['points']
            
            # 디버깅을 위한 기본 정보
            self.get_logger().info(f'클러스터 {i}: 포인트 수 = {len(points)}')
            
            # 포인트 수 필터링 (완화)
            if len(points) < self.min_cluster_size:
                self.get_logger().info(f'클러스터 {i}: 포인트 수 부족 ({len(points)} < {self.min_cluster_size})')
                continue
            if len(points) > self.max_cluster_size:
                self.get_logger().info(f'클러스터 {i}: 포인트 수 초과 ({len(points)} > {self.max_cluster_size})')
                continue
            
            # 클러스터의 바운딩 박스 계산
            min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
            min_y, max_y = np.min(points[:, 1]), np.max(points[:, 1])
            
            width = max_y - min_y  # y축 방향 폭
            length = max_x - min_x  # x축 방향 길이
            
            self.get_logger().info(f'클러스터 {i}: 크기 = {length:.2f}m x {width:.2f}m')
            
            # 크기가 너무 작으면 제외 (더 관대하게)
            if width < 0.05 or length < 0.05:
                self.get_logger().info(f'클러스터 {i}: 크기 너무 작음')
                continue
            
            # 자동차 크기 필터링 (더 관대하게)
            if not (self.min_car_width <= width <= self.max_car_width):
                self.get_logger().info(f'클러스터 {i}: 폭이 범위를 벗어남 ({width:.2f}m)')
                continue
            if not (self.min_car_length <= length <= self.max_car_length):
                self.get_logger().info(f'클러스터 {i}: 길이가 범위를 벗어남 ({length:.2f}m)')
                continue
            
            # 형태 비율 검사 (더 관대하게)
            aspect_ratio = max(length, width) / min(length, width) if min(length, width) > 0 else 0
            self.get_logger().info(f'클러스터 {i}: 비율 = {aspect_ratio:.2f}')
            if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
                self.get_logger().info(f'클러스터 {i}: 비율이 범위를 벗어남 ({aspect_ratio:.2f})')
                continue
            
            # 포인트 밀도 검사 (더 관대하게)
            area = length * width
            density = len(points) / area if area > 0 else 0
            self.get_logger().info(f'클러스터 {i}: 밀도 = {density:.2f}')
            if density < 2 or density > 500:  # 더 관대한 밀도 범위
                self.get_logger().info(f'클러스터 {i}: 밀도가 범위를 벗어남 ({density:.2f})')
                continue
            
            # 거리 필터링 (더 관대하게)
            center_distance = np.sqrt(cluster['center'][0]**2 + cluster['center'][1]**2)
            self.get_logger().info(f'클러스터 {i}: 거리 = {center_distance:.2f}m')
            if center_distance < 0.3:  # 0.3m 이상 떨어진 객체만 (완화)
                self.get_logger().info(f'클러스터 {i}: 너무 가까움 ({center_distance:.2f}m)')
                continue
            
            # 연속성 검사를 더 관대하게 또는 일시적으로 비활성화
            continuity = self.check_point_continuity(points)
            self.get_logger().info(f'클러스터 {i}: 연속성 = {continuity}')
            # if not continuity:
            #     self.get_logger().info(f'클러스터 {i}: 연속성 부족')
            #     continue
            
            # 모든 조건을 통과한 클러스터
            self.get_logger().info(f'클러스터 {i}: 자동차로 인식됨!')
            car_clusters.append(cluster)
        
        return car_clusters

    def check_point_continuity(self, points):
        """포인트들이 연속적으로 분포하는지 검사 (더 관대하게)"""
        if len(points) < 3:
            return True  # 포인트가 적으면 통과
        
        # 중심점으로부터의 각도 계산
        center_x = np.mean(points[:, 0])
        center_y = np.mean(points[:, 1])
        
        angles = []
        for point in points:
            angle = math.atan2(point[1] - center_y, point[0] - center_x)
            angles.append(angle)
        
        angles.sort()
        
        # 각도 차이들 계산
        angle_diffs = []
        for i in range(1, len(angles)):
            diff = angles[i] - angles[i-1]
            # 2π 경계 처리
            if diff > math.pi:
                diff = diff - 2*math.pi
            elif diff < -math.pi:
                diff = diff + 2*math.pi
            angle_diffs.append(abs(diff))
        
        # 연속성 검사: 더 관대한 기준
        large_gaps = sum(1 for diff in angle_diffs if diff > math.pi/2)  # 90도 이상 차이 (완화)
        continuity_ratio = (len(angle_diffs) - large_gaps) / len(angle_diffs) if angle_diffs else 1
        
        return continuity_ratio > 0.3  # 30% 이상 연속적이면 통과 (완화)

    def publish_cluster_markers(self, clusters, timestamp):
        """클러스터 중심점을 RViz 마커로 발행"""
        marker_array = MarkerArray()
        
        # 이전 마커들을 모두 삭제
        for i in range(self.previous_marker_count):
            delete_marker = Marker()
            delete_marker.header.frame_id = 'laser'
            delete_marker.header.stamp = timestamp
            delete_marker.ns = 'cluster_centers'
            delete_marker.id = i
            delete_marker.action = Marker.DELETE
            marker_array.markers.append(delete_marker)
        
        # 새로운 마커들 추가
        for i, cluster in enumerate(clusters):
            marker = Marker()
            marker.header.frame_id = 'laser'
            marker.header.stamp = timestamp
            marker.ns = 'cluster_centers'
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            
            # 마커 위치 설정
            marker.pose.position.x = float(cluster['center'][0])
            marker.pose.position.y = float(cluster['center'][1])
            marker.pose.position.z = 0.0
            marker.pose.orientation.w = 1.0
            
            # 마커 크기 설정 (자동차는 더 크게)
            marker.scale.x = 0.4
            marker.scale.y = 0.4
            marker.scale.z = 0.4
            
            # 마커 색상 설정 (파란색으로 변경 - 자동차임을 표시)
            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 1.0
            marker.color.a = 1.0
            
            marker.lifetime.sec = 0  # 영구적으로 표시
            marker_array.markers.append(marker)
        
        # 현재 마커 수 업데이트
        self.previous_marker_count = len(clusters)
        
        self.cluster_marker_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    node = LidarProcessor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
