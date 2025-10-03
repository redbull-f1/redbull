#!/usr/bin/env python3
"""
[ModelMarkerPublisher 노드 설명]

이 코드는 ROS2 환경에서 LiDAR 센서 데이터를 받아 TinyCenterSpeed 신경망을 이용해 실시간으로 주변 객체(차량 등)의 중심점을 예측하고,
해당 위치를 RViz2에서 MarkerArray로 시각화하는 노드입니다.

- /scan 토픽(LaserScan)을 구독하여 2프레임씩 전처리 후 6채널 입력으로 만듭니다.
- TinyCenterSpeed 신경망(CenterSpeedDense)으로 히트맵(heatmap) 예측을 수행합니다.
- 히트맵에서 k개의 피크(=객체 중심점)를 찾아 x, y 좌표로 변환합니다.
- 예측된 중심점 좌표를 MarkerArray로 RViz2에 시각화합니다.
- 파라미터로 이미지 크기, 픽셀 크기, 감지 임계값, 객체 개수 등을 설정할 수 있습니다.

※ heatmap에서 찾은 x, y는 각 객체의 중심점(centroid) 좌표입니다.

주요 함수 및 역할:
- preprocess_scan: LiDAR raw 데이터를 이미지로 변환(전처리)
- find_k_peaks: 히트맵에서 k개의 피크(=객체 중심점) 추출
- publish_markers: 예측된 중심점을 MarkerArray로 시각화

실제 차량 시뮬레이션, 자율주행 시뮬레이션 등에서 실시간 객체 중심점 감지 및 시각화에 활용할 수 있습니다.
"""
import os
import sys
import numpy as np
import torch
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import LaserScan

# 모델 로드 함수 (bag2csv_model_sim.py와 동일)
def get_model(redbull_root, image_size=64, device='cpu'):
    model_paths = [
        os.path.join(redbull_root, 'train'),
        os.path.join(redbull_root, 'train', 'models'),
        redbull_root
    ]
    for path in model_paths:
        if os.path.exists(path) and path not in sys.path:
            sys.path.insert(0, path)
    from models.CenterSpeed import CenterSpeedDense
    model = CenterSpeedDense(input_channels=6, image_size=image_size)
    model_path = os.path.join(redbull_root, 'train', 'trained_models', 'TinyCenterSpeed.pt')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.eval()
    model.to(device)
    return model

def preprocess_scan(scan_msg, image_size=64, pixelsize=0.1):
    lidar_data = np.array(scan_msg.ranges, dtype=np.float32)
    origin_offset = (image_size // 2) * pixelsize
    if hasattr(scan_msg, 'intensities') and len(scan_msg.intensities) > 0:
        intensities = np.array(scan_msg.intensities, dtype=np.float32)
        if intensities.max() > intensities.min():
            intensities = (intensities - intensities.min()) / (intensities.max() - intensities.min())
        else:
            intensities = np.full_like(lidar_data, 0.5, dtype=np.float32)
    else:
        intensities = np.full_like(lidar_data, 0.5, dtype=np.float32)
    angles = np.linspace(scan_msg.angle_min, scan_msg.angle_max, len(lidar_data))
    cos_angles = np.cos(angles)
    sin_angles = np.sin(angles)
    preprocessed_scans = np.zeros((1, 3, image_size, image_size), dtype=np.float32)
    x = lidar_data * cos_angles
    y = lidar_data * sin_angles
    forward_filter = x >= 0
    x = x[forward_filter]
    y = y[forward_filter]
    intensities = intensities[forward_filter]
    x_coord = ((x + origin_offset) / pixelsize).astype(int)
    y_coord = ((y + origin_offset) / pixelsize).astype(int)
    valid_indices = (x_coord >= 0) & (x_coord < image_size) & (y_coord >= 0) & (y_coord < image_size)
    x_coord = x_coord[valid_indices]
    y_coord = y_coord[valid_indices]
    if len(x_coord) > 0:
        preprocessed_scans[:, 0, y_coord, x_coord] = 1
        preprocessed_scans[:, 1, y_coord, x_coord] = np.maximum(
            preprocessed_scans[:, 1, y_coord, x_coord], intensities[valid_indices])
        preprocessed_scans[:, 2, y_coord, x_coord] += 1
    return preprocessed_scans

def find_k_peaks(image, k, threshold=0.3, pixelsize=0.1, origin_offset=3.2):
    radius = 8 / pixelsize
    image = image.copy()
    opp_coordinates = np.zeros((k, 2))
    valid_peaks = 0
    for i in range(k):
        max_idx = np.argmax(image.reshape(-1))
        if image.flat[max_idx] < threshold:
            break
        max_coords = np.unravel_index(max_idx, image.shape)
        if len(max_coords) == 2:
            opp_coordinates[i, 0] = max_coords[1]  # x
            opp_coordinates[i, 1] = max_coords[0]  # y
        else:
            break
        valid_peaks += 1
        if i == k - 1:
            break
        top = max(0, int(max_coords[0] - radius))
        bottom = min(image.shape[0], int(max_coords[0] + radius))
        left = max(0, int(max_coords[1] - radius))
        right = min(image.shape[1], int(max_coords[1] + radius))
        image[top:bottom, left:right] = 0
    if valid_peaks == 0:
        return [], []
    x = opp_coordinates[:valid_peaks, 0] * pixelsize - origin_offset
    y = opp_coordinates[:valid_peaks, 1] * pixelsize - origin_offset
    return x, y

class ModelMarkerPublisher(Node):
    def __init__(self):
        super().__init__('model_marker_publisher')
        # 파라미터 선언 (sim_detector.py 스타일)
        self.declare_parameter('image_size', 64)
        self.declare_parameter('pixelsize', 0.1)
        self.declare_parameter('num_opponents', 1)
        self.declare_parameter('threshold', 0.3)
        self.declare_parameter('marker_topic', 'model_obstacles')
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('laser_frame', 'laser')

        self.image_size = self.get_parameter('image_size').value
        self.pixelsize = self.get_parameter('pixelsize').value
        self.num_opponents = self.get_parameter('num_opponents').value
        self.threshold = self.get_parameter('threshold').value
        self.marker_topic = self.get_parameter('marker_topic').value
        self.scan_topic = self.get_parameter('scan_topic').value
        self.laser_frame = self.get_parameter('laser_frame').value
        self.origin_offset = (self.image_size // 2) * self.pixelsize

        # 모델 로드
        redbull_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = get_model(redbull_root, image_size=self.image_size, device=device)
        self.frame1 = None
        self.frame2 = None
        self.last_peaks = []
        self.last_stamp = None

        self.marker_pub = self.create_publisher(MarkerArray, self.marker_topic, 10)
        self.subscription = self.create_subscription(
            LaserScan,
            self.scan_topic,
            self.scan_callback,
            10)
        self.timer = self.create_timer(0.05, self.publish_markers)  # 20Hz

    def scan_callback(self, msg):
        if self.frame1 is None:
            self.frame1 = preprocess_scan(msg, self.image_size, self.pixelsize)
            return
        if self.frame2 is None:
            self.frame2 = preprocess_scan(msg, self.image_size, self.pixelsize)
            return
        self.frame1 = self.frame2
        self.frame2 = preprocess_scan(msg, self.image_size, self.pixelsize)
        input_tensor = np.concatenate([self.frame1, self.frame2], axis=1)
        input_tensor = torch.FloatTensor(input_tensor).to(next(self.model.parameters()).device)
        with torch.no_grad():
            outputs = self.model(input_tensor)
        if isinstance(outputs, torch.Tensor):
            output = outputs.cpu().numpy()
        else:
            output = outputs[0].cpu().numpy()
        if output.ndim == 4:
            output = output[0]
        heatmap = output[0]
        x, y = find_k_peaks(heatmap, self.num_opponents, self.threshold, self.pixelsize, self.origin_offset)
        self.last_peaks = list(zip(x, y))
        self.last_stamp = msg.header.stamp

    def publish_markers(self):
        marker_array = MarkerArray()
        for i, (x, y) in enumerate(self.last_peaks):
            marker = Marker()
            marker.header.frame_id = self.laser_frame
            marker.header.stamp = self.last_stamp if self.last_stamp else self.get_clock().now().to_msg()
            marker.ns = 'model_obstacles'
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = float(x)
            marker.pose.position.y = float(y)
            marker.pose.position.z = 0.0
            marker.scale.x = 0.3
            marker.scale.y = 0.3
            marker.scale.z = 0.3
            marker.color.a = 1.0
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker_array.markers.append(marker)
        self.marker_pub.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    node = ModelMarkerPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
