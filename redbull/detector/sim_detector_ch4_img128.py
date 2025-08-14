#!/usr/bin/env python3
"""
[SimDetector 노드 설명]

이 코드는 ROS2 환경에서 LiDAR 센서와 Odometry 데이터를 받아 TinyCenterSpeed 신경망을 이용해 실시간으로 주변 차량(상대 차량)을 감지하고, 그 결과를 RViz2에서 시각화(MarkerArray)로 보여주는 노드입니다.

- LiDAR 스캔(/scan)과 Odometry(/ego_racecar/odom) 토픽을 구독합니다.
- 2프레임의 LiDAR 데이터를 전처리하여 4채널 입력으로 만듭니다.
- TinyCenterSpeed 신경망(CenterSpeedDense)으로 히트맵(heatmap) 예측을 수행합니다.
- 히트맵에서 k개의 피크(상대 차량 위치 후보)를 찾고, 이를 차량 좌표계에서 지도 좌표계(map)로 변환합니다.
- 감지된 차량 위치를 MarkerArray로 RViz2에 시각화합니다.
- 파라미터로 모델 경로, 이미지 크기, 감지 임계값, 픽셀 크기, 토픽명 등을 설정할 수 있습니다.

주요 함수 및 역할:
- preprocess_scan: LiDAR raw 데이터를 이미지로 변환(전처리)
- find_k_peaks: 히트맵에서 k개의 피크(상대 차량 위치) 추출
- lidar_to_map: 차량 좌표계(x, y)를 지도 좌표계로 변환
- process: 주기적으로 신경망 추론 및 결과 시각화

실제 차량 시뮬레이션, 자율주행 시뮬레이션 등에서 실시간 상대 차량 감지 및 시각화에 활용할 수 있습니다.
"""

import rclpy # rclpy를 사용하면 ROS 2 메시지 주고받기, 노드 생성 및 관리, 서비스 호출, 파라미터 설정 등 다양한 작업을 Python으로 처리할 수 있습니다. 
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import numpy as np
import torch
import os # 운영체제 
import sys # 파이썬 인터프리터 

class SimDetector(Node):
    def __init__(self):
        super().__init__('sim_detector')
        # Parameters
        self.declare_parameter('model_path', 
                               '/home/harry/ros2_ws/src/TinyCenterSpeed/src/pt/centerspeed_best_epoch_1_24_good.pt')


        self.declare_parameter('image_size', 128)
        self.declare_parameter('dense', True)
        self.declare_parameter('num_opponents', 1)
        self.declare_parameter('detection_threshold', 0.57) #0.67이 좋음 /home/harry/ros2_ws/src/TinyCenterSpeed/src/pt/centerspeed_best_epoch_1_24_good.pt
        self.declare_parameter('pixelsize', 0.1)
        self.declare_parameter('origin_offset', None)
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('laser_frame', 'laser')
        self.declare_parameter('odom_topic', '/ego_racecar/odom')
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('marker_topic', '/sim_detector_objects')

        self.model_path = self.get_parameter('model_path').value
        self.image_size = self.get_parameter('image_size').value
        self.dense = self.get_parameter('dense').value
        self.num_opponents = self.get_parameter('num_opponents').value
        self.detection_threshold = self.get_parameter('detection_threshold').value
        self.pixelsize = self.get_parameter('pixelsize').value
        self.origin_offset = (self.image_size // 2) * self.pixelsize
        self.map_frame = self.get_parameter('map_frame').value
        self.laser_frame = self.get_parameter('laser_frame').value
        self.odom_topic = self.get_parameter('odom_topic').value
        self.scan_topic = self.get_parameter('scan_topic').value
        self.marker_topic = self.get_parameter('marker_topic').value

        # Model import
        current_dir = os.path.dirname(os.path.abspath(__file__))
        redbull_root = os.path.dirname(current_dir)
        model_paths = [
            os.path.join(redbull_root, 'train'),
            os.path.join(redbull_root, 'train', 'models'),
            redbull_root
        ]


        for p in model_paths:
            if os.path.exists(p) and p not in sys.path:
                sys.path.insert(0, p)
        # models 폴더 경로를 sys.path에 추가
        models_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/models'))
        print(f"[DEBUG] models_path: {models_path}")
        if models_path not in sys.path:
            sys.path.insert(0, models_path)
        try:
            from CenterSpeed import CenterSpeedDense
        except ImportError as e:
            print(f"[ERROR] CenterSpeed import 실패: {e}")
            raise

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            print("cuda")
        else:
            print("cpu")

        # Instantiate CenterSpeedDense for 4-channel input (2 frames)
        self.net = CenterSpeedDense(input_channels=4, image_size=self.image_size)
        self.net.load_state_dict(torch.load(self.model_path, map_location=self.device), strict=False)
        self.net.eval()
        self.net.to(self.device)

        # ROS2
        self.odom = None
        self.scan_data = None
        self.create_subscription(Odometry, self.odom_topic, self.odom_callback, 10)
        self.create_subscription(LaserScan, self.scan_topic, self.scan_callback, 10)
        self.marker_pub = self.create_publisher(MarkerArray, self.marker_topic, 10)
        self.frame1 = None
        self.frame2 = None
        self.timer = self.create_timer(1.0 / 40.0, self.process)

    def odom_callback(self, msg):
        self.odom = msg

    def scan_callback(self, msg):
        self.scan_data = msg

    def preprocess_scan(self, scan_msg):
        lidar_data = np.array(scan_msg.ranges, dtype=np.float32)
        # intensities 채널은 사용하지 않고, occupancy/density만 사용한다고 가정 (4채널)
        angles = np.linspace(scan_msg.angle_min, scan_msg.angle_max, len(lidar_data))
        cos_angles = np.cos(angles)
        sin_angles = np.sin(angles)
        preprocessed_scans = np.zeros((1, 2, self.image_size, self.image_size), dtype=np.float32)
        x = lidar_data * cos_angles
        y = lidar_data * sin_angles
        forward_filter = x >= 0
        x = x[forward_filter]
        y = y[forward_filter]
        x_coord = ((x + self.origin_offset) / self.pixelsize).astype(int)
        y_coord = ((y + self.origin_offset) / self.pixelsize).astype(int)
        valid_indices = (x_coord >= 0) & (x_coord < self.image_size) & (y_coord >= 0) & (y_coord < self.image_size)
        x_coord = x_coord[valid_indices]
        y_coord = y_coord[valid_indices]
        if len(x_coord) > 0:
            preprocessed_scans[:, 0, y_coord, x_coord] = 1  # occupancy
            preprocessed_scans[:, 1, y_coord, x_coord] += 1 # density
        return preprocessed_scans

    def index_to_cartesian(self, x_img, y_img):
        x = x_img * self.pixelsize - self.origin_offset
        y = y_img * self.pixelsize - self.origin_offset
        return x, y

    def find_k_peaks(self, image, k, threshold=0.3, radius=8):
        radius = radius / self.pixelsize
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
        x = opp_coordinates[:valid_peaks, 0] * self.pixelsize - self.origin_offset
        y = opp_coordinates[:valid_peaks, 1] * self.pixelsize - self.origin_offset
        return x, y

    def lidar_to_map(self, x_lidar, y_lidar, odom):
        px = odom.pose.pose.position.x
        py = odom.pose.pose.position.y
        q = odom.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        x_map = np.cos(yaw) * x_lidar - np.sin(yaw) * y_lidar + px
        y_map = np.sin(yaw) * x_lidar + np.cos(yaw) * y_lidar + py
        return x_map, y_map
    
    def process(self):
        if self.scan_data is None or self.odom is None:
            return

        if self.frame1 is None:
            self.frame1 = self.preprocess_scan(self.scan_data)
            return
        if self.frame2 is None:
            self.frame2 = self.preprocess_scan(self.scan_data)
            return

        # 최신 2프레임 구성
        self.frame1 = self.frame2
        self.frame2 = self.preprocess_scan(self.scan_data)

        # (1, 4, H, W)
        input_tensor = np.concatenate([self.frame1, self.frame2], axis=1)
        input_tensor = torch.from_numpy(input_tensor).float().to(self.device)


        ### 추론 시간 측정 
        import time
        start = time.time()
        with torch.no_grad():
            outputs = self.net(input_tensor)      # 기대: [1, C, H, W] (C는 4라고 가정)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
        end = time.time()
        # print(f"Inference time: {end - start:.4f} seconds")


        # --- 여기부터가 핵심 수정 ---
        # heatmap 채널 명시적으로 선택 후 sigmoid로 [0,1] 확률화
        heatmap = outputs[0, 0]                   # 배치 0, 채널 0 = heatmap
        heatmap = torch.sigmoid(heatmap).cpu().numpy()
        # --- 핵심 수정 끝 ---

        # 피크 탐색 (threshold는 0.4~0.6대 권장)
        x, y = self.find_k_peaks(heatmap, self.num_opponents, self.detection_threshold)

        marker_array = MarkerArray()
        clear_marker = Marker()
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)

        if len(x) > 0:
            for i, (x_lidar, y_lidar) in enumerate(zip(x, y)):
                x_map, y_map = self.lidar_to_map(x_lidar, y_lidar, self.odom)
                marker = Marker()
                marker.header.frame_id = self.map_frame
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.ns = "sim_detector"
                marker.id = i
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                marker.pose.position.x = x_map
                marker.pose.position.y = y_map
                marker.pose.position.z = 0.1
                marker.scale.x = 0.4
                marker.scale.y = 0.4
                marker.scale.z = 0.4
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
                marker.color.a = 1.0
                marker_array.markers.append(marker)

        self.marker_pub.publish(marker_array)




    # def process(self):
    #     if self.scan_data is None or self.odom is None:
    #         return
    #     if self.frame1 is None:
    #         self.frame1 = self.preprocess_scan(self.scan_data)
    #         return
    #     if self.frame2 is None:
    #         self.frame2 = self.preprocess_scan(self.scan_data)
    #         return
    #     self.frame1 = self.frame2
    #     self.frame2 = self.preprocess_scan(self.scan_data)
    #     if self.frame1 is None or self.frame2 is None:
    #         return
    #     input_tensor = np.concatenate([self.frame1, self.frame2], axis=1)  # (1, 4, H, W)
    #     input_tensor = torch.FloatTensor(input_tensor).to(self.device)
    #     with torch.no_grad():
    #         outputs = self.net(input_tensor)
    #     if isinstance(outputs, tuple):
    #         output_hm = outputs[0]
    #     else:
    #         output_hm = outputs
    #     if len(output_hm.shape) == 4:
    #         output_hm = output_hm.squeeze(0)
    #     if len(output_hm.shape) == 3:
    #         output_hm = output_hm[0]
    #     if len(output_hm.shape) != 2:
    #         return
    #     x, y = self.find_k_peaks(output_hm.cpu().numpy(), self.num_opponents, self.detection_threshold)
    #     marker_array = MarkerArray()
    #     clear_marker = Marker()
    #     clear_marker.action = Marker.DELETEALL
    #     marker_array.markers.append(clear_marker)
    #     if len(x) > 0:
    #         for i, (x_lidar, y_lidar) in enumerate(zip(x, y)):
    #             x_map, y_map = self.lidar_to_map(x_lidar, y_lidar, self.odom)
    #             marker = Marker()
    #             marker.header.frame_id = self.map_frame
    #             marker.header.stamp = self.get_clock().now().to_msg()
    #             marker.ns = "sim_detector"
    #             marker.id = i
    #             marker.type = Marker.SPHERE
    #             marker.action = Marker.ADD
    #             marker.pose.position.x = x_map
    #             marker.pose.position.y = y_map
    #             marker.pose.position.z = 0.1
    #             marker.scale.x = 0.4
    #             marker.scale.y = 0.4
    #             marker.scale.z = 0.4
    #             marker.color.r = 1.0
    #             marker.color.g = 0.0
    #             marker.color.b = 0.0
    #             marker.color.a = 1.0
    #             marker.lifetime.sec = 0
    #             marker.lifetime.nanosec = 0
    #             marker_array.markers.append(marker)
    #     self.marker_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    node = SimDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
