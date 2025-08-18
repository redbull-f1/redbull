#!/usr/bin/env python3
"""
[SimDetector 노드 설명 - 레이저 좌표계 시각화판]

LiDAR(/scan)과 Odometry(/ego_racecar/odom)를 구독하고,
TinyCenterSpeed(CenterSpeedDense) 모델로 heatmap(+vx,vy,yaw)을 추론한 뒤
- heatmap에서 k개 피크(상대 차량 후보)를 찾고,
- [변경] 좌표 변환 없이 LiDAR(=laser) 좌표계 그대로 사용,
- RViz2 MarkerArray로 시각화,
- redbull_msgs/ObstacleArray(ObstacleWpnt[])로 퍼블리시,
- 간단 좌표 기반 추적으로 ID 유지

주요 파라미터:
- model_path, image_size, detection_threshold, pixelsize, num_opponents
- 토픽명: odom_topic, scan_topic, marker_topic, obstacles_topic 등
- [중요] RViz Fixed Frame = 'laser'
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
# 메시지 패키지 경로는 사용자 환경에 맞춰 유지
# from redbull_msgs.msg import ObstacleArray, ObstacleWpnt
from pl_msg.msg import ObstacleArray, ObstacleWpnt


# from redbull.msg import ObstacleArray, ObstacleWpnt
# from redbull.redbull_msgs.msg import ObstacleArray, ObstacleWpnt

import numpy as np
import torch
import os
import sys
import time


class SimDetector(Node):
    def __init__(self):
        super().__init__('sim_detector')

        # -------------------- Parameters --------------------
        self.declare_parameter(
            'model_path',
            '/home/harry/ros2_ws/src/redbull/redbull/src/trained_models/redbull_objfree_trainfree24494_epoch_14.pt'
        )
        # 다른 후보 경로들(참고용)
        # /home/harry/ros2_ws/src/TinyCenterSpeed/src/pt/redbull60000_Spiel10000_epoch_34.pt
        # /home/harry/ros2_ws/src/TinyCenterSpeed/src/pt/centerspeed_best_epoch_1_24.pt
        # /home/harry/ros2_ws/src/TinyCenterSpeed/src/pt/redbull_best_epoch_1_25_loss1_374.pt
        # /home/harry/ros2_ws/src/TinyCenterSpeed/src/pt/redbull60000_Spiel10000_1floor40000_epoch_28.pt
        # /home/harry/ros2_ws/src/TinyCenterSpeed/src/pt/redbull_objfree_best_epoch_10.pt
        # /home/harry/ros2_ws/src/TinyCenterSpeed/src/pt/redbull_objfree_best_epoch_11_free.pt
        # /home/harry/ros2_ws/src/TinyCenterSpeed/src/pt/redbull_objfree_trainfree24494_epoch_14.pt
        # /home/harry/ros2_ws/src/redbull/redbull/src/trained_models/redbull_objfree_trainfree24494_epoch_14.pt   (0815 베스트)

        self.declare_parameter('image_size', 128)
        self.declare_parameter('dense', True)
        self.declare_parameter('num_opponents', 1)
        self.declare_parameter('detection_threshold', 0.85)   # heatmap 확률 스레시홀드
        self.declare_parameter('pixelsize', 0.1)
        self.declare_parameter('map_frame', 'map')             # 참고용(미사용)
        self.declare_parameter('laser_frame', 'laser')         # 시각화/퍼블리시 기준 프레임
        self.declare_parameter('odom_topic', '/ego_racecar/odom')
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('marker_topic', '/sim_detector_objects')
        self.declare_parameter('obstacles_topic', 'obstacles') # redbull_msgs/ObstacleArray 퍼블리시 토픽
        self.declare_parameter('publish_markers', True)        # RViz 시각화 on/off
        self.declare_parameter('id_resolution_m', 0.2)         # ID 유지용 라운딩 분해능(미터)

        self.model_path          = self.get_parameter('model_path').value
        self.image_size          = int(self.get_parameter('image_size').value)
        self.dense               = bool(self.get_parameter('dense').value)
        self.num_opponents       = int(self.get_parameter('num_opponents').value)
        self.detection_threshold = float(self.get_parameter('detection_threshold').value)
        self.pixelsize           = float(self.get_parameter('pixelsize').value)
        self.origin_offset       = (self.image_size // 2) * self.pixelsize
        self.map_frame           = self.get_parameter('map_frame').value
        self.laser_frame         = self.get_parameter('laser_frame').value
        self.odom_topic          = self.get_parameter('odom_topic').value
        self.scan_topic          = self.get_parameter('scan_topic').value
        self.marker_topic        = self.get_parameter('marker_topic').value
        self.obstacles_topic     = self.get_parameter('obstacles_topic').value
        self.publish_markers     = bool(self.get_parameter('publish_markers').value)
        self.id_resolution_m     = float(self.get_parameter('id_resolution_m').value)

        # -------------------- Model import --------------------
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
        self.get_logger().info(f'Using device: {self.device}')

        self.net = CenterSpeedDense(input_channels=4, image_size=self.image_size)
        state = torch.load(self.model_path, map_location=self.device)
        # 호환성 문제 대비
        if isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
        self.net.load_state_dict(state, strict=False)
        self.net.eval().to(self.device)

        # -------------------- ROS I/O --------------------
        self.odom = None
        self.scan_data = None
        self.create_subscription(Odometry,   self.odom_topic, self.odom_callback, 10)
        self.create_subscription(LaserScan,  self.scan_topic, self.scan_callback, 10)
        if self.publish_markers:
            self.marker_pub = self.create_publisher(MarkerArray, self.marker_topic, 10)
        self.obstacle_pub = self.create_publisher(ObstacleArray, self.obstacles_topic, 10)

        # 프레임 버퍼(2프레임 → 4채널)
        self.frame1 = None  # (1, 2, H, W) : [occupancy, density]
        self.frame2 = None

        # 간단 추적(좌표 라운딩 → ID)
        self.next_id = 1
        self.tracked_objects = {}  # { (round(x,res), round(y,res)) : id }

        # 40Hz 타이머
        self.timer = self.create_timer(1.0 / 40.0, self.process)

    # -------------------- Callbacks --------------------
    def odom_callback(self, msg: Odometry):
        # 현재 버전에서는 좌표 변환에 사용하지 않지만, 유지(확장성/디버깅용)
        self.odom = msg

    def scan_callback(self, msg: LaserScan):
        self.scan_data = msg

    # -------------------- Helpers --------------------
    def preprocess_scan(self, scan_msg: LaserScan):
        """ LiDAR 스캔 → (1, 2, H, W): occupancy(0/1), density(카운트) """
        lidar = np.array(scan_msg.ranges, dtype=np.float32)

        # 각도 배열
        angles = np.linspace(scan_msg.angle_min, scan_msg.angle_max, len(lidar))
        cos_a = np.cos(angles)
        sin_a = np.sin(angles)

        # 전방(+)만 사용 (후방도 쓰려면 fwd 필터 제거)
        x = lidar * cos_a
        y = lidar * sin_a
        fwd = x >= 0
        x = x[fwd]; y = y[fwd]

        # 그리드 인덱스
        xi = ((x + self.origin_offset) / self.pixelsize).astype(int)
        yi = ((y + self.origin_offset) / self.pixelsize).astype(int)
        H = W = self.image_size
        valid = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
        xi = xi[valid]; yi = yi[valid]

        img = np.zeros((1, 2, H, W), dtype=np.float32)
        if xi.size > 0:
            img[0, 0, yi, xi] = 1.0           # occupancy
            np.add.at(img[0, 1], (yi, xi), 1) # density += 1
        return img

    def index_to_cartesian(self, x_img, y_img):
        """ 픽셀 인덱스 → LiDAR(=laser) 좌표계 x,y [m] """
        x = x_img * self.pixelsize - self.origin_offset
        y = y_img * self.pixelsize - self.origin_offset
        return x, y

    def find_k_peaks(self, image, k, threshold=0.3, radius=0.1):
        """
        image: (H, W) numpy, 0~1
        k개 피크 좌표(픽셀 인덱스) 반환.
        """
        H, W = image.shape
        mask = image.copy()
        # 픽셀 반경(현재 128x128 기준 60은 매우 큼, 필요시 10~20 권장)
        r_pix = 60
        peaks = []

        for _ in range(k):
            flat_idx = np.argmax(mask)
            val = mask.flat[flat_idx]
            if val < threshold:
                break
            y, x = np.unravel_index(flat_idx, (H, W))
            peaks.append((x, y, val))
            # 주변 제거
            y0 = max(0, y - r_pix); y1 = min(H, y + r_pix + 1)
            x0 = max(0, x - r_pix); x1 = min(W, x + r_pix + 1)
            mask[y0:y1, x0:x1] = 0.0

        print(peaks)
        return peaks  # [(x, y, score), ...]

    # -------------------- Main loop --------------------
    def process(self):
        if self.scan_data is None or self.odom is None:
            return

        # 2프레임 버퍼 채우기
        if self.frame1 is None:
            self.frame1 = self.preprocess_scan(self.scan_data)
            return
        if self.frame2 is None:
            self.frame2 = self.preprocess_scan(self.scan_data)
            return

        # 최신 프레임 교체
        self.frame1 = self.frame2
        self.frame2 = self.preprocess_scan(self.scan_data)

        # (1, 4, H, W) 입력 구성
        inp = np.concatenate([self.frame1, self.frame2], axis=1)  # (1, 4, H, W)
        inp_t = torch.from_numpy(inp).float().to(self.device)

        # # ---------- 추론 ----------
        # t0 = time.time()
        # with torch.no_grad():
        #     out = self.net(inp_t)  # 기대: [1, 4, H, W] (0: heatmap logit, 1: vx, 2: vy, 3: yaw)
        #     if isinstance(out, tuple):
        #         out = out[0]
        # inf_ms = (time.time() - t0) * 1000.0
        # ---------- 추론 ----------
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = self.net(inp_t)
            if isinstance(out, tuple):
                out = out[0]
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        inf_ms = (time.perf_counter() - t0) * 1000.0

        ## GPU 메모리 정보
        if self.device.type == 'cuda':
            mem_mb     = torch.cuda.memory_allocated() / 1e6
            max_mem_mb = torch.cuda.max_memory_allocated() / 1e6
            self.get_logger().info(f"GPU mem {mem_mb:.1f}MB (max {max_mem_mb:.1f}MB)")


        # heatmap 확률화
        heat_prob = torch.sigmoid(out[0, 0]).cpu().numpy()  # (H, W)

        # vx, vy, yaw 채널 (그대로 실수값 가정)
        vx_map = out[0, 1].cpu().numpy()  # (H, W)
        vy_map = out[0, 2].cpu().numpy()
        yaw_map = out[0, 3].cpu().numpy()

        # ---------- 피크 탐색 ----------
        peaks = self.find_k_peaks(
            image=heat_prob,
            k=self.num_opponents,
            threshold=self.detection_threshold,
            radius=8
        )  # [(x_pix, y_pix, score), ...]

        # ---------- 결과 구축: MarkerArray + ObstacleArray ----------
        marker_array = None
        if self.publish_markers:
            marker_array = MarkerArray()
            # 이전 마커 삭제
            clear = Marker()
            clear.action = Marker.DELETEALL
            marker_array.markers.append(clear)

        # ObstacleArray 메시지 (laser 프레임으로 퍼블리시)
        obst_msg = ObstacleArray()
        obst_msg.header = Header()
        obst_msg.header.stamp = self.get_clock().now().to_msg()
        obst_msg.header.frame_id = self.laser_frame   # 핵심: 레이저 프레임
        obst_msg.obstacles = []

        # 피크마다 LiDAR 좌표 및 (vx,vy,yaw) 샘플링
        for i, (x_pix, y_pix, score) in enumerate(peaks):
            # 픽셀 → LiDAR(=laser) 좌표(전방 평면)
            x_lidar, y_lidar = self.index_to_cartesian(x_pix, y_pix)

            # vx, vy, yaw 픽셀 샘플(필요 시 bilinear 등으로 개선 가능)
            vx  = float(vx_map[y_pix, x_pix])
            vy  = float(vy_map[y_pix, x_pix])
            yaw = float(yaw_map[y_pix, x_pix])

            # ---------- 간단 추적으로 ID 부여 (라이다 좌표 기준) ----------
            key = (round(x_lidar / self.id_resolution_m) * self.id_resolution_m,
                   round(y_lidar / self.id_resolution_m) * self.id_resolution_m)
            if key not in self.tracked_objects:
                self.tracked_objects[key] = self.next_id
                self.next_id += 1
            obj_id = self.tracked_objects[key]

            # ObstacleWpnt 작성 (라이다 좌표 그대로)
            ob = ObstacleWpnt()
            ob.id  = obj_id
            ob.x   = float(x_lidar)
            ob.y   = float(y_lidar)
            ob.vx  = vx
            ob.vy  = vy
            ob.yaw = yaw
            ob.size = 0.5
            obst_msg.obstacles.append(ob)

            # RViz Marker (laser 프레임으로 그대로 표시)
            if self.publish_markers:
                marker = Marker()
                marker.header.frame_id = self.laser_frame
                marker.header.stamp    = obst_msg.header.stamp
                marker.ns   = "Real_detector_ObstacleArray"
                marker.id   = obj_id   # ID와 동기화
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                marker.pose.position.x = x_lidar
                marker.pose.position.y = y_lidar
                marker.pose.position.z = 0.1
                marker.scale.x = 0.4
                marker.scale.y = 0.4
                marker.scale.z = 0.4
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
                marker.color.a = 1.0
                marker_array.markers.append(marker)

        # 퍼블리시
        self.obstacle_pub.publish(obst_msg)
        self.get_logger().info(f"Inference {inf_ms:.1f}ms, Published {len(obst_msg.obstacles)} obstacles (laser frame).")

        if self.publish_markers and marker_array is not None:
            self.marker_pub.publish(marker_array)

    # --------------------------------------------------

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
