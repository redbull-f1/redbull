#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SimDetector (map&laser markers, map-frame obstacles)
- LiDAR(/scan) → TinyCenterSpeedDense 추론
- heatmap 피크 추출 (LiDAR 좌표)
- LiDAR→base_link(오프셋/회전) → map(odom 포즈)로 변환
- Greedy NN + TTL + min_hits + EMA 로 ID 안정화
- /obstacles: redbull_msgs/ObstacleArray (frame_id=map, x/y/vx/vy/yaw 모두 map 기준)
- RViz Markers:
  * /sim_detector_objects_map   (frame_id=map)
  * /sim_detector_objects_laser (frame_id=laser)
"""

import os
import sys
import math
import numpy as np
from collections import deque

import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
# from redbull_msgs.msg import ObstacleArray, ObstacleWpnt
from f110_msgs.msg import ObstacleArray, ObstacleWpnt

import torch


# -------------------- Utils --------------------
def euclidean(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def yaw_from_quat(q):
    # z-yaw만 가정(2D)
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)

def wrap_angle(a):
    """[-pi, pi]"""
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a

def rot2d(x, y, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return c * x - s * y, s * x + c * y


class Track:
    __slots__ = ("id", "x", "y", "yaw", "vx", "vy",
                 "hits", "missed", "confirmed", "history")

    def __init__(self, tid, x, y, yaw=0.0, vx=0.0, vy=0.0):
        self.id = tid
        self.x = x; self.y = y
        self.yaw = yaw
        self.vx = vx; self.vy = vy
        self.hits = 1
        self.missed = 0
        self.confirmed = False
        self.history = deque(maxlen=5)


class SimDetector(Node):
    def __init__(self):
        super().__init__('sim_detector')

        # -------------------- Parameters --------------------
        self.declare_parameter('model_path', '/home/ailab0/Downloads/0_best_objfree_trainfree41561_20250817_004834_epoch_11_loss_1_63189.pt')
        self.declare_parameter('image_size', 128)
        self.declare_parameter('dense', True)
        self.declare_parameter('num_opponents', 1)
        self.declare_parameter('detection_threshold', 0.8)
        self.declare_parameter('pixelsize', 0.1)

        # Frames & Topics
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('laser_frame', 'laser')
        self.declare_parameter('odom_topic', '/car_state/odom')  # base_link pose in map
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('marker_topic_map', '/objects_map')
        self.declare_parameter('marker_topic_laser', '/objects_laser')
        self.declare_parameter('obstacles_topic', '/obstacles')
        self.declare_parameter('publish_markers', True)

        # Tracker params
        self.declare_parameter('gate_dist', 1.3)  # match gate [m]
        self.declare_parameter('min_hits', 3)     # confirmed after N hits
        self.declare_parameter('max_missed', 15)
        self.declare_parameter('ema_alpha', 0.2)
        self.declare_parameter('merge_dist', 1.0)
        self.declare_parameter('publish_unconfirmed', False)

        # LiDAR offset/rotation w.r.t base_link
        self.declare_parameter('lidar_offset_x', 0.287)  # [m] forward(+x)
        self.declare_parameter('lidar_offset_y', 0.0)    # [m] left(+y)
        self.declare_parameter('lidar_yaw', 0.0)         # [rad] LiDAR yaw vs base_link

        # -------------------- Read params --------------------
        self.model_path          = self.get_parameter('model_path').value
        self.image_size          = int(self.get_parameter('image_size').value)
        self.dense               = bool(self.get_parameter('dense').value)
        self.num_opponents       = int(self.get_parameter('num_opponents').value)
        self.detection_threshold = float(self.get_parameter('detection_threshold').value)
        self.pixelsize           = float(self.get_parameter('pixelsize').value)

        self.map_frame           = self.get_parameter('map_frame').value
        self.laser_frame         = self.get_parameter('laser_frame').value
        self.odom_topic          = self.get_parameter('odom_topic').value
        self.scan_topic          = self.get_parameter('scan_topic').value
        self.marker_topic_map    = self.get_parameter('marker_topic_map').value
        self.marker_topic_laser  = self.get_parameter('marker_topic_laser').value
        self.obstacles_topic     = self.get_parameter('obstacles_topic').value
        self.publish_markers     = bool(self.get_parameter('publish_markers').value)

        self.gate_dist           = float(self.get_parameter('gate_dist').value)
        self.min_hits            = int(self.get_parameter('min_hits').value)
        self.max_missed          = int(self.get_parameter('max_missed').value)
        self.ema_alpha           = float(self.get_parameter('ema_alpha').value)
        self.merge_dist          = float(self.get_parameter('merge_dist').value)
        self.publish_unconfirmed = bool(self.get_parameter('publish_unconfirmed').value)

        self.lidar_offset_x      = float(self.get_parameter('lidar_offset_x').value)
        self.lidar_offset_y      = float(self.get_parameter('lidar_offset_y').value)
        self.lidar_yaw           = float(self.get_parameter('lidar_yaw').value)

        self.origin_offset       = (self.image_size // 2) * self.pixelsize

        # -------------------- Model import --------------------
        current_dir = os.path.dirname(os.path.abspath(__file__))
        redbull_root = os.path.dirname(current_dir)
        for p in [os.path.join(redbull_root, 'train'),
                  os.path.join(redbull_root, 'train', 'models'),
                  redbull_root,
                  os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/models'))]:
            if os.path.exists(p) and p not in sys.path:
                sys.path.insert(0, p)

        try:
            from CenterSpeed import CenterSpeedDense
        except ImportError as e:
            self.get_logger().error(f"CenterSpeed import 실패: {e}")
            raise

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f'Using device: {self.device}')

        self.net = CenterSpeedDense(input_channels=4, image_size=self.image_size)
        state = torch.load(self.model_path, map_location=self.device)
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
            self.marker_pub_map   = self.create_publisher(MarkerArray, self.marker_topic_map, 10)
            self.marker_pub_laser = self.create_publisher(MarkerArray, self.marker_topic_laser, 10)

        self.obstacle_pub = self.create_publisher(ObstacleArray, self.obstacles_topic, 10)

        # Frame buffer (2 frames → 4ch)
        self.frame1 = None  # (1, 2, H, W) : [occupancy, density]
        self.frame2 = None

        # Tracks (map frame 기준 추적)
        self.next_id = 1
        self.tracks = []  # list[Track]

    # -------------------- Callbacks --------------------
    def odom_callback(self, msg: Odometry):
        # 기대: msg.header.frame_id == map, child_frame_id == base_link (혹은 빈 문자열)
        self.odom = msg

    def scan_callback(self, msg: LaserScan):
        self.scan_data = msg
        self.process()

    # -------------------- Helpers --------------------
    def preprocess_scan(self, scan_msg: LaserScan):
        """ LiDAR 스캔 → (1, 2, H, W): occupancy(0/1), density(카운트) """
        lidar = np.array(scan_msg.ranges, dtype=np.float32)
        angles = np.linspace(scan_msg.angle_min, scan_msg.angle_max, len(lidar))
        cos_a, sin_a = np.cos(angles), np.sin(angles)

        # 전방(+)만 사용
        x = lidar * cos_a
        y = lidar * sin_a
        fwd = x >= 0
        x = x[fwd]; y = y[fwd]

        xi = ((x + self.origin_offset) / self.pixelsize).astype(int)
        yi = ((y + self.origin_offset) / self.pixelsize).astype(int)
        H = W = self.image_size
        valid = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
        xi = xi[valid]; yi = yi[valid]

        img = np.zeros((1, 2, H, W), dtype=np.float32)
        if xi.size > 0:
            img[0, 0, yi, xi] = 1.0
            np.add.at(img[0, 1], (yi, xi), 1)
        return img

    def index_to_cartesian(self, x_img, y_img):
        x = x_img * self.pixelsize - self.origin_offset
        y = y_img * self.pixelsize - self.origin_offset
        return x, y

    def find_k_peaks(self, image, k, threshold=0.3, radius=0.8):
        """ NMS 유사 억제(radius[m]) """
        H, W = image.shape
        mask = image.copy()
        r_pix = max(1, int(radius / self.pixelsize))
        peaks = []
        for _ in range(k):
            flat_idx = np.argmax(mask)
            val = mask.flat[flat_idx]
            if val < threshold:
                break
            y, x = np.unravel_index(flat_idx, (H, W))
            peaks.append((x, y, val))
            y0 = max(0, y - r_pix); y1 = min(H, y + r_pix + 1)
            x0 = max(0, x - r_pix); x1 = min(W, x + r_pix + 1)
            mask[y0:y1, x0:x1] = 0.0
        print("peaks:", peaks)
        return peaks

    def _ema(self, old, new):
        return (1.0 - self.ema_alpha) * old + self.ema_alpha * new

    def associate(self, tracks, meas_map):
        """Greedy 최근접 매칭(1:1) + 거리 게이팅 (map frame)"""
        if not tracks or not meas_map:
            return [], list(range(len(tracks))), list(range(len(meas_map)))
        D = np.zeros((len(tracks), len(meas_map)), dtype=np.float32)
        for i, t in enumerate(tracks):
            for j, m in enumerate(meas_map):
                D[i, j] = euclidean((t.x, t.y), (m[0], m[1]))
        pairs = [(i, j, D[i, j]) for i in range(len(tracks)) for j in range(len(meas_map))]
        pairs.sort(key=lambda x: x[2])
        taken_t, taken_m, matches = set(), set(), []
        for ti, mi, d in pairs:
            if d > self.gate_dist:
                break
            if ti in taken_t or mi in taken_m:
                continue
            taken_t.add(ti); taken_m.add(mi)
            matches.append((ti, mi))
        unmatched_tracks = [i for i in range(len(tracks)) if i not in taken_t]
        unmatched_meas   = [j for j in range(len(meas_map)) if j not in taken_m]
        return matches, unmatched_tracks, unmatched_meas

    def merge_duplicates(self):
        """서로 너무 가까운 트랙 병합"""
        if len(self.tracks) < 2:
            return
        to_remove = set()
        for i in range(len(self.tracks)):
            if i in to_remove:
                continue
            for j in range(i + 1, len(self.tracks)):
                if j in to_remove:
                    continue
                ti, tj = self.tracks[i], self.tracks[j]
                if euclidean((ti.x, ti.y), (tj.x, tj.y)) < self.merge_dist:
                    keep, drop = (ti, tj) if ti.hits >= tj.hits else (tj, ti)
                    keep.x = self._ema(keep.x, drop.x)
                    keep.y = self._ema(keep.y, drop.y)
                    to_remove.add(self.tracks.index(drop))
        if to_remove:
            self.tracks = [t for k, t in enumerate(self.tracks) if k not in to_remove]

    # -------------------- Frame transforms (LiDAR<->base_link<->map) --------------------
    def lidar_to_map(self, x_lidar, y_lidar, odom: Odometry):
        """LiDAR→base_link(오프셋/회전) → map(odom 포즈)"""
        px = odom.pose.pose.position.x
        py = odom.pose.pose.position.y
        yaw_ego = yaw_from_quat(odom.pose.pose.orientation)

        # LiDAR -> base_link
        x_bl, y_bl = rot2d(x_lidar, y_lidar, self.lidar_yaw)
        x_bl += self.lidar_offset_x
        y_bl += self.lidar_offset_y

        # base_link -> map
        x_map, y_map = rot2d(x_bl, y_bl, yaw_ego)
        x_map += px; y_map += py
        return x_map, y_map, yaw_ego

    def lidar_vec_to_map(self, vx_lidar, vy_lidar, odom: Odometry):
        """속도 벡터 LiDAR→map 회전만 (평행이동 없음)"""
        yaw_ego = yaw_from_quat(odom.pose.pose.orientation)
        # LiDAR -> base_link
        vx_bl, vy_bl = rot2d(vx_lidar, vy_lidar, self.lidar_yaw)
        # base_link -> map
        vx_map, vy_map = rot2d(vx_bl, vy_bl, yaw_ego)
        return vx_map, vy_map, yaw_ego

    def map_to_lidar(self, x_map, y_map, odom: Odometry):
        """map 좌표의 점을 LiDAR 프레임으로(마커용 역변환)"""
        px = odom.pose.pose.position.x
        py = odom.pose.pose.position.y
        yaw_ego = yaw_from_quat(odom.pose.pose.orientation)

        # map -> base_link
        x_bl, y_bl = rot2d(x_map - px, y_map - py, -yaw_ego)
        # base_link -> LiDAR
        x_lidar, y_lidar = rot2d(x_bl - self.lidar_offset_x, y_bl - self.lidar_offset_y, -self.lidar_yaw)
        return x_lidar, y_lidar

    # -------------------- Main loop --------------------
    def process(self):
        # if self.scan_data is None or self.odom is None:
        #     return

        # 2프레임 버퍼 구축
        if self.frame1 is None:
            self.frame1 = self.preprocess_scan(self.scan_data); return
        if self.frame2 is None:
            self.frame2 = self.preprocess_scan(self.scan_data); return
        self.frame1 = self.frame2
        self.frame2 = self.preprocess_scan(self.scan_data)

        # 네트 입력
        inp = np.concatenate([self.frame1, self.frame2], axis=1)  # (1, 4, H, W)
        inp_t = torch.from_numpy(inp).float().to(self.device)

        # ---------- 추론 ----------
        with torch.no_grad():
            out = self.net(inp_t)  # [1,4,H,W] : heat_logit, vx, vy, yaw (LiDAR frame)
            if isinstance(out, tuple):
                out = out[0]

        heat_prob = torch.sigmoid(out[0, 0]).cpu().numpy()
        vx_l_grid = out[0, 1].cpu().numpy()
        vy_l_grid = out[0, 2].cpu().numpy()
        yaw_l_grid = out[0, 3].cpu().numpy()   # LiDAR frame yaw(가정)

        # ---------- 피크 탐색 ----------
        peaks = self.find_k_peaks(
            image=heat_prob,
            k=self.num_opponents,
            threshold=self.detection_threshold,
            radius=8 * self.pixelsize
        )

        # ---------- 측정 구성 ----------
        # 두 좌표 모두 계산: LiDAR 측정, map 변환 측정
        meas_lidar = []   # (x_l, y_l, yaw_l, vx_l, vy_l)
        meas_map   = []   # (x_m, y_m, yaw_m, vx_m, vy_m)
        for (x_pix, y_pix, score) in peaks:
            x_l, y_l = self.index_to_cartesian(x_pix, y_pix)
            vx_l = float(vx_l_grid[y_pix, x_pix])
            vy_l = float(vy_l_grid[y_pix, x_pix])
            yaw_l = float(yaw_l_grid[y_pix, x_pix])

            # LiDAR→map 변환
            x_m, y_m, ego_yaw = self.lidar_to_map(x_l, y_l, self.odom)
            vx_m, vy_m, _     = self.lidar_vec_to_map(vx_l, vy_l, self.odom)
            yaw_m             = wrap_angle(yaw_l + self.lidar_yaw + ego_yaw)

            meas_lidar.append((x_l, y_l, yaw_l, vx_l, vy_l))
            meas_map.append((x_m, y_m, yaw_m, vx_m, vy_m))

        # ---------- 연관(associate) (map frame) ----------
        matches, unmatched_tracks_idx, unmatched_meas_idx = self.associate(self.tracks, meas_map)

        # 1) 매칭된 트랙 업데이트 (EMA, 정적 가정: vx,vy 0으로 수렴)
        for ti, mi in matches:
            t = self.tracks[ti]
            mx, my, myaw, mvx, mvy = meas_map[mi]
            t.x = self._ema(t.x, mx)
            t.y = self._ema(t.y, my)
            t.vx = self._ema(t.vx, 0.0)  # 정적 가정
            t.vy = self._ema(t.vy, 0.0)
            t.yaw = self._ema(t.yaw, myaw)
            t.hits += 1
            t.missed = 0
            if not t.confirmed and t.hits >= self.min_hits:
                t.confirmed = True
            t.history.append((t.x, t.y))

        # 2) 미매칭 트랙 유지/제거
        survivors = []
        for idx in unmatched_tracks_idx:
            t = self.tracks[idx]
            t.missed += 1
            if t.missed <= self.max_missed:
                survivors.append(t)

        matched_idxs = {ti for ti, _ in matches}
        self.tracks = [self.tracks[i] for i in range(len(self.tracks))
                       if (i in matched_idxs) or (self.tracks[i] in survivors)]

        # 3) 미매칭 측정으로 새 트랙 생성 (map frame)
        for mi in unmatched_meas_idx:
            mx, my, myaw, mvx, mvy = meas_map[mi]
            t = Track(self.next_id, mx, my, myaw, 0.0, 0.0)
            self.next_id += 1
            if self.min_hits <= 1:
                t.confirmed = True
            self.tracks.append(t)

        # 4) 중복 트랙 병합
        self.merge_duplicates()

        # ---------- Publish: ObstacleArray (map frame) ----------
        obst_msg = ObstacleArray()
        obst_msg.header = Header()
        obst_msg.header.stamp = self.get_clock().now().to_msg()
        obst_msg.header.frame_id = self.map_frame
        obst_msg.obstacles = []

        publishables = [t for t in self.tracks if (t.confirmed or self.publish_unconfirmed)]
        for t in publishables:
            ob = ObstacleWpnt()
            ob.id  = int(t.id)
            ob.x   = float(t.x)
            ob.y   = float(t.y)
            # 정적 가정: vx,vy 0.0 (필요하면 추정치 사용 가능)
            ob.vx  = 0.0
            ob.vy  = 0.0
            ob.yaw = float(wrap_angle(t.yaw))
            ob.size = 0.5
            obst_msg.obstacles.append(ob)

        self.obstacle_pub.publish(obst_msg)

        # ---------- RViz Markers ----------
        if self.publish_markers:
            # (a) map frame markers
            ma_map = MarkerArray()
            clear_map = Marker()
            clear_map.header.frame_id = self.map_frame
            clear_map.header.stamp    = obst_msg.header.stamp
            clear_map.action = Marker.DELETEALL
            ma_map.markers.append(clear_map)

            for t in publishables:
                m = Marker()
                m.header.frame_id = self.map_frame
                m.header.stamp    = obst_msg.header.stamp
                m.ns   = "sim_detector_map"
                m.id   = int(t.id)
                m.type = Marker.SPHERE
                m.action = Marker.ADD
                m.pose.position.x = float(t.x)
                m.pose.position.y = float(t.y)
                m.pose.position.z = 0.1
                m.scale.x = 0.4; m.scale.y = 0.4; m.scale.z = 0.4
                m.color.r = 1.0; m.color.g = 0.0; m.color.b = 0.0; m.color.a = 1.0
                ma_map.markers.append(m)

                txt = Marker()
                txt.header.frame_id = self.map_frame
                txt.header.stamp    = obst_msg.header.stamp
                txt.ns = "sim_detector_text_map"
                txt.id = 100000 + int(t.id)
                txt.type = Marker.TEXT_VIEW_FACING
                txt.action = Marker.ADD
                txt.pose.position.x = float(t.x)
                txt.pose.position.y = float(t.y)
                txt.pose.position.z = 0.7
                txt.scale.z = 0.45
                txt.color.r = 1.0; txt.color.g = 1.0; txt.color.b = 1.0; txt.color.a = 1.0
                txt.text = f"ID {t.id}"
                ma_map.markers.append(txt)

            self.marker_pub_map.publish(ma_map)

            # (b) laser frame markers (map→laser 역변환으로 동일 트랙 위치를 laser에 그리기)
            ma_laser = MarkerArray()
            clear_l = Marker()
            clear_l.header.frame_id = self.laser_frame
            clear_l.header.stamp    = obst_msg.header.stamp
            clear_l.action = Marker.DELETEALL
            ma_laser.markers.append(clear_l)

            for t in publishables:
                x_l, y_l = self.map_to_lidar(t.x, t.y, self.odom)
                m = Marker()
                m.header.frame_id = self.laser_frame
                m.header.stamp    = obst_msg.header.stamp
                m.ns   = "sim_detector_laser"
                m.id   = int(t.id)
                m.type = Marker.SPHERE
                m.action = Marker.ADD
                m.pose.position.x = float(x_l)
                m.pose.position.y = float(y_l)
                m.pose.position.z = 0.1
                m.scale.x = 0.4; m.scale.y = 0.4; m.scale.z = 0.4
                m.color.r = 0.0; m.color.g = 0.6; m.color.b = 1.0; m.color.a = 1.0
                ma_laser.markers.append(m)

                txt = Marker()
                txt.header.frame_id = self.laser_frame
                txt.header.stamp    = obst_msg.header.stamp
                txt.ns = "sim_detector_text_laser"
                txt.id = 200000 + int(t.id)
                txt.type = Marker.TEXT_VIEW_FACING
                txt.action = Marker.ADD
                txt.pose.position.x = float(x_l)
                txt.pose.position.y = float(y_l)
                txt.pose.position.z = 0.7
                txt.scale.z = 0.4
                txt.color.r = 1.0; txt.color.g = 1.0; txt.color.b = 1.0; txt.color.a = 1.0
                txt.text = f"ID {t.id}"
                ma_laser.markers.append(txt)

            self.marker_pub_laser.publish(ma_laser)


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