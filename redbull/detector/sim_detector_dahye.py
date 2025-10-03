#!/usr/bin/env python3
"""
SimDetector (with stable ID tracking)
- LiDAR(/scan), Odom(/ego_racecar/odom) → TinyCenterSpeedDense 추론
- heatmap 피크 추출 → LiDAR→map 변환
- 최근접 매칭 + TTL + min_hits + EMA 로 ID 안정화(정적 가정)
- redbull_msgs/ObstacleArray 퍼블리시, RViz Marker 표출

※ LiDAR가 base_link에서 +x로 laser_distance_from_base_link(m)만큼 앞에 있음(기본 0.275m).
"""

import os
import sys
import time
import math
import numpy as np
from collections import deque

import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
from redbull_msgs.msg import ObstacleArray, ObstacleWpnt
import pynvml
import torch

# -------------------- Tracking utils --------------------
def euclidean(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

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

# -------------------- Main Node --------------------
class SimDetector(Node):
    def __init__(self):
        super().__init__('sim_detector')

        # -------------------- Parameters --------------------
        # self.declare_parameter('model_path', '/home/harry/ros2_ws/src/TinyCenterSpeed/src/pt/_0super_best_CenterSpeedDense_trainfree41561_20250817_004834_epoch_11_loss_1_63189.pt')  # 0.7
        # self.declare_parameter('model_path', '/home/harry/ros2_ws/src/TinyCenterSpeed/src/pt/0_best_objfree_trainfree52497_20250818_170557_epoch_14_loss_088283.pt')  # 0.8 0.7도 굿
        # self.declare_parameter('model_path', '/home/harry/ros2_ws/src/TinyCenterSpeed/src/pt/1_CenterSpeeedDense_CBAM_20250820_1350_epoch_16_loss_063867.pt')   #0.65
        # self.declare_parameter('model_path', '/home/harry/ros2_ws/src/TinyCenterSpeed/src/pt/1_CenterSpeedBottleneckCBAM20250821_045235_epoch_18.pt')

        self.declare_parameter('model_path', '/home/harry/ros2_ws/src/TinyCenterSpeed/src/pt/1_CenterSpeedBottleneckSENet20250821_072554_epoch_27.pt')         
        # self.declare_parameter('model_path', '/home/harry/ros2_ws/src/TinyCenterSpeed/src/pt/0_CenterSpeedResidual_free52497_20250819_123751_epoch_25_loss_085583.pt')
        # self.declare_parameter('model_path', '/home/harry/ros2_ws/src/TinyCenterSpeed/src/pt/CenterSpeedDenseCBAM_free52497_20250819_215354_epoch_18.pt')  # CBAM 개판인데 ㅅㅂ

        # self.declare_parameter('model_path', '/home/harry/ros2_ws/src/TinyCenterSpeed/src/pt/1_CenterSpeedDenseBottleneck_free52497_20250820_010558_epoch_25_loss_1_88935.pt')

        # self.declare_parameter('model_path', '/home/harry/ros2_ws/src/TinyCenterSpeed/src/pt/CenterSpeedDense_dynamic99_20250820_04%_epoch_45.pt')

        # self.declare_parameter('model_path', '/home/harry/ros2_ws/src/TinyCenterSpeed/src/pt/1_CenterSpeedDenseCBAM_free52497_20250819_220113_epoch_20_loss_1_47034.pt')
        # /home/harry/ros2_ws/src/TinyCenterSpeed/src/pt/1_CenterSpeedDenseCBAM_free52497_20250819_220113_epoch_20_loss_1_47034.pt

        #/home/harry/ros2_ws/src/TinyCenterSpeed/src/pt/1_CenterSpeedDenseCBAM_free52497_20250819_195759_epoch_32_loss_1_74977.pt
        # self.declare_parameter('model_path', '/home/harry/ros2_ws/src/TinyCenterSpeed/src/pt/0_objfree_trainfree52497_20250817_141346_epoch_9_loss_079843.pt')
        
        # /home/harry/ros2_ws/src/TinyCenterSpeed/src/pt/0_CenterSpeedResidual_free52497_20250819_123751_epoch_25_loss_085583.pt
        self.declare_parameter('image_size', 128)
        self.declare_parameter('dense', True)
        self.declare_parameter('num_opponents', 1)
        self.declare_parameter('detection_threshold', 0.7)
        self.declare_parameter('pixelsize', 0.1)
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('laser_frame', 'laser')
        self.declare_parameter('odom_topic', '/ego_racecar/odom')  # /car_state/odom
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('marker_topic', '/sim_detector_objects')
        self.declare_parameter('obstacles_topic', 'obstacles')
        self.declare_parameter('publish_markers', True)

        # ★ LiDAR 오프셋 (base_link에서 +x로 전방 오프셋[m])
        self.declare_parameter('laser_distance_from_base_link', 0.275)

        # ★ 트래커 파라미터
        self.declare_parameter('gate_dist', 0.8)          # 같은 물체로 인정할 최대 거리[m]
        self.declare_parameter('min_hits', 3)             # 확정까지 필요한 연속 관측 수
        self.declare_parameter('max_missed', 3)          # 끊겨도 유지할 프레임 수
        self.declare_parameter('ema_alpha', 0.2)          # 좌표 EMA 계수
        self.declare_parameter('merge_dist', 2)         # 근접 중복 트랙 병합 임계[m]
        self.declare_parameter('publish_unconfirmed', False)

        # Read params
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

        self.laser_offset_x      = float(self.get_parameter('laser_distance_from_base_link').value)  # +x 전방

        self.gate_dist           = float(self.get_parameter('gate_dist').value)
        self.min_hits            = int(self.get_parameter('min_hits').value)
        self.max_missed          = int(self.get_parameter('max_missed').value)
        self.ema_alpha           = float(self.get_parameter('ema_alpha').value)
        self.merge_dist          = float(self.get_parameter('merge_dist').value)
        self.publish_unconfirmed = bool(self.get_parameter('publish_unconfirmed').value)

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
            from CenterSpeed import CenterSpeedDense, CenterSpeedDenseResidual, CenterSpeedDenseCBAM, CenterSpeedDenseBottleneck, CenterSpeedBottleneckCBAM, CenterSpeedBottleneckSENet
        except ImportError as e:
            self.get_logger().error(f"CenterSpeed import 실패: {e}")
            raise
        
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f'Using device: {self.device}')
        self.get_logger().info(f'LiDAR offset from base_link (x): {self.laser_offset_x:.3f} m')

        # self.net = CenterSpeedDenseBottleneck(input_channels=4)

        # self.net = CenterSpeedDenseCBAM(input_channels=4)
        # self.net = CenterSpeedDenseResidual(input_channels=4, image_size=self.image_size)
        # self.net = CenterSpeedDense(input_channels=4, image_size=self.image_size)
        # self.net = CenterSpeedBottleneckCBAM(input_channels=4, image_size=self.image_size)
        self.net = CenterSpeedBottleneckSENet(input_channels=4, image_size=self.image_size)

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
            self.marker_pub = self.create_publisher(MarkerArray, self.marker_topic, 10)
        self.obstacle_pub = self.create_publisher(ObstacleArray, self.obstacles_topic, 10)

        # 프레임 버퍼(2프레임 → 4채널)
        self.frame1 = None  # (1, 2, H, W) : [occupancy, density]
        self.frame2 = None

        # ★ 트래커 상태
        self.next_id = 1
        self.tracks = []  # list[Track]

        # 40Hz 타이머
        # self.timer = self.create_timer(1.0 / 40.0, self.process)

    # -------------------- Callbacks --------------------
    def odom_callback(self, msg: Odometry):
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

    def find_k_peaks(self, image, k, threshold=0.3, radius=0.1):
        H, W = image.shape
        mask = image.copy()
        r_pix = max(1, int(radius / self.pixelsize)) if radius != 0.1 else 8
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

    def lidar_to_map(self, x_lidar, y_lidar, odom: Odometry):
        """
        LiDAR(=laser frame) 좌표 -> map 좌표.
        가정: laser 프레임은 base_link와 평행(회전 차이 0), base_link에서 +x로 self.laser_offset_x 만큼 전방.
        수식: p_bl = [x_lidar + dx, y_lidar]
              p_map = R(yaw) * p_bl + [px, py]
        """
        px = odom.pose.pose.position.x
        py = odom.pose.pose.position.y
        q = odom.pose.pose.orientation

        # yaw 추출
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y*q.y + q.z*q.z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        # laser -> base_link: +x로 dx 전방 오프셋 반영
        x_bl = x_lidar + self.laser_offset_x
        y_bl = y_lidar

        # base_link -> map
        cy, sy = np.cos(yaw), np.sin(yaw)
        x_map = cy * x_bl - sy * y_bl + px
        y_map = sy * x_bl + cy * y_bl + py
        return x_map, y_map, yaw

    # -------------------- Tracking core --------------------
    def _ema(self, old, new):
        return (1.0 - self.ema_alpha) * old + self.ema_alpha * new

    def associate(self, tracks, meas):
        """Greedy 최근접 매칭(1:1) + 거리 게이팅"""
        if not tracks or not meas:
            return [], list(range(len(tracks))), list(range(len(meas)))
        D = np.zeros((len(tracks), len(meas)), dtype=np.float32)
        for i, t in enumerate(tracks):
            for j, m in enumerate(meas):
                D[i, j] = euclidean((t.x, t.y), (m[0], m[1]))
        pairs = [(i, j, D[i, j]) for i in range(len(tracks)) for j in range(len(meas))]
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
        unmatched_meas   = [j for j in range(len(meas)) if j not in taken_m]
        return matches, unmatched_tracks, unmatched_meas

    def merge_duplicates(self):
        """서로 너무 가까운 트랙 병합(오래된/확정된 트랙 우선 유지)"""
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

    # -------------------- Main loop --------------------
    def process(self):
        if self.scan_data is None or self.odom is None:
            return

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
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)

        # 추론 전 메모리/Util
        mem_before = pynvml.nvmlDeviceGetMemoryInfo(handle).used
        util_before = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu

        with torch.no_grad():
            torch.cuda.synchronize()
            t0 = time.time()
            out = self.net(inp_t)
            torch.cuda.synchronize()
            t1 = time.time()
            infer_ms = (t1 - t0) * 1000.0

        mem_after = pynvml.nvmlDeviceGetMemoryInfo(handle).used
        util_after = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu

        # print(f"[SimDetector] 추론시간 {infer_ms:.2f} ms | "
        #     f"메모리 사용 Δ {(mem_after - mem_before)/1024**2:.5f} MB "
        #     f"(총 {mem_after/1024**2:.1f} MB) | "
        #     f"GPU Util before {util_before}% → after {util_after}%")


        heat_prob = torch.sigmoid(out[0, 0]).cpu().numpy()
        vx_map = out[0, 1].cpu().numpy()
        vy_map = out[0, 2].cpu().numpy()
        yaw_map = out[0, 3].cpu().numpy()

        # ---------- 피크 탐색 ----------
        peaks = self.find_k_peaks(
            image=heat_prob,
            k=self.num_opponents,
            threshold=self.detection_threshold,
            radius=8 * self.pixelsize  # 대략 8픽셀
        )

        # ---------- 측정(Meas) 구성 (map 좌표) ----------
        meas = []  # list[(x, y, yaw, vx, vy)]
        for (x_pix, y_pix, score) in peaks:
            x_lidar, y_lidar = self.index_to_cartesian(x_pix, y_pix)
            x_map, y_map, ego_yaw = self.lidar_to_map(x_lidar, y_lidar, self.odom)
            vx = float(vx_map[y_pix, x_pix])
            vy = float(vy_map[y_pix, x_pix])
            yaw = float(yaw_map[y_pix, x_pix])
            meas.append((x_map, y_map, yaw, vx, vy))

        # ---------- 연관(associate) ----------
        matches, unmatched_tracks_idx, unmatched_meas_idx = self.associate(self.tracks, meas)

        # 1) 매칭된 트랙 업데이트 (EMA, 정적 가정)
        for ti, mi in matches:
            t = self.tracks[ti]
            mx, my, myaw, mvx, mvy = meas[mi]
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

        # 2) 미매칭 트랙은 미스 증가 → TTL 초과 시 제거 후보
        survivors = []
        for idx in unmatched_tracks_idx:
            t = self.tracks[idx]
            t.missed += 1
            if t.missed <= self.max_missed:
                survivors.append(t)

        matched_idxs = {ti for ti, _ in matches}
        self.tracks = [self.tracks[i] for i in range(len(self.tracks))
                       if (i in matched_idxs) or (self.tracks[i] in survivors)]

        # 3) 미매칭 측정으로 새 트랙 생성
        for mi in unmatched_meas_idx:
            mx, my, myaw, mvx, mvy = meas[mi]
            t = Track(self.next_id, mx, my, myaw, 0.0, 0.0)
            self.next_id += 1
            self.tracks.append(t)

        # 4) 중복 트랙 병합(옵션)
        self.merge_duplicates()

        # ---------- Publish: ObstacleArray ----------
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
            ob.yaw = float(t.yaw)
            ob.vx  = 0.0
            ob.vy  = 0.0
            ob.size = 0.5
            obst_msg.obstacles.append(ob)

        self.obstacle_pub.publish(obst_msg)

        # ---------- RViz Markers ----------
        if self.publish_markers:
            marker_array = MarkerArray()
            clear = Marker()
            clear.action = Marker.DELETEALL
            marker_array.markers.append(clear)

            for t in publishables:
                m = Marker()
                m.header.frame_id = self.map_frame
                m.header.stamp    = obst_msg.header.stamp
                m.ns   = "sim_detector"
                m.id   = int(t.id)
                m.type = Marker.SPHERE
                m.action = Marker.ADD
                m.pose.position.x = float(t.x)
                m.pose.position.y = float(t.y)
                m.pose.position.z = 0.1
                m.scale.x = 0.4; m.scale.y = 0.4; m.scale.z = 0.4
                m.color.r = 1.0; m.color.g = 0.0; m.color.b = 0.0; m.color.a = 1.0
                marker_array.markers.append(m)

                # (선택) ID 텍스트
                txt = Marker()
                txt.header.frame_id = self.map_frame
                txt.header.stamp    = obst_msg.header.stamp
                txt.ns = "sim_detector_text"
                txt.id = 100000 + int(t.id)
                txt.type = Marker.TEXT_VIEW_FACING
                txt.action = Marker.ADD
                txt.pose.position.x = float(t.x)
                txt.pose.position.y = float(t.y)
                txt.pose.position.z = 0.7
                txt.scale.z = 0.5
                txt.color.r = 1.0; txt.color.g = 1.0; txt.color.b = 1.0; txt.color.a = 1.0
                txt.text = f"ID {t.id}"
                marker_array.markers.append(txt)

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


# #!/usr/bin/env python3
# """
# SimDetector (with stable ID tracking)
# - LiDAR(/scan), Odom(/ego_racecar/odom) → TinyCenterSpeedDense 추론
# - heatmap 피크 추출 → LiDAR→map 변환
# - 최근접 매칭 + TTL + min_hits + EMA 로 ID 안정화(정적 가정)
# - redbull_msgs/ObstacleArray 퍼블리시, RViz Marker 표출
# """

# import os
# import sys
# import time
# import math
# import numpy as np
# from collections import deque

# import rclpy
# from rclpy.node import Node
# from std_msgs.msg import Header
# from sensor_msgs.msg import LaserScan
# from nav_msgs.msg import Odometry
# from visualization_msgs.msg import Marker, MarkerArray
# from redbull_msgs.msg import ObstacleArray, ObstacleWpnt

# import torch

# # -------------------- Tracking utils --------------------
# def euclidean(a, b):
#     return math.hypot(a[0] - b[0], a[1] - b[1])

# class Track:
#     __slots__ = ("id", "x", "y", "yaw", "vx", "vy",
#                  "hits", "missed", "confirmed", "history")
#     def __init__(self, tid, x, y, yaw=0.0, vx=0.0, vy=0.0):
#         self.id = tid
#         self.x = x; self.y = y
#         self.yaw = yaw
#         self.vx = vx; self.vy = vy
#         self.hits = 1
#         self.missed = 0
#         self.confirmed = False
#         self.history = deque(maxlen=5)

# # -------------------- Main Node --------------------
# class SimDetector(Node):
#     def __init__(self):
#         super().__init__('sim_detector')

#         # -------------------- Parameters --------------------
#         # self.declare_parameter('model_path', '/home/harry/ros2_ws/src/TinyCenterSpeed/src/pt/0_best_objfree_trainfree41561_20250817_004834_epoch_11_loss_1_63189.pt')   #이건 잘 됨 sim에서 0.8로 
#         self.declare_parameter('model_path', '/home/harry/ros2_ws/src/TinyCenterSpeed/src/pt/1_objfree_trainfree5249720250818_170557_epoch_14_loss_088283.pt')
#         # /home/harry/ros2_ws/src/TinyCenterSpeed/src/pt/1_objfree_trainfree5249720250818_170557_epoch_14_loss_088283.pt
#         # self.declare_parameter('model_path', '/home/harry/ros2_ws/src/TinyCenterSpeed/src/pt/0_objfree_trainfree52497_20250817_141346_epoch_9_loss_079843.pt')
#         # /home/harry/ros2_ws/src/TinyCenterSpeed/src/pt/0_objfree_trainfree52497_20250817_141346_epoch_9_loss_079843.pt

#         self.declare_parameter('image_size', 128)
#         self.declare_parameter('dense', True)
#         self.declare_parameter('num_opponents', 1)
#         self.declare_parameter('detection_threshold', 0.8)
#         self.declare_parameter('pixelsize', 0.1)
#         self.declare_parameter('map_frame', 'map')
#         self.declare_parameter('laser_frame', 'laser')  
#         self.declare_parameter('odom_topic', '/ego_racecar/odom')  # /car_state/odom
#         self.declare_parameter('scan_topic', '/scan')
#         self.declare_parameter('marker_topic', '/sim_detector_objects')
#         self.declare_parameter('obstacles_topic', 'obstacles')
#         self.declare_parameter('publish_markers', True)

#         # ★ 트래커 파라미터
#         self.declare_parameter('gate_dist', 0.8)          # 같은 물체로 인정할 최대 거리[m]
#         self.declare_parameter('min_hits', 3)             # 확정까지 필요한 연속 관측 수
#         self.declare_parameter('max_missed', 15)          # 끊겨도 유지할 프레임 수
#         self.declare_parameter('ema_alpha', 0.2)          # 좌표 EMA 계수
#         self.declare_parameter('merge_dist', 1)         # 근접 중복 트랙 병합 임계[m]
#         self.declare_parameter('publish_unconfirmed', False)

#         # Read params
#         self.model_path          = self.get_parameter('model_path').value
#         self.image_size          = int(self.get_parameter('image_size').value)
#         self.dense               = bool(self.get_parameter('dense').value)
#         self.num_opponents       = int(self.get_parameter('num_opponents').value)
#         self.detection_threshold = float(self.get_parameter('detection_threshold').value)
#         self.pixelsize           = float(self.get_parameter('pixelsize').value)
#         self.origin_offset       = (self.image_size // 2) * self.pixelsize
#         self.map_frame           = self.get_parameter('map_frame').value
#         self.laser_frame         = self.get_parameter('laser_frame').value
#         self.odom_topic          = self.get_parameter('odom_topic').value
#         self.scan_topic          = self.get_parameter('scan_topic').value
#         self.marker_topic        = self.get_parameter('marker_topic').value
#         self.obstacles_topic     = self.get_parameter('obstacles_topic').value
#         self.publish_markers     = bool(self.get_parameter('publish_markers').value)

#         self.gate_dist           = float(self.get_parameter('gate_dist').value)
#         self.min_hits            = int(self.get_parameter('min_hits').value)
#         self.max_missed          = int(self.get_parameter('max_missed').value)
#         self.ema_alpha           = float(self.get_parameter('ema_alpha').value)
#         self.merge_dist          = float(self.get_parameter('merge_dist').value)
#         self.publish_unconfirmed = bool(self.get_parameter('publish_unconfirmed').value)

#         # -------------------- Model import --------------------
#         current_dir = os.path.dirname(os.path.abspath(__file__))
#         redbull_root = os.path.dirname(current_dir)
#         for p in [os.path.join(redbull_root, 'train'),
#                   os.path.join(redbull_root, 'train', 'models'),
#                   redbull_root,
#                   os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/models'))]:
#             if os.path.exists(p) and p not in sys.path:
#                 sys.path.insert(0, p)

#         try:
#             from CenterSpeed import CenterSpeedDense
#         except ImportError as e:
#             self.get_logger().error(f"CenterSpeed import 실패: {e}")
#             raise

#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.get_logger().info(f'Using device: {self.device}')

#         self.net = CenterSpeedDense(input_channels=4, image_size=self.image_size)
#         state = torch.load(self.model_path, map_location=self.device)
#         if isinstance(state, dict) and 'state_dict' in state:
#             state = state['state_dict']
#         self.net.load_state_dict(state, strict=False)
#         self.net.eval().to(self.device)

#         # -------------------- ROS I/O --------------------
#         self.odom = None
#         self.scan_data = None
#         self.create_subscription(Odometry,   self.odom_topic, self.odom_callback, 10)
#         self.create_subscription(LaserScan,  self.scan_topic, self.scan_callback, 10)
#         if self.publish_markers:
#             self.marker_pub = self.create_publisher(MarkerArray, self.marker_topic, 10)
#         self.obstacle_pub = self.create_publisher(ObstacleArray, self.obstacles_topic, 10)

#         # 프레임 버퍼(2프레임 → 4채널)
#         self.frame1 = None  # (1, 2, H, W) : [occupancy, density]
#         self.frame2 = None

#         # ★ 트래커 상태
#         self.next_id = 1
#         self.tracks = []  # list[Track]

#         # 40Hz 타이머
#         self.timer = self.create_timer(1.0 / 40.0, self.process)

#     # -------------------- Callbacks --------------------
#     def odom_callback(self, msg: Odometry):
#         self.odom = msg

#     def scan_callback(self, msg: LaserScan):
#         self.scan_data = msg

#     # -------------------- Helpers --------------------
#     def preprocess_scan(self, scan_msg: LaserScan):
#         """ LiDAR 스캔 → (1, 2, H, W): occupancy(0/1), density(카운트) """
#         lidar = np.array(scan_msg.ranges, dtype=np.float32)
#         angles = np.linspace(scan_msg.angle_min, scan_msg.angle_max, len(lidar))
#         cos_a, sin_a = np.cos(angles), np.sin(angles)

#         # 전방(+)만 사용
#         x = lidar * cos_a
#         y = lidar * sin_a
#         fwd = x >= 0
#         x = x[fwd]; y = y[fwd]

#         xi = ((x + self.origin_offset) / self.pixelsize).astype(int)
#         yi = ((y + self.origin_offset) / self.pixelsize).astype(int)
#         H = W = self.image_size
#         valid = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
#         xi = xi[valid]; yi = yi[valid]

#         img = np.zeros((1, 2, H, W), dtype=np.float32)
#         if xi.size > 0:
#             img[0, 0, yi, xi] = 1.0
#             np.add.at(img[0, 1], (yi, xi), 1)
#         return img

#     def index_to_cartesian(self, x_img, y_img):
#         x = x_img * self.pixelsize - self.origin_offset
#         y = y_img * self.pixelsize - self.origin_offset
#         return x, y

#     def find_k_peaks(self, image, k, threshold=0.3, radius=0.1):
#         H, W = image.shape
#         mask = image.copy()
#         # NOTE: 필요 시 radius→픽셀 변환 사용. 여기선 넉넉히 제거.
#         r_pix = max(1, int(radius / self.pixelsize)) if radius != 0.1 else 8
#         peaks = []
#         for _ in range(k):
#             flat_idx = np.argmax(mask)
#             val = mask.flat[flat_idx]
#             if val < threshold:
#                 break
#             y, x = np.unravel_index(flat_idx, (H, W))
#             peaks.append((x, y, val))
#             y0 = max(0, y - r_pix); y1 = min(H, y + r_pix + 1)
#             x0 = max(0, x - r_pix); x1 = min(W, x + r_pix + 1)
#             mask[y0:y1, x0:x1] = 0.0
#         print("peaks:", peaks)
#         return peaks

#     def lidar_to_map(self, x_lidar, y_lidar, odom: Odometry):
#         px = odom.pose.pose.position.x
#         py = odom.pose.pose.position.y
#         q = odom.pose.pose.orientation
#         # yaw 추출
#         siny_cosp = 2 * (q.w * q.z + q.x * q.y)
#         cosy_cosp = 1 - 2 * (q.y*q.y + q.z*q.z)
#         yaw = np.arctan2(siny_cosp, cosy_cosp)
#         # 회전 + 평행이동
#         x_map = np.cos(yaw) * x_lidar - np.sin(yaw) * y_lidar + px
#         y_map = np.sin(yaw) * x_lidar + np.cos(yaw) * y_lidar + py
#         return x_map, y_map, yaw

#     # -------------------- Tracking core --------------------
#     def _ema(self, old, new):
#         return (1.0 - self.ema_alpha) * old + self.ema_alpha * new

#     def associate(self, tracks, meas):
#         """Greedy 최근접 매칭(1:1) + 거리 게이팅"""
#         if not tracks or not meas:
#             return [], list(range(len(tracks))), list(range(len(meas)))
#         D = np.zeros((len(tracks), len(meas)), dtype=np.float32)
#         for i, t in enumerate(tracks):
#             for j, m in enumerate(meas):
#                 D[i, j] = euclidean((t.x, t.y), (m[0], m[1]))
#         pairs = [(i, j, D[i, j]) for i in range(len(tracks)) for j in range(len(meas))]
#         pairs.sort(key=lambda x: x[2])
#         taken_t, taken_m, matches = set(), set(), []
#         for ti, mi, d in pairs:
#             if d > self.gate_dist:
#                 break
#             if ti in taken_t or mi in taken_m:
#                 continue
#             taken_t.add(ti); taken_m.add(mi)
#             matches.append((ti, mi))
#         unmatched_tracks = [i for i in range(len(tracks)) if i not in taken_t]
#         unmatched_meas   = [j for j in range(len(meas)) if j not in taken_m]
#         return matches, unmatched_tracks, unmatched_meas

#     def merge_duplicates(self):
#         """서로 너무 가까운 트랙 병합(오래된/확정된 트랙 우선 유지)"""
#         if len(self.tracks) < 2:
#             return
#         to_remove = set()
#         for i in range(len(self.tracks)):
#             if i in to_remove: 
#                 continue
#             for j in range(i + 1, len(self.tracks)):
#                 if j in to_remove:
#                     continue
#                 ti, tj = self.tracks[i], self.tracks[j]
#                 if euclidean((ti.x, ti.y), (tj.x, tj.y)) < self.merge_dist:
#                     keep, drop = (ti, tj) if ti.hits >= tj.hits else (tj, ti)
#                     keep.x = self._ema(keep.x, drop.x)
#                     keep.y = self._ema(keep.y, drop.y)
#                     to_remove.add(self.tracks.index(drop))
#         if to_remove:
#             self.tracks = [t for k, t in enumerate(self.tracks) if k not in to_remove]

#     # -------------------- Main loop --------------------
#     def process(self):
#         if self.scan_data is None or self.odom is None:
#             return

#         # 2프레임 버퍼 구축
#         if self.frame1 is None:
#             self.frame1 = self.preprocess_scan(self.scan_data); return
#         if self.frame2 is None:
#             self.frame2 = self.preprocess_scan(self.scan_data); return
#         self.frame1 = self.frame2
#         self.frame2 = self.preprocess_scan(self.scan_data)

#         # 네트 입력
#         inp = np.concatenate([self.frame1, self.frame2], axis=1)  # (1, 4, H, W)
#         inp_t = torch.from_numpy(inp).float().to(self.device)

#         # ---------- 추론 ----------
#         with torch.no_grad():
#             out = self.net(inp_t)  # 기대: [1, 4, H, W] (0: heatmap logit, 1: vx, 2: vy, 3: yaw)
#             if isinstance(out, tuple):
#                 out = out[0]

#         heat_prob = torch.sigmoid(out[0, 0]).cpu().numpy()
#         vx_map = out[0, 1].cpu().numpy()
#         vy_map = out[0, 2].cpu().numpy()
#         yaw_map = out[0, 3].cpu().numpy()

#         # ---------- 피크 탐색 ----------
#         peaks = self.find_k_peaks(
#             image=heat_prob,
#             k=self.num_opponents,
#             threshold=self.detection_threshold,
#             radius=8 * self.pixelsize  # 대략 8픽셀
#         )

#         # ---------- 측정(Meas) 구성 (map 좌표) ----------
#         meas = []  # list[(x, y, yaw, vx, vy)]
#         for (x_pix, y_pix, score) in peaks:
#             x_lidar, y_lidar = self.index_to_cartesian(x_pix, y_pix)
#             x_map, y_map, ego_yaw = self.lidar_to_map(x_lidar, y_lidar, self.odom)
#             vx = float(vx_map[y_pix, x_pix])
#             vy = float(vy_map[y_pix, x_pix])
#             yaw = float(yaw_map[y_pix, x_pix])
#             meas.append((x_map, y_map, yaw, vx, vy))

#         # ---------- 연관(associate) ----------
#         matches, unmatched_tracks_idx, unmatched_meas_idx = self.associate(self.tracks, meas)

#         # 1) 매칭된 트랙 업데이트 (EMA, 정적 가정)
#         for ti, mi in matches:
#             t = self.tracks[ti]
#             mx, my, myaw, mvx, mvy = meas[mi]
#             t.x = self._ema(t.x, mx)
#             t.y = self._ema(t.y, my)
#             t.vx = self._ema(t.vx, 0.0)  # 정적 가정
#             t.vy = self._ema(t.vy, 0.0)
#             t.yaw = self._ema(t.yaw, myaw)
#             t.hits += 1
#             t.missed = 0
#             if not t.confirmed and t.hits >= self.min_hits:
#                 t.confirmed = True
#             t.history.append((t.x, t.y))

#         # 2) 미매칭 트랙은 미스 증가 → TTL 초과 시 제거 후보
#         survivors = []
#         for idx in unmatched_tracks_idx:
#             t = self.tracks[idx]
#             t.missed += 1
#             if t.missed <= self.max_missed:
#                 survivors.append(t)

#         matched_idxs = {ti for ti, _ in matches}
#         self.tracks = [self.tracks[i] for i in range(len(self.tracks))
#                        if (i in matched_idxs) or (self.tracks[i] in survivors)]

#         # 3) 미매칭 측정으로 새 트랙 생성
#         for mi in unmatched_meas_idx:
#             mx, my, myaw, mvx, mvy = meas[mi]
#             t = Track(self.next_id, mx, my, myaw, 0.0, 0.0)
#             self.next_id += 1
#             self.tracks.append(t)

#         # 4) 중복 트랙 병합(옵션)
#         self.merge_duplicates()

#         # ---------- Publish: ObstacleArray ----------
#         obst_msg = ObstacleArray()
#         obst_msg.header = Header()
#         obst_msg.header.stamp = self.get_clock().now().to_msg()
#         obst_msg.header.frame_id = self.map_frame
#         obst_msg.obstacles = []

#         publishables = [t for t in self.tracks if (t.confirmed or self.publish_unconfirmed)]
#         for t in publishables:
#             ob = ObstacleWpnt()
#             ob.id  = int(t.id)
#             ob.x   = float(t.x)
#             ob.y   = float(t.y)
#             ob.yaw = float(t.yaw)
#             ob.vx  = 0.0
#             ob.vy  = 0.0
#             ob.size = 0.5
#             obst_msg.obstacles.append(ob)

#         self.obstacle_pub.publish(obst_msg)

#         # ---------- RViz Markers ----------
#         if self.publish_markers:
#             marker_array = MarkerArray()
#             clear = Marker()
#             clear.action = Marker.DELETEALL
#             marker_array.markers.append(clear)

#             for t in publishables:
#                 m = Marker()
#                 m.header.frame_id = self.map_frame
#                 m.header.stamp    = obst_msg.header.stamp
#                 m.ns   = "sim_detector"
#                 m.id   = int(t.id)
#                 m.type = Marker.SPHERE
#                 m.action = Marker.ADD
#                 m.pose.position.x = float(t.x)
#                 m.pose.position.y = float(t.y)
#                 m.pose.position.z = 0.1
#                 m.scale.x = 0.4; m.scale.y = 0.4; m.scale.z = 0.4
#                 m.color.r = 1.0; m.color.g = 0.0; m.color.b = 0.0; m.color.a = 1.0
#                 marker_array.markers.append(m)

#                 # (선택) ID 텍스트
#                 txt = Marker()
#                 txt.header.frame_id = self.map_frame
#                 txt.header.stamp    = obst_msg.header.stamp
#                 txt.ns = "sim_detector_text"
#                 txt.id = 100000 + int(t.id)
#                 txt.type = Marker.TEXT_VIEW_FACING
#                 txt.action = Marker.ADD
#                 txt.pose.position.x = float(t.x)
#                 txt.pose.position.y = float(t.y)
#                 txt.pose.position.z = 0.7
#                 txt.scale.z = 0.5
#                 txt.color.r = 1.0; txt.color.g = 1.0; txt.color.b = 1.0; txt.color.a = 1.0
#                 txt.text = f"ID {t.id}"
#                 marker_array.markers.append(txt)

#             self.marker_pub.publish(marker_array)

# # --------------------------------------------------
# def main(args=None):
#     rclpy.init(args=args)
#     node = SimDetector()
#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         pass
#     finally:
#         node.destroy_node()
#         rclpy.shutdown()

# if __name__ == '__main__':
#     main()
