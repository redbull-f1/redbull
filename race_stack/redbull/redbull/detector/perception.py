#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import math
import time
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# ROS 2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header

# Optional: OpenCV for connected components
try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

# --- Paths (relative to this file) ---
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, '..'))  # .../redbull
MODELS_DIR = os.path.join(ROOT_DIR, 'train', 'models')
MODEL_WEIGHTS = os.path.join(ROOT_DIR, 'train', 'trained_models', 'TinyCenterSpeed.pt')

# Ensure model path in sys.path and import
if MODELS_DIR not in sys.path:
    sys.path.append(MODELS_DIR)

ModelClass = None
try:
    from CenterSpeed import TinyCenterSpeedDense as ModelClass
except Exception:
    try:
        from CenterSpeed import CenterSpeedDense as ModelClass
    except Exception as e:
        raise ImportError(f"Failed to import CenterSpeed model from {MODELS_DIR}: {e}")

# Grid parameters (must match training)
GRID_SIZE = 64
PIXEL_SIZE = 0.1  # meters per pixel
GRID_EXTENT = GRID_SIZE * PIXEL_SIZE  # total size in meters
HALF_EXTENT = GRID_EXTENT / 2.0

# Detection parameters
HEATMAP_THRESH = 0.5  # probability threshold
MIN_AREA_PIXELS = 4     # filter small blobs
MARKER_LIFETIME = 0.2   # seconds


def yaw_to_quaternion(yaw: float) -> Tuple[float, float, float, float]:
    half_yaw = yaw * 0.5
    return (0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw))


def world_to_pixel(x: float, y: float) -> Tuple[int, int]:
    # x right, y forward; grid origin at center (col: x, row: y)
    col = int(x / PIXEL_SIZE + GRID_SIZE / 2)
    row = int(y / PIXEL_SIZE + GRID_SIZE / 2)
    return col, row


def pixel_to_world(col: int, row: int) -> Tuple[float, float]:
    x = (col - GRID_SIZE / 2) * PIXEL_SIZE
    y = (row - GRID_SIZE / 2) * PIXEL_SIZE
    return x, y


def scan_to_grid(msg: LaserScan) -> np.ndarray:
    # 3-channel grid: [occupancy, intensity, zeros]
    grid = np.zeros((3, GRID_SIZE, GRID_SIZE), dtype=np.float32)
    angles = msg.angle_min + np.arange(len(msg.ranges)) * msg.angle_increment
    ranges = np.array(msg.ranges, dtype=np.float32)

    # filter valid ranges
    valid = np.isfinite(ranges) & (ranges >= msg.range_min) & (ranges <= msg.range_max)
    angles = angles[valid]
    ranges = ranges[valid]

    xs = ranges * np.cos(angles)
    ys = ranges * np.sin(angles)

    # Project into grid bounds
    mask = (xs >= -HALF_EXTENT) & (xs < HALF_EXTENT) & (ys >= -HALF_EXTENT) & (ys < HALF_EXTENT)
    xs = xs[mask]
    ys = ys[mask]

    cols = (xs / PIXEL_SIZE + GRID_SIZE / 2).astype(np.int32)
    rows = (ys / PIXEL_SIZE + GRID_SIZE / 2).astype(np.int32)

    # clamp
    cols = np.clip(cols, 0, GRID_SIZE - 1)
    rows = np.clip(rows, 0, GRID_SIZE - 1)

    grid[0, rows, cols] = 1.0  # occupancy
    grid[1, rows, cols] = 1.0  # simple intensity placeholder
    # grid[2] left as zeros
    return grid


def extract_obstacles(heatmap: np.ndarray,
                      vx_map: np.ndarray,
                      vy_map: np.ndarray,
                      yaw_map: np.ndarray) -> List[dict]:
    # heatmap expected in [0,1]; threshold then connected components
    bin_map = (heatmap >= HEATMAP_THRESH).astype(np.uint8)

    obstacles = []

    if _HAS_CV2:
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_map, connectivity=8)
        for i in range(1, num_labels):  # skip background 0
            area = stats[i, cv2.CC_STAT_AREA]
            if area < MIN_AREA_PIXELS:
                continue
            cx_f, cy_f = centroids[i]  # (x: col, y: row) float
            col = int(round(cx_f))
            row = int(round(cy_f))
            col = max(0, min(GRID_SIZE - 1, col))
            row = max(0, min(GRID_SIZE - 1, row))

            # estimate radius from area
            # radius_px = math.sqrt(area / math.pi)
            radius_px = 0.5
            radius_m = radius_px * PIXEL_SIZE

            x, y = pixel_to_world(col, row)

            # sample motion/yaw at centroid; optionally average around a small window
            vx = float(vx_map[row, col])
            vy = float(vy_map[row, col])
            yaw = float(yaw_map[row, col])

            obstacles.append({
                'x': x,
                'y': y,
                'vx': vx,
                'vy': vy,
                'yaw': yaw,
                'size': radius_m
            })
    else:
        # Fallback: pick top-K peaks via non-maximum suppression
        K = 10
        hm = heatmap.copy()
        for _ in range(K):
            idx = np.argmax(hm)
            val = hm.flat[idx]
            if val < HEATMAP_THRESH:
                break
            row, col = np.unravel_index(idx, hm.shape)
            # simple fixed radius
            radius_px = 2.0
            radius_m = radius_px * PIXEL_SIZE
            x, y = pixel_to_world(col, row)
            vx = float(vx_map[row, col])
            vy = float(vy_map[row, col])
            yaw = float(yaw_map[row, col])
            obstacles.append({'x': x, 'y': y, 'vx': vx, 'vy': vy, 'yaw': yaw, 'size': radius_m})
            # suppress neighborhood
            r = 3
            r0, r1 = max(0, row - r), min(GRID_SIZE, row + r + 1)
            c0, c1 = max(0, col - r), min(GRID_SIZE, col + r + 1)
            hm[r0:r1, c0:c1] = 0.0

    return obstacles


class TinyCenterSpeedPerception(Node):
    def __init__(self):
        super().__init__('tiny_centerspeed_perception')

        # Device & model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ModelClass(image_size=GRID_SIZE) if 'image_size' in ModelClass.__init__.__code__.co_varnames else ModelClass()
        self.model.to(self.device)
        self.model.eval()

        # Load weights
        state = torch.load(MODEL_WEIGHTS, map_location=self.device)
        self.model.load_state_dict(state)
        self.get_logger().info(f"Loaded model weights from {MODEL_WEIGHTS} on {self.device}")

        # ROS I/O
        self.marker_pub = self.create_publisher(MarkerArray, '/obstacles', 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)

        # Keep last frame for potential temporal stacking (currently unused)
        self.last_grid = None

    def scan_callback(self, msg: LaserScan):
        try:
            grid = scan_to_grid(msg)  # (3, H, W)

            # Build 2-frame (prev + current) 6-channel input as in training
            prev = self.last_grid if self.last_grid is not None else np.zeros_like(grid)
            stacked = np.concatenate([prev.astype(np.float32), grid.astype(np.float32)], axis=0)  # (6,H,W)
            inp = torch.from_numpy(stacked).unsqueeze(0).to(self.device)  # (1,6,H,W)

            t0 = time.time()
            with torch.no_grad():
                out = self.model(inp)  # (1,4,H,W)
            infer_ms = (time.time() - t0) * 1000.0

            out = out[0].detach().cpu().numpy()  # (4,H,W)
            heatmap = 1.0 / (1.0 + np.exp(-out[0]))  # sigmoid
            vx_map = out[1]
            vy_map = out[2]
            yaw_map = out[3]

            obstacles = extract_obstacles(heatmap, vx_map, vy_map, yaw_map)
            markers = self.build_markers(obstacles, msg.header)
            self.marker_pub.publish(markers)

            self.get_logger().debug(f"Published {len(obstacles)} obstacles, infer {infer_ms:.1f} ms")
            self.last_grid = grid
        except Exception as e:
            self.get_logger().error(f"scan_callback failed: {e}")

    def build_markers(self, obstacles: List[dict], header: Header) -> MarkerArray:
        arr = MarkerArray()

        # Clear previous markers
        del_marker = Marker()
        del_marker.header = header
        del_marker.action = Marker.DELETEALL
        arr.markers.append(del_marker)

        for i, ob in enumerate(obstacles):
            # Sphere marker for obstacle body
            m = Marker()
            m.header = header
            m.ns = 'obstacles'
            m.id = i * 2
            m.action = Marker.ADD
            m.type = Marker.SPHERE
            m.pose.position.x = ob['x']
            m.pose.position.y = ob['y']
            m.pose.position.z = 0.0
            q = yaw_to_quaternion(ob['yaw'])
            m.pose.orientation.x = q[0]
            m.pose.orientation.y = q[1]
            m.pose.orientation.z = q[2]
            m.pose.orientation.w = q[3]
            # Size: sphere diameter from estimated radius
            diameter = max(2.0 * ob['size'], 0.05)
            m.scale.x = diameter
            m.scale.y = diameter
            m.scale.z = 0.2
            m.color.r = 0.1
            m.color.g = 0.8
            m.color.b = 0.2
            m.color.a = 0.9
            m.lifetime = rclpy.duration.Duration(seconds=MARKER_LIFETIME).to_msg()
            arr.markers.append(m)

            # Text marker for vx, vy, yaw
            t = Marker()
            t.header = header
            t.ns = 'obstacles_text'
            t.id = i * 2 + 1
            t.action = Marker.ADD
            t.type = Marker.TEXT_VIEW_FACING
            t.pose.position.x = ob['x']
            t.pose.position.y = ob['y']
            t.pose.position.z = 0.4
            t.scale.z = 0.25
            t.color.r = 1.0
            t.color.g = 1.0
            t.color.b = 1.0
            t.color.a = 0.9
            t.text = f"vx:{ob['vx']:.2f} vy:{ob['vy']:.2f} yaw:{ob['yaw']:.2f} size:{ob['size']:.2f}"
            t.lifetime = rclpy.duration.Duration(seconds=MARKER_LIFETIME).to_msg()
            arr.markers.append(t)

        return arr


def main(argv=None):
    rclpy.init(args=argv)
    node = TinyCenterSpeedPerception()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
