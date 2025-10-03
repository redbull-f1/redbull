#!/usr/bin/env python3

import os
import sys
import numpy as np
import csv
import time
import torch
from scipy.spatial.transform import Rotation
from nav_msgs.msg import Odometry

'''
python3 /home/harry/ros2_ws/src/redbull/data/bag_model2csv_sim.py --bag /home/harry/Downloads/levine_obs.db3
'''

def get_model(redbull_root, image_size=64, device='cpu'):
    # Add model paths
    model_paths = [
        os.path.join(redbull_root, 'train'),
        os.path.join(redbull_root, 'train', 'models'),
        redbull_root
    ]
    for path in model_paths:
        if os.path.exists(path) and path not in sys.path:
            sys.path.insert(0, path)
    try:
        from models.CenterSpeed import CenterSpeedDense
    except ImportError:
        return None
    model = CenterSpeedDense(image_size=image_size)
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
    # image: 2D numpy array
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

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--bag', type=str, required=True, help='Input ROS2 bag file (db3)')
    parser.add_argument('--csv', type=str, default=None, help='Output CSV file path')
    parser.add_argument('--num_opponents', type=int, default=1, help='Number of objects to extract')
    parser.add_argument('--threshold', type=float, default=0.3, help='Detection threshold')
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--pixelsize', type=float, default=0.1)
    args = parser.parse_args()

    # Output CSV path
    if args.csv is None:
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        args.csv = f'/home/harry/ros2_ws/src/redbull/data/model_output_{timestamp_str}.csv'

    import rosbag2_py
    from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    from sensor_msgs.msg import LaserScan

    # Model
    redbull_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(redbull_root, image_size=args.image_size, device=device)
    if model is None:
        print('Model load failed')
        return

    # For 2-frame input
    frame1 = None
    frame2 = None
    origin_offset = (args.image_size // 2) * args.pixelsize

    # Bag read
    storage_options = StorageOptions(uri=args.bag, storage_id='sqlite3')
    converter_options = ConverterOptions('', '')
    reader = SequentialReader()
    reader.open(storage_options, converter_options)
    topic_types = {t.name: t.type for t in reader.get_all_topics_and_types()}

    csv_rows = []
    while reader.has_next():
        topic, data, timestamp = reader.read_next()
        if topic == '/ego_racecar/odom':
            msg_type = get_message(topic_types[topic])
            msg = deserialize_message(data, msg_type)
            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            # Save odom for timestamp
            last_odom_time = t
        if topic == '/scan':
            msg_type = get_message(topic_types[topic])
            scan_msg = deserialize_message(data, msg_type)
            # Preprocess
            if frame1 is None:
                frame1 = preprocess_scan(scan_msg, args.image_size, args.pixelsize)
                continue
            if frame2 is None:
                frame2 = preprocess_scan(scan_msg, args.image_size, args.pixelsize)
                continue
            frame1 = frame2
            frame2 = preprocess_scan(scan_msg, args.image_size, args.pixelsize)
            if frame1 is None or frame2 is None:
                continue
            input_tensor = np.concatenate([frame1, frame2], axis=1)
            input_tensor = torch.FloatTensor(input_tensor).to(device)
            with torch.no_grad():
                outputs = model(input_tensor)
            if isinstance(outputs, tuple):
                output_hm = outputs[0]
            else:
                output_hm = outputs
            if len(output_hm.shape) == 4:
                output_hm = output_hm.squeeze(0)
            if len(output_hm.shape) == 3:
                output_hm = output_hm[0]
            if len(output_hm.shape) != 2:
                continue
            # Find peaks
            x, y = find_k_peaks(output_hm.cpu().numpy(), args.num_opponents, args.threshold, args.pixelsize, origin_offset)
            # Save to CSV (timestamp, x, y)
            for px, py in zip(x, y):
                csv_rows.append([f"{last_odom_time:.6f}", f"{px:.6f}", f"{py:.6f}"])
    # Save CSV
    os.makedirs(os.path.dirname(args.csv), exist_ok=True)
    with open(args.csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'x', 'y'])
        writer.writerows(csv_rows)
    print(f"Saved: {args.csv}")

if __name__ == '__main__':
    main()
