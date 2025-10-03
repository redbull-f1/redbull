#!/usr/bin/env python3
import os
import sys
import numpy as np
import csv
import torch
import time
import glob

'''
bag 파일 불러와서 모델 예측 결과를 CSV로 저장합니다.
bag 파일 경로: /home/harry/Downloads/levine_obs.db3
'''

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

def main():
    import rosbag2_py
    from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    from sensor_msgs.msg import LaserScan
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--bag_dir', type=str, default='/home/harry/sim_ws/src/f1tenth_gym_ros/bag', help='Bag directory')
    parser.add_argument('--csv_dir', type=str, default='/home/harry/ros2_ws/src/redbull/csv', help='CSV output directory')
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--pixelsize', type=float, default=0.1)
    parser.add_argument('--num_opponents', type=int, default=1)
    parser.add_argument('--threshold', type=float, default=0.3)
    args = parser.parse_args()

    redbull_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(redbull_root, image_size=args.image_size, device=device)

    bag_files = sorted(glob.glob(os.path.join(args.bag_dir, '*.db3')))
    all_rows = []
    for bag_path in bag_files:
        print(f"Processing {bag_path}")
        storage_options = StorageOptions(uri=bag_path, storage_id='sqlite3')
        converter_options = ConverterOptions('', '')
        reader = SequentialReader()
        reader.open(storage_options, converter_options)
        topic_types = {t.name: t.type for t in reader.get_all_topics_and_types()}
        frame1 = None
        frame2 = None
        origin_offset = (args.image_size // 2) * args.pixelsize
        while reader.has_next():
            topic, data, timestamp = reader.read_next()
            if topic == '/scan':
                msg_type = get_message(topic_types[topic])
                scan_msg = deserialize_message(data, msg_type)
                t = scan_msg.header.stamp.sec + scan_msg.header.stamp.nanosec * 1e-9
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
                # CenterSpeedDense: outputs shape (4, H, W) or (B, 4, H, W)
                if isinstance(outputs, torch.Tensor):
                    output = outputs.cpu().numpy()
                else:
                    output = outputs[0].cpu().numpy()
                if output.ndim == 4:
                    output = output[0]
                # output[0]: heatmap, output[1]: vx, output[2]: vy, output[3]: yaw
                heatmap = output[0]
                vx_map = output[1]
                vy_map = output[2]
                yaw_map = output[3]
                x, y = find_k_peaks(heatmap, args.num_opponents, args.threshold, args.pixelsize, origin_offset)
                for i in range(len(x)):
                    # 이미지 좌표로 변환
                    x_img = int(round((x[i] + origin_offset) / args.pixelsize))
                    y_img = int(round((y[i] + origin_offset) / args.pixelsize))
                    # 이미지 범위 체크
                    if 0 <= x_img < heatmap.shape[1] and 0 <= y_img < heatmap.shape[0]:
                        vx = vx_map[y_img, x_img]
                        vy = vy_map[y_img, x_img]
                        yaw = yaw_map[y_img, x_img]
                    else:
                        vx = vy = yaw = 0.0
                    all_rows.append([
                        t,
                        x[i],
                        y[i],
                        vx,
                        vy,
                        yaw
                    ])
    all_rows.sort(key=lambda row: row[0])
    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    out_csv = os.path.join(args.csv_dir, f'allbags_model_{timestamp_str}.csv')
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'x', 'y', 'vx', 'vy', 'yaw'])
        for row in all_rows:
            writer.writerow([f"{row[0]:.6f}", f"{row[1]:.6f}", f"{row[2]:.6f}", f"{row[3]:.6f}", f"{row[4]:.6f}", f"{row[5]:.6f}"])
    print(f"Saved: {out_csv}")

if __name__ == '__main__':
    main()
