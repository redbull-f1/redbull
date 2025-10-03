#!/usr/bin/env python3
import csv
import os
import numpy as np
'''
python3 /home/harry/ros2_ws/src/redbull/data/RMSE_csv.py --gt_csv /home/harry/ros2_ws/src/redbull/data/obstacle_lidar_20250808_233222.csv --model_csv /home/harry/ros2_ws/src/redbull/data/model_output_20250808_233940.csv
'''

def read_csv_to_dict(path):
    data = {}
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        # Try to detect x/y column names
        x_key = None
        y_key = None
        header = reader.fieldnames
        if 'x' in header and 'y' in header:
            x_key, y_key = 'x', 'y'
        elif 'x_lidar' in header and 'y_lidar' in header:
            x_key, y_key = 'x_lidar', 'y_lidar'
        else:
            raise KeyError(f"No x/y or x_lidar/y_lidar columns in {path}")
        for row in reader:
            t = float(row['timestamp'])
            x = float(row[x_key])
            y = float(row[y_key])
            data[t] = (x, y)
    return data

def find_closest_timestamp(target, candidates, max_diff=0.05):  # 0.025로 바꿀까?
    # Find closest timestamp in candidates to target, within max_diff seconds
    arr = np.array(list(candidates))
    idx = (np.abs(arr - target)).argmin()
    if abs(arr[idx] - target) <= max_diff:
        return arr[idx]
    return None

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_csv', type=str, required=True, help='bag2csv_sim 결과 CSV (obstacle_lidar_*.csv)')
    parser.add_argument('--model_csv', type=str, required=True, help='bag_model2csv_sim 결과 CSV (model_output_*.csv)')
    parser.add_argument('--out_csv', type=str, default=None, help='출력 RMSE CSV 경로')
    parser.add_argument('--max_time_diff', type=float, default=0.05, help='timestamp 매칭 허용 오차(초)')
    args = parser.parse_args()

    import time
    if args.out_csv is None:
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        args.out_csv = os.path.join(os.path.dirname(args.gt_csv), f'RMSE_result_{timestamp_str}.csv')

    gt = read_csv_to_dict(args.gt_csv)
    model = read_csv_to_dict(args.model_csv)

    gt_times = sorted(gt.keys())
    model_times = sorted(model.keys())

    rows = []
    x_rmse_list = []
    y_rmse_list = []
    for t in gt_times:
        t2 = find_closest_timestamp(t, model_times, args.max_time_diff)
        if t2 is not None:
            x1, y1 = gt[t]
            x2, y2 = model[t2]
            x_err = x2 - x1
            y_err = y2 - y1
            rows.append([f"{t:.6f}", f"{x1:.6f}", f"{y1:.6f}", f"{x2:.6f}", f"{y2:.6f}", f"{x_err**2:.6f}", f"{y_err**2:.6f}"])
            x_rmse_list.append(x_err**2)
            y_rmse_list.append(y_err**2)
    # 전체 RMSE
    if x_rmse_list and y_rmse_list:
        x_rmse = np.sqrt(np.mean(x_rmse_list))
        y_rmse = np.sqrt(np.mean(y_rmse_list))
        print(f"X RMSE: {x_rmse:.4f}, Y RMSE: {y_rmse:.4f}")
    else:
        x_rmse = 0.0
        y_rmse = 0.0
        print("No matched timestamps found.")
    # Add RMSE columns to output
    for i, row in enumerate(rows):
        if i == 0:
            row.extend([f"{x_rmse:.6f}", f"{y_rmse:.6f}"])
        else:
            row.extend(["", ""])
    # 전체 RMSE
    if x_rmse_list and y_rmse_list:
        x_rmse = np.sqrt(np.mean(x_rmse_list))
        y_rmse = np.sqrt(np.mean(y_rmse_list))
        print(f"X RMSE: {x_rmse:.4f}, Y RMSE: {y_rmse:.4f}")
    else:
        print("No matched timestamps found.")
    # Save CSV
    with open(args.out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'gt_x', 'gt_y', 'model_x', 'model_y', 'x_sq_error', 'y_sq_error', 'x_RMSE', 'y_RMSE'])
        writer.writerows(rows)
    print(f"Saved: {args.out_csv}")

if __name__ == '__main__':
    main()
