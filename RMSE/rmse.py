import csv
import os
import numpy as np

'''
    두 timestamp 간의 차이가 40hz 이하인 경우에만 매칭 하도록 하자 
'''

def read_csv(filepath, cols, skip_header=True):
    data = []
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        if skip_header:
            next(reader)
        for row in reader:
            data.append([float(row[c]) for c in cols])
    return np.array(data)

def find_closest_indices(timestamps1, timestamps2):
    # For each t1, find index of t2 that is closest
    indices = []
    for t1 in timestamps1:
        idx = np.abs(timestamps2 - t1).argmin()
        indices.append(idx)
    return np.array(indices)

def main():
    # 파일 경로
    pred_path = '/home/harry/ros2_ws/src/redbull/csv/redbull_obs_test2.csv'
    gt_path = '/home/harry/sim_ws/src/f1tenth_gym_ros/gt_csv/redbull_obs_test2.csv'
    out_path = '/home/harry/ros2_ws/src/redbull/RMSE/rmse_redbull_obs_test2.csv'

    # 예측값: [timestamp, x, y, vx, vy, yaw]
    pred = read_csv(pred_path, [0,1,2,3,4,5])
    # GT: [timestamp, lidar, intensity, x, y, vx, vy, yaw]
    gt = read_csv(gt_path, [0,3,4,5,6,7])

    # timestamp 매칭
    pred_ts = pred[:,0]
    gt_ts = gt[:,0]
    gt_idx = find_closest_indices(pred_ts, gt_ts)
    matched_gt = gt[gt_idx]

    # RMSE 계산
    rmse_vals = np.sqrt(np.mean((pred[:,1:] - matched_gt[:,1:])**2, axis=0))

    # CSV 저장 (1행: 예측값, 2행: RMSE)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp','x','y','vx','vy','yaw'])
        # writer.writerow([f'{v:.6f}' for v in pred[0]])
        writer.writerow(['RMSE'] + [f'{v:.6f}' for v in rmse_vals])
    print(f'Saved: {out_path}')

if __name__ == '__main__':
    main()
