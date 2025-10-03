#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, csv, math, glob, argparse
import numpy as np
import torch
from typing import Tuple

"""
export PYTHONPATH=/home/harry/ros2_ws/src:$PYTHONPATH
/usr/bin/python3 /home/harry/ros2_ws/src/TinyCenterSpeed/src/train/evaluate_RMSE_TP_FN.py

"""


# ==============================
# 🔧 기본 설정 (여기만 바꾸면 됨)
# /home/harry/ros2_ws/src/TinyCenterSpeed/src/pt/_0super_best_CenterSpeedDense_trainfree41561_20250817_004834_epoch_11_loss_1_63189.pt
# /home/harry/ros2_ws/src/TinyCenterSpeed/src/pt/0_best_objfree_trainfree52497_20250818_170557_epoch_14_loss_088283.pt
#/home/harry/ros2_ws/src/TinyCenterSpeed/src/pt/0_CenterSpeedResidual_free52497_20250819_123751_epoch_25_loss_085583.pt
#/home/harry/ros2_ws/src/TinyCenterSpeed/src/pt/1_CenterSpeedDenseBottleneck_free52497_20250820_010558_epoch_25_loss_1_88935.pt
# /home/harry/ros2_ws/src/TinyCenterSpeed/src/pt/CenterSpeedBottleneckCBAM20250821_040035_epoch_4.pt
# /home/harry/ros2_ws/src/TinyCenterSpeed/src/pt/1_CenterSpeedBottleneckSENet20250821_072554_epoch_27.pt
# /home/harry/ros2_ws/src/TinyCenterSpeed/src/pt/1_CenterSpeedBottleneckCBAM_no_spatial20250821_223218_epoch_16.pt
# ==============================
DEFAULTS = {
    "csv_dir": "/home/harry/evaluation_bag/evluation_csv",
    "ckpt": "/home/harry/ros2_ws/src/TinyCenterSpeed/src/pt/1_CenterSpeedBottleneckCBAM_no_spatial20250821_223218_epoch_16.pt",
    "model": "bottleneck_cbam",          # dense / residual / cbam / bottleneck, bottleneck_cbam,bottleneck_SENet
    "image_size": 128,
    "pixelsize": 0.1,          # [m/pixel]
    "dist_thr": 0.35,           # TP 임계 (m)
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}
# ==============================

# Repo 경로 세팅: train/에서 2-up → TinyCenterSpeed 루트
CURRENT = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
REPO_ROOT = os.path.dirname(os.path.dirname(CURRENT))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

# 모델/데이터셋 import (네 레포 구조에 맞춤)
from TinyCenterSpeed.src.models.CenterSpeed import (
    CenterSpeedDense, CenterSpeedDenseResidual,
    CenterSpeedDenseCBAM, CenterSpeedDenseBottleneck,
    CenterSpeedBottleneckCBAM,CenterSpeedBottleneckSENet
)
from TinyCenterSpeed.dataset.CenterSpeed_dataset import CenterSpeedDataset


# -----------------------------
# CSV 파싱 유틸
#   row: lidar_tuple, intensity_tuple, x, y, vx, vy, yaw
# -----------------------------
def parse_csv_row(line_bytes: bytes) -> Tuple[np.ndarray, np.ndarray]:
    row = next(csv.reader([line_bytes.decode("utf-8").strip()]))
    if len(row) < 7:
        raise ValueError(f"CSV row has <7 columns: {row}")

    lidar_field = row[0].strip()
    # "(a, b, ...)" → "a, b, ..."
    if "(" in lidar_field and ")" in lidar_field:
        lidar_field = lidar_field[lidar_field.find("(")+1 : lidar_field.rfind(")")]

    lidar = np.fromstring(lidar_field, sep=",", dtype=np.float32)
    if lidar.size == 0:
        lidar = np.fromstring(lidar_field, sep=" ", dtype=np.float32)
    if lidar.size == 0:
        raise ValueError("Failed to parse lidar tuple")

    x, y, vx, vy, yaw = map(float, row[2:7])
    return lidar, np.array([x, y, vx, vy, yaw], dtype=np.float32)


def wrap_angle_rad(a: float) -> float:
    return (a + math.pi) % (2*math.pi) - math.pi


def find_top1_peak(heat: np.ndarray):
    idx = int(np.argmax(heat))
    H, W = heat.shape
    y, x = np.unravel_index(idx, (H, W))
    return x, y, float(heat[y, x])


def pix_to_lidar(x_pix: int, y_pix: int, pixelsize: float, origin_offset: float):
    # (0,0) 픽셀이 (-origin_offset, -origin_offset) m에 해당하도록 복원
    x = x_pix * pixelsize - origin_offset
    y = y_pix * pixelsize - origin_offset
    return float(x), float(y)


MODEL_ZOO = {
    "dense": CenterSpeedDense,
    "residual": CenterSpeedDenseResidual,
    "cbam": CenterSpeedDenseCBAM,
    "bottleneck": CenterSpeedDenseBottleneck,
    "bottleneck_cbam": CenterSpeedBottleneckCBAM,
    "bottleneck_SENet": CenterSpeedBottleneckSENet
}


def main():
    # argparse는 옵션 덮어쓰기 용도로만 사용 (기본 실행 가능)
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_dir", default=DEFAULTS["csv_dir"])
    ap.add_argument("--ckpt", default=DEFAULTS["ckpt"])
    ap.add_argument("--model", default=DEFAULTS["model"], choices=list(MODEL_ZOO.keys()))
    ap.add_argument("--image_size", type=int, default=DEFAULTS["image_size"])
    ap.add_argument("--pixelsize", type=float, default=DEFAULTS["pixelsize"])
    ap.add_argument("--dist_thr", type=float, default=DEFAULTS["dist_thr"])
    ap.add_argument("--device", default=DEFAULTS["device"])
    args = ap.parse_args()

    # --------- 데이터 전처리 세팅: 학습과 동일하게 ---------
    ds = CenterSpeedDataset(dataset_path=args.csv_dir, transform=None, dense=False)
    ds.change_image_size(args.image_size)
    ds.change_pixel_size(args.pixelsize)

    # --------- 모델 및 체크포인트 ---------
    Net = MODEL_ZOO[args.model]
    net = Net(input_channels=4, image_size=args.image_size).to(args.device)

    state = torch.load(args.ckpt, map_location=args.device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    # DataParallel 등 'module.' prefix 제거, head mismatch 허용
    state = {k.replace("module.", ""): v for k, v in state.items()}
    missing, unexpected = net.load_state_dict(state, strict=False)
    if missing:   print("[warn] missing keys:", missing)
    if unexpected:print("[warn] unexpected keys:", unexpected)
    net.eval()
    print(f"[info] Using checkpoint: {args.ckpt}")

    # --------- 평가 루프 ---------
    H = W = args.image_size
    pixelsize = args.pixelsize
    origin_offset = (H // 2) * pixelsize

    csv_files = sorted([os.path.join(args.csv_dir, f) for f in os.listdir(args.csv_dir) if f.endswith(".csv")])
    if not csv_files:
        raise FileNotFoundError(f"No CSV files in: {args.csv_dir}")

    # 메트릭 누적
    n = 0
    se_x = se_y = se_vx = se_vy = se_yaw = 0.0
    sum_dx = sum_dy = 0.0     # 🔥 평균 편차용 누적합
    tp = fn = 0

    for path in csv_files:
        with open(path, "rb") as f:
            lines = f.readlines()

        if len(lines) < 2:
            continue

        # (t-1, t) 슬라이딩 윈도우
        for i in range(len(lines) - 1):
            lidar0, _     = parse_csv_row(lines[i])
            lidar1, gt_np = parse_csv_row(lines[i+1])  # GT는 두 번째 줄 기준

            # 학습 기준 길이 정합(예: 1080 빔)
            L_target = ds.cos.numel()  # CenterSpeedDataset에서 사용하던 각도 테이블 길이
            if lidar0.size != L_target:
                lidar0 = (lidar0[:L_target] if lidar0.size > L_target
                          else np.pad(lidar0, (0, L_target - lidar0.size), mode="edge"))
            if lidar1.size != L_target:
                lidar1 = (lidar1[:L_target] if lidar1.size > L_target
                          else np.pad(lidar1, (0, L_target - lidar1.size), mode="edge"))

            # LiDAR → BEV (occupancy, density) 각 2채널
            f0 = ds.preprocess(torch.from_numpy(lidar0))
            f1 = ds.preprocess(torch.from_numpy(lidar1))

            # (B,4,H,W) 입력 구성
            inp = torch.stack([f0, f1], dim=0).view(1, 4, H, W).to(args.device)

            with torch.no_grad():
                out = net(inp)  # [1,4,H,W]
                heat = torch.sigmoid(out[0, 0]).cpu().numpy()
                vx_map = out[0, 1].cpu().numpy()
                vy_map = out[0, 2].cpu().numpy()
                yaw_map = out[0, 3].cpu().numpy()

            # heatmap 최대점 → LiDAR 좌표 복원
            x_pix, y_pix, _ = find_top1_peak(heat)
            px, py = pix_to_lidar(x_pix, y_pix, pixelsize, origin_offset)

            # 예측 속성
            pvx = float(vx_map[y_pix, x_pix])
            pvy = float(vy_map[y_pix, x_pix])
            pyaw = float(yaw_map[y_pix, x_pix])

            # GT
            gx, gy, gvx, gvy, gyaw = map(float, gt_np)

            # 오차
            dx = px - gx
            dy = py - gy
            dvx = pvx - gvx
            dvy = pvy - gvy
            dyaw = wrap_angle_rad(pyaw - gyaw)

            # 누적
            se_x  += dx * dx
            se_y  += dy * dy
            se_vx += dvx * dvx
            se_vy += dvy * dvy
            se_yaw += dyaw * dyaw
            sum_dx += dx           # 🔥 평균 편차 누적
            sum_dy += dy           # 🔥 평균 편차 누적
            n += 1

            # 0.5m 이내 → TP, 초과 → FN
            if math.hypot(dx, dy) <= args.dist_thr:
                tp += 1
            else:
                fn += 1

    # RMSE / Mean Bias
    def rmse(se, count): return math.sqrt(se / max(1, count))
    rmse_x   = rmse(se_x, n)
    rmse_y   = rmse(se_y, n)
    rmse_pos = math.sqrt((se_x + se_y) / max(1, n))
    rmse_vx  = rmse(se_vx, n)
    rmse_vy  = rmse(se_vy, n)
    rmse_yaw = rmse(se_yaw, n)
    mean_dx  = sum_dx / max(1, n)   # 🔥 평균 Δx
    mean_dy  = sum_dy / max(1, n)   # 🔥 평균 Δy

    # 출력
    print("====================================================")
    print(f"CSV folder       : {args.csv_dir}")
    print(f"Checkpoint       : {args.ckpt}")
    print(f"Model            : {args.model}")
    print(f"Image size       : {H}x{W}, pixelsize {pixelsize} m")
    print(f"Evaluated samples: {n}")
    print(f"Distance thr     : {args.dist_thr:.3f} m")
    print("----------------------------------------------------")
    print(f"RMSE x    : {rmse_x:.6f} m")
    print(f"RMSE y    : {rmse_y:.6f} m")
    print(f"RMSE pos  : {rmse_pos:.6f} m  (sqrt(mean(dx^2+dy^2)))")
    print(f"Mean Δx   : {mean_dx:.6f} m")  # 🔥 추가 출력
    print(f"Mean Δy   : {mean_dy:.6f} m")  # 🔥 추가 출력
    print(f"RMSE vx   : {rmse_vx:.6f} m/s")
    print(f"RMSE vy   : {rmse_vy:.6f} m/s")
    print(f"RMSE yaw  : {rmse_yaw:.6f} rad")
    print("----------------------------------------------------")
    print(f"TP: {tp}  |  FN: {fn}")
    print("====================================================")


if __name__ == "__main__":
    main()
