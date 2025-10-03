#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, csv, math, argparse
import numpy as np
import torch

# ==============================
# 🔧 기본 설정
# /home/harry/ros2_ws/src/TinyCenterSpeed/src/pt/_0super_best_CenterSpeedDense_trainfree41561_20250817_004834_epoch_11_loss_1_63189.pt
#/home/harry/ros2_ws/src/TinyCenterSpeed/src/pt/0_best_objfree_trainfree52497_20250818_170557_epoch_14_loss_088283.pt
#/home/harry/ros2_ws/src/TinyCenterSpeed/src/pt/0_CenterSpeedResidual_free52497_20250819_123751_epoch_25_loss_085583.pt
#/home/harry/ros2_ws/src/TinyCenterSpeed/src/pt/1_CenterSpeedDenseBottleneck_free52497_20250820_010558_epoch_25_loss_1_88935.pt
# /home/harry/ros2_ws/src/TinyCenterSpeed/src/pt/1_CenterSpeedBottleneckCBAM20250821_045235_epoch_18.pt
# /home/harry/ros2_ws/src/TinyCenterSpeed/src/pt/CenterSpeedBottleneckSENet20250821_072057_epoch_25.pt
# /home/harry/ros2_ws/src/TinyCenterSpeed/src/pt/1_CenterSpeedBottleneckSENet20250821_072554_epoch_27.pt
# /home/harry/ros2_ws/src/TinyCenterSpeed/src/pt/1_CenterSpeedBottleneckCBAM_no_spatial20250821_223218_epoch_16.pt
# ==============================
DEFAULTS = {
    "csv_dir": "/home/harry/evaluation_bag/evluation_csv/free",
    "ckpt": "/home/harry/ros2_ws/src/TinyCenterSpeed/src/pt/1_CenterSpeedBottleneckCBAM_no_spatial20250821_223218_epoch_16.pt",
    "model": "bottleneck_cbam",   # dense / residual / cbam / bottleneck, bottleneck_cbam,bottleneck_SENet
    "image_size": 128,
    "pixelsize": 0.1,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "prob_thr": 0.7,   # heatmap 확률 threshold
}
# ==============================

# Repo 경로 세팅
CURRENT = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
REPO_ROOT = os.path.dirname(os.path.dirname(CURRENT))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

# 모델/데이터셋 import
from TinyCenterSpeed.src.models.CenterSpeed import (
    CenterSpeedDense, CenterSpeedDenseResidual,
    CenterSpeedDenseCBAM, CenterSpeedDenseBottleneck,
    CenterSpeedBottleneckCBAM, CenterSpeedBottleneckSENet
)
from TinyCenterSpeed.dataset.CenterSpeed_dataset import CenterSpeedDataset


# -----------------------------
# CSV 파싱
# -----------------------------
def parse_csv_row(line_bytes: bytes):
    row = next(csv.reader([line_bytes.decode("utf-8").strip()]))
    lidar_field = row[0].strip()
    if "(" in lidar_field and ")" in lidar_field:
        lidar_field = lidar_field[lidar_field.find("(")+1 : lidar_field.rfind(")")]
    lidar = np.fromstring(lidar_field, sep=",", dtype=np.float32)
    if lidar.size == 0:
        lidar = np.fromstring(lidar_field, sep=" ", dtype=np.float32)
    return lidar


def find_top1_peak(heat: np.ndarray):
    idx = int(np.argmax(heat))
    H, W = heat.shape
    y, x = np.unravel_index(idx, (H, W))
    return x, y, float(heat[y, x])


MODEL_ZOO = {
    "dense": CenterSpeedDense,
    "residual": CenterSpeedDenseResidual,
    "cbam": CenterSpeedDenseCBAM,
    "bottleneck": CenterSpeedDenseBottleneck,
    "bottleneck_cbam": CenterSpeedBottleneckCBAM,
    "bottleneck_SENet": CenterSpeedBottleneckSENet
}


# -----------------------------
# 메인
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_dir", default=DEFAULTS["csv_dir"])
    ap.add_argument("--ckpt", default=DEFAULTS["ckpt"])
    ap.add_argument("--model", default=DEFAULTS["model"], choices=list(MODEL_ZOO.keys()))
    ap.add_argument("--image_size", type=int, default=DEFAULTS["image_size"])
    ap.add_argument("--pixelsize", type=float, default=DEFAULTS["pixelsize"])
    ap.add_argument("--device", default=DEFAULTS["device"])
    ap.add_argument("--prob_thr", type=float, default=DEFAULTS["prob_thr"])
    args = ap.parse_args()

    # Dataset 준비
    ds = CenterSpeedDataset(dataset_path=args.csv_dir, transform=None, dense=False)
    ds.change_image_size(args.image_size)
    ds.change_pixel_size(args.pixelsize)

    # 모델 준비
    Net = MODEL_ZOO[args.model]
    net = Net(input_channels=4, image_size=args.image_size).to(args.device)
    state = torch.load(args.ckpt, map_location=args.device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {k.replace("module.", ""): v for k, v in state.items()}
    net.load_state_dict(state, strict=False)
    net.eval()
    print(f"[info] Using checkpoint: {args.ckpt}")

    # 평가 루프
    H = W = args.image_size
    csv_files = sorted([os.path.join(args.csv_dir, f) for f in os.listdir(args.csv_dir) if f.endswith(".csv")])
    if not csv_files:
        raise FileNotFoundError(f"No CSV files in: {args.csv_dir}")

    fp = tn = 0
    n = 0

    for path in csv_files:
        lines = open(path, "rb").readlines()
        if len(lines) < 2:
            continue

        # (t-1, t) 쌍으로
        for i in range(len(lines) - 1):
            lidar0 = parse_csv_row(lines[i])
            lidar1 = parse_csv_row(lines[i+1])

            # 빔 수 맞추기 (1080 기준)
            L_target = ds.cos.numel()
            if lidar0.size != L_target:
                lidar0 = (lidar0[:L_target] if lidar0.size > L_target
                          else np.pad(lidar0, (0, L_target - lidar0.size), mode="edge"))
            if lidar1.size != L_target:
                lidar1 = (lidar1[:L_target] if lidar1.size > L_target
                          else np.pad(lidar1, (0, L_target - lidar1.size), mode="edge"))

            # 전처리
            f0 = ds.preprocess(torch.from_numpy(lidar0))
            f1 = ds.preprocess(torch.from_numpy(lidar1))
            inp = torch.stack([f0, f1], dim=0).view(1, 4, H, W).to(args.device)

            with torch.no_grad():
                out = net(inp)
                heat = torch.sigmoid(out[0, 0]).cpu().numpy()

            _, _, score = find_top1_peak(heat)

            # 조건에 따라 FP/TN 증가
            if score >= args.prob_thr:
                fp += 1
            else:
                tn += 1
            n += 1

    print("====================================================")
    print(f"CSV folder       : {args.csv_dir}")
    print(f"Checkpoint       : {args.ckpt}")
    print(f"Model            : {args.model}")
    print(f"Evaluated samples: {n}")
    print("----------------------------------------------------")
    print(f"FP: {fp}  |  TN: {tn}")
    print("====================================================")


if __name__ == "__main__":
    main()
