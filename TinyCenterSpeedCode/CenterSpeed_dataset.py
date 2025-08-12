# -*- coding: utf-8 -*-
import os
import csv
import math
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class CenterSpeedDataset(Dataset):
    """
    TinyCenterSpeed Dataset (intensity 채널 제거판, 고속 로더)
    - 입력: 2프레임 × (occupancy, density) = 4채널  -> (4, H, W)
    - GT:
        heatmap: (H, W)
        dense:   (H, W, 3)  # vx, vy, yaw 값을 heatmap 형태로 확장
        data:    (5,)       # x, y, vx, vy, yaw
    """
    def __init__(self, dataset_path, transform=None, dense=False):
        self.dataset_path = dataset_path
        self.transform = transform
        self.use_heatmaps = True
        self.dense = dense
        self.consider_free_paths = True

        # 해상도/격자 설정
        self.pixelsize = 0.1     # [m/pixel]
        self.image_size = 128    # H=W
        self.feature_size = 2    # (occupancy, density)
        self.seq_len = 2         # 프레임 수

        self.origin_offset = (self.image_size // 2) * self.pixelsize

        # σ 추천값 (40cm 객체, 0.1m/px → 반지름 2px)
        self.sx = self.sy = 2.0

        self.len = 0
        self.number_of_sets = None

        # LiDAR 각도 (FOV -135°~+135°), 실제 차량 1080빔 가정
        num_beams = 1080
        self.angles = np.linspace(-2.356194496154785, 2.356194496154785, num_beams).astype(np.float32)
        # torch 텐서로 보관(연산 속도 및 타입 일관성)
        self.cos = torch.from_numpy(np.cos(self.angles))
        self.sin = torch.from_numpy(np.sin(self.angles))

        # CSV 인덱스 생성 (파일별 데이터 라인의 바이트 오프셋)
        self.setup()

    def setup(self):
        """디렉터리 내 CSV를 스캔해 전체 길이/인덱스 맵 + 라인 오프셋 인덱스 생성"""
        self.file_paths = [os.path.join(self.dataset_path, f)
                           for f in os.listdir(self.dataset_path) if f.endswith('.csv')]
        self.file_paths.sort()
        self.file_indices = []
        self.offsets_per_file = []   # 각 파일의 데이터 라인 시작 바이트 오프셋 배열
        num_rows_per_file = []

        self.len = 0
        for file_path in self.file_paths:
            offsets = []
            with open(file_path, 'rb') as f:
                # 헤더 1줄 스킵
                _ = f.readline()
                pos = f.tell()
                line = f.readline()
                while line:
                    offsets.append(pos)
                    pos = f.tell()
                    line = f.readline()
            # 시퀀스 길이 때문에 마지막 (seq_len-1)개는 시작점으로 못 씀
            num_rows = max(0, len(offsets) - (self.seq_len - 1))
            self.offsets_per_file.append(offsets)
            self.file_indices.append((self.len, self.len + num_rows))
            self.len += num_rows
            num_rows_per_file.append(num_rows)

        # 필요 시 로그 출력 (많으면 주석 처리 권장)
        for path in self.file_paths:
            idx = self.file_paths.index(path)
            print("Reading file: ", path)
            print("Entries     : ", num_rows_per_file[idx])
        print("Total rows : ", self.len)
        print("File index : ", self.file_indices)

    def change_pixel_size(self, pixelsize: float):
        self.pixelsize = pixelsize
        self.origin_offset = (self.image_size // 2) * self.pixelsize
        print("Pixel size   ->", self.pixelsize)
        print("Origin offset->", self.origin_offset)

    def change_image_size(self, image_size: int):
        self.image_size = int(image_size)
        self.origin_offset = (self.image_size // 2) * self.pixelsize
        print("Image size   ->", self.image_size)
        print("Origin offset->", self.origin_offset)

    # --------- 초고속 한 줄 파서 ---------
    # def _parse_row_fast(self, line_bytes: bytes):
    #     """
    #     CSV 한 줄 파싱 (형식: lidar, intensities, x, y, vx, vy, yaw)
    #     - lidar는 '(a, b, c, ...)' 형태라고 가정 → 괄호 제거 후 쉼표 분리
    #     - intensities는 무시
    #     반환: (torch.FloatTensor[lidar_len], torch.FloatTensor[5])
    #     """
    #     s = line_bytes.decode('utf-8').rstrip('\n')
    #     # lidar 필드는 괄호 포함이므로 첫 '('와 짝이 되는 ')'까지 잘라냄
    #     lpar = s.find('(')
    #     rpar = s.find(')', lpar + 1)
    #     lidar_str = s[lpar + 1:rpar]  # 괄호 내부
    #     rest = s[rpar + 2:]           # '), ' 다음부터 나머지 필드

    #     # lidar 값 파싱 (쉼표 기반)
    #     lidar_vals = np.fromstring(lidar_str, sep=',')
    #     lidar_t = torch.from_numpy(lidar_vals.astype(np.float32))

    #     # 나머지 6개 필드: intensities(무시), x, y, vx, vy, yaw
    #     parts = rest.split(',')
    #     # parts[0]: intensities ... 무시
    #     x = float(parts[1]); y = float(parts[2])
    #     vx = float(parts[3]); vy = float(parts[4]); yaw = float(parts[5])
    #     data_t = torch.tensor([x, y, vx, vy, yaw], dtype=torch.float32)
    #     return lidar_t, data_t


    def _parse_row_fast(self, line_bytes: bytes):
        """
        CSV 한 줄 파싱 (형식: lidar, intensities, x, y, vx, vy, yaw)
        - lidar/intensities는 괄호+쉼표 포함 가능, 따옴표로 감싸질 수도 있음
        - csv.reader로 안전 파싱 후, lidar는 괄호 제거 → np.fromstring
        - intensities는 무시
        반환: (torch.FloatTensor[lidar_len], torch.FloatTensor[5])
        """
        s = line_bytes.decode('utf-8').strip()
        row = next(csv.reader([s], delimiter=',', quotechar='"'))
        if len(row) < 7:
            raise ValueError(f"Expected >=7 columns, got {len(row)} in row: {row}")

        lidar_field = row[0].strip()
        # intensities_field = row[1]  # 사용 안 함

        # 뒤 5개는 숫자여야 함
        try:
            x   = float(row[2].strip())
            y   = float(row[3].strip())
            vx  = float(row[4].strip())
            vy  = float(row[5].strip())
            yaw = float(row[6].strip())
        except ValueError as e:
            raise ValueError(f"Failed to parse numeric fields: {row[2:7]}") from e

        # lidar: '(a, b, ...)' 또는 그냥 'a, b, ...'일 수 있음 → 괄호 제거
        if '(' in lidar_field and ')' in lidar_field:
            lidar_field = lidar_field[lidar_field.find('(')+1 : lidar_field.rfind(')')]

        # 쉼표 기준 파싱 (공백 허용). 실패 시 공백 기준 한 번 더 시도
        lidar_vals = np.fromstring(lidar_field, sep=',', dtype=np.float32)
        if lidar_vals.size == 0:
            lidar_vals = np.fromstring(lidar_field, sep=' ', dtype=np.float32)
        if lidar_vals.size == 0:
            raise ValueError(f"Failed to parse lidar tuple from field: {lidar_field[:80]}...")

        lidar_t = torch.from_numpy(lidar_vals.astype(np.float32))
        data_t  = torch.tensor([x, y, vx, vy, yaw], dtype=torch.float32)
        return lidar_t, data_t



    def __getitem__(self, index):
        """
        반환:
          input_data: (seq_len*feature_size, H, W)  # (4, 128, 128)
          heatmap   : (H, W)
          data      : (5,)
          [dense 있을 때] dense_features: (H, W, 3)
          free      : bool (장애물 없는 자유 경로 여부)
        """
        free = False

        # 이 인덱스가 속한 파일/로우 계산
        file_index = next(i for i, (start, end) in enumerate(self.file_indices) if start <= index < end)
        row_index = index - self.file_indices[file_index][0]  # 이 파일 내 시작 로우

        offsets = self.offsets_per_file[file_index]
        path = self.file_paths[file_index]

        seq_data = []
        data = None

        # 필요한 2줄만 빠르게 읽어서 파싱
        with open(path, 'rb') as f:
            # t=0
            f.seek(offsets[row_index])
            line0 = f.readline()
            lidar0, _ = self._parse_row_fast(line0)
            seq_data.append(self.preprocess(lidar0))

            # t=1 (마지막 프레임의 data를 GT로 사용)
            f.seek(offsets[row_index + self.seq_len - 1])
            line1 = f.readline()
            lidar1, data = self._parse_row_fast(line1)
            seq_data.append(self.preprocess(lidar1))

        # 라이다 빔 길이 검사 (길이 불일치 즉시 감지)
        assert lidar0.numel() == self.cos.numel(), \
            f"Lidar length {lidar0.numel()} != angles {self.cos.numel()}"

        # 입력 스택
        input_data = torch.stack(seq_data).view(self.seq_len * self.feature_size,
                                                self.image_size, self.image_size)

        # GT heatmap
        heatmap = self.heatmap(data)

        # free-path 규칙 (뒤에 있거나 너무 멀면 free)
        if data[0] < 0 or np.hypot(data[0].item(), data[1].item()) > 6.4:
            free = True

        # 변환
        if self.transform:
            input_data, heatmap, data = self.transform((input_data, heatmap, data))

        if self.dense:
            dense_features = self.populate_dense_features(data=data)
            return (input_data,
                    heatmap.view(self.image_size, self.image_size),
                    data.view(5),
                    dense_features,
                    free)

        return (input_data,
                heatmap.view(self.image_size, self.image_size),
                data.view(5),
                free)

    def __len__(self):
        return self.len

    @staticmethod
    def gaussian_2d(x, y, x0, y0, sx, sy, A):
        return A * np.exp(-((x - x0) ** 2 / (2 * sx ** 2) + (y - y0) ** 2 / (2 * sy ** 2)))

    def populate_dense_features(self, data) -> torch.Tensor:
        """
        vx, vy, yaw 값을 heatmap 형태로 확장 (GT 위치 중심의 가우시안)
        """
        tensor = torch.zeros((self.image_size, self.image_size, 3), dtype=torch.float32)
        x, y = np.meshgrid(np.arange(self.image_size), np.arange(self.image_size))
        x0 = int((data[0] + self.origin_offset) / self.pixelsize)
        y0 = int((data[1] + self.origin_offset) / self.pixelsize)

        if self.consider_free_paths:
            if data[0] < 0 or np.hypot(data[0], data[1]) > 6.4:
                return tensor  # 뒤/먼 경우 dense도 제로

        for i in range(3):
            tensor[:, :, i] = self.gaussian_2d(x, y, x0, y0, self.sx, self.sy, data[i + 2])
        return tensor

    def heatmap(self, data):
        """GT 위치 가우시안 히트맵 생성"""
        x, y = np.meshgrid(np.arange(self.image_size), np.arange(self.image_size))
        x0 = int((data[0] + self.origin_offset) / self.pixelsize)
        y0 = int((data[1] + self.origin_offset) / self.pixelsize)
        hm = self.gaussian_2d(x, y, x0, y0, self.sx, self.sy, 1)

        if self.consider_free_paths:
            if data[0] < 0 or np.hypot(data[0], data[1]) > 6.4:
                hm = np.zeros((self.image_size, self.image_size))

        return torch.tensor(hm, dtype=torch.float32)

    def preprocess(self, lidar_ranges):
        """
        라이다 극좌표 → 카르테시안 투영 → 격자화
        채널:
          [0] occupancy: 해당 픽셀에 하나라도 레이저가 들어오면 1
          [1] density  : 해당 픽셀에 누적된 레이저 점 개수
        """
        if not isinstance(lidar_ranges, torch.Tensor):
            lidar_ranges = torch.as_tensor(lidar_ranges, dtype=torch.float32)

        # cos/sin을 같은 디바이스로
        cos = self.cos.to(lidar_ranges.device)
        sin = self.sin.to(lidar_ranges.device)

        input_data = torch.zeros((self.feature_size, self.image_size, self.image_size),
                                 dtype=torch.float32, device=lidar_ranges.device)

        x = lidar_ranges * cos
        y = lidar_ranges * sin

        x_idx = ((x + self.origin_offset) / self.pixelsize).to(torch.int64)
        y_idx = ((y + self.origin_offset) / self.pixelsize).to(torch.int64)

        valid = (x_idx >= 0) & (x_idx < self.image_size) & (y_idx >= 0) & (y_idx < self.image_size)
        x_idx = x_idx[valid]
        y_idx = y_idx[valid]

        # occupancy / density
        input_data[0, y_idx, x_idx] = 1.0
        input_data[1, y_idx, x_idx] += 1.0

        # DataLoader는 CPU에서 돌기 때문에 CPU로 반환(안전)
        return input_data.cpu()

    def cartesian_to_pixel(self, x, y):
        px = int(x / self.pixelsize + self.image_size / 2)
        py = int(y / self.pixelsize + self.image_size / 2)
        return px, py

    def visualize(self, index, show_preprocessed=True, show_gt=True, show_raw=True):
        """
        시각화 유틸 (intensity 관련 플롯 제거)
        """
        config = [show_preprocessed, show_gt, show_raw]
        plot_rows = sum(1 for c in config if c)
        if plot_rows == 0:
            print("No plots selected!")
            return

        fig, axs = plt.subplots(plot_rows + 1, 3, figsize=(10, 15))
        sample = self.__getitem__(index)
        # dense 모드 여부에 따라 반환 튜플 길이가 다름
        if self.dense:
            input_t, gt, data, _, free = sample
        else:
            input_t, gt, data, free = sample

        if self.transform is not None and hasattr(self.transform, "transforms"):
            transform_names = ", ".join([t.__class__.__name__ for t in self.transform.transforms])
        else:
            transform_names = "None"

        # 정보 박스
        for j in range(3):
            axs[0, j].axis('off')
        axs[0, 0].set_title('Dataset Info')
        axs[0, 0].text(0, 0.3, f'Length: {self.len}'
                               f'\nPath: {self.dataset_path}'
                               f'\nTransforms: {transform_names}'
                               f'\n\nIndex: {index}'
                               f'\nFree track: {free}'
                               f'\nPixel size: {self.pixelsize}'
                               f'\nImage size: {self.image_size}'
                               f'\nGaussian σ: {self.sx}', fontsize=10)

        rowp = 1
        if show_preprocessed:
            axs[rowp, 0].set_title('Occupancy')
            axs[rowp, 0].imshow(input_t[0], cmap='plasma')
            axs[rowp, 1].imshow(input_t[1], cmap='plasma')
            axs[rowp, 1].set_title('Density')
            axs[rowp, 2].axis('off')
            for i in range(2):
                axs[rowp, i].axis('off')
                axs[rowp, i].scatter(self.image_size // 2, self.image_size // 2,
                                     label='Ego', color='g', s=10)
                px, py = self.cartesian_to_pixel(data[0], data[1])
                axs[rowp, i].scatter(px, py, label='GT Pos', color='r', s=10)
                axs[rowp, i].quiver(px, py, data[2], data[3], label='GT Vel', color='r', scale=20)
                yaw_deg = np.rad2deg(data[4])
                rect = patches.Rectangle((px - 2, py - 4), 8, 4, angle=yaw_deg, fill=False, color='r')
                axs[rowp, i].add_patch(rect)
            axs[rowp, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
            rowp += 1

        if show_gt:
            axs[rowp, 0].imshow(gt, cmap='plasma')
            axs[rowp, 0].set_title('GT heatmap')
            axs[rowp, 1].imshow(np.clip(input_t[0] + gt, 0, 1), cmap='plasma')
            axs[rowp, 1].set_title('Occupancy + GT heatmap')
            axs[rowp, 0].axis('off')
            axs[rowp, 1].axis('off')
            axs[rowp, 2].axis('off')
            rowp += 1

        if show_raw:
            axs[rowp, 0].axis('off'); axs[rowp, 1].axis('off'); axs[rowp, 2].axis('off')
            axs[rowp, 1].text(0.1, 0.5, "Raw lidar plot omitted (no intensity).\n"
                                        "Add logging in __getitem__ if needed.", fontsize=10)


class RandomRotation:
    """
    입력/GT를 동일 각도로 랜덤 회전.
    - 채널 수에 의존하지 않도록 구현 (입력 shape에서 자동 추론)
    """
    def __init__(self, angle=45, image_size=128):  # 기본 128로 변경
        self.angle = angle
        self.image_size = image_size

    def __call__(self, sample):
        input_t, heatmap, data = sample
        angle = random.uniform(-self.angle, self.angle)
        angle_rad = -math.radians(angle)

        # 입력 각 채널 회전
        rotated_list = []
        for i in range(input_t.shape[0]):
            pil_img = transforms.functional.to_pil_image(input_t[i])
            rot_img = transforms.functional.rotate(pil_img, angle)
            rotated_list.append(transforms.ToTensor()(rot_img))
        input_rot = torch.stack(rotated_list, dim=0)

        # heatmap 회전
        hm_img = transforms.functional.to_pil_image(heatmap)
        hm_rot = transforms.ToTensor()(transforms.functional.rotate(hm_img, angle))

        # (x, y), (vx, vy), yaw 회전
        R = torch.FloatTensor([[np.cos(angle_rad), -np.sin(angle_rad)],
                               [np.sin(angle_rad),  np.cos(angle_rad)]])
        data[0:2] = torch.matmul(R, data[0:2])
        data[2:4] = torch.matmul(R.T, data[2:4])
        data[4] = (data[4] - math.radians(angle)) % (2 * math.pi)
        if data[4] > math.pi:
            data[4] -= 2 * math.pi

        return input_rot.view(input_rot.shape[0], self.image_size, self.image_size), \
               hm_rot.view(self.image_size, self.image_size), \
               data.view(5)


class RandomFlip:
    """
    좌우 플립 (y축 대칭)
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        input_t, heatmap, data = sample
        if random.random() < self.p:
            # 입력: 가로(=W) 플립 → tensor dim=2 기준
            input_t = torch.flip(input_t, [2])
            # heatmap은 가로(W) 플립
            heatmap = torch.flip(heatmap, [1])

            # 좌표(y) 및 속도(vy) 부호 반전, yaw 반전
            data[1] = -data[1]
            data[3] = -data[3]
            data[4] = -data[4]
        return input_t, heatmap, data


# # -*- coding: utf-8 -*-
# import torch
# import numpy as np
# import pandas as pd
# import csv
# from torch.utils.data import Dataset
# from torchvision import transforms
# import random
# import math
# import os
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches


# class CenterSpeedDataset(Dataset):
#     """
#     TinyCenterSpeed Dataset (intensity 채널 제거판)
#     - 입력: 2프레임 × (occupancy, density) = 4채널
#     - GT:
#         heatmap: (H, W)
#         dense:   (H, W, 3)  # vx, vy, yaw 값을 heatmap 형태로 확장
#         data:    (5,)       # x, y, vx, vy, yaw
#     """
#     def __init__(self, dataset_path, transform=None, dense=False):
#         self.dataset_path = dataset_path
#         self.transform = transform
#         self.use_heatmaps = True
#         self.dense = dense
#         self.consider_free_paths = True

#         # 해상도/격자 설정
#         self.pixelsize = 0.1     # [m/pixel]
#         self.image_size = 128      # H=W
#         self.feature_size = 2     # ✨ intensity 제거 → (occupancy, density)만 사용
#         self.seq_len = 2          # 프레임 수

#         self.origin_offset = (self.image_size // 2) * self.pixelsize
#         self.sx = self.sy = 5     # heatmap 가우시안 표준편차 (pixel)
#         self.len = None
#         self.number_of_sets = None

#         # 라이다 각도 미리 계산 (FOV -135°~+135°, 약 0.25° 간격) 라이다 데이타 1081개 (실제 차)
#         # self.cos = np.cos(np.arange(-2.356194496154785, 2.356194496154785, 0.004363323096185923))
#         # self.sin = np.sin(np.arange(-2.356194496154785, 2.356194496154785, 0.004363323096185923))

#         # num_beams = 1080
#         # self.angles = np.linspace(-2.356194496154785, 2.356194496154785, num_beams)
#         # self.cos = np.cos(self.angles)
#         # self.sin = np.sin(self.angles)


#         # __init__ 안의 라이다 각도 부분 교체
#         num_beams = 1080
#         self.angles = np.linspace(-2.356194496154785, 2.356194496154785, num_beams).astype(np.float32)
#         self.cos = torch.from_numpy(np.cos(self.angles))
#         self.sin = torch.from_numpy(np.sin(self.angles))

#         # σ 추천값 반영 (40cm @ 0.1m/px → r=2px)
#         self.sx = self.sy = 2.0  # 기존 5 → 2.0

#         self.setup()

#     def setup(self):
#         """디렉터리 내 CSV를 스캔해 전체 길이/인덱스 맵 구성"""
#         self.file_paths = [os.path.join(self.dataset_path, f)
#                            for f in os.listdir(self.dataset_path) if f.endswith('.csv')]
#         self.len = 0
#         self.file_indices = []
#         num_rows_per_file = []
#         for file_path in self.file_paths:
#             # 헤더 1줄 + (seq_len-1) 만큼 시퀀스 슬라이딩 불가 구간 제외
#             num_rows = sum(1 for _ in open(file_path)) - 1 - (self.seq_len - 1)
#             self.file_indices.append((self.len, self.len + num_rows))
#             self.len += num_rows
#             num_rows_per_file.append(num_rows)

#         for path in self.file_paths:
#             print("Reading file: ", path)
#             print("Entries     : ", num_rows_per_file[self.file_paths.index(path)])

#         print("Total rows : ", self.len)
#         print("File index : ", self.file_indices)

#     def change_pixel_size(self, pixelsize: float):
#         self.pixelsize = pixelsize
#         self.origin_offset = (self.image_size // 2) * self.pixelsize
#         print("Pixel size   ->", self.pixelsize)
#         print("Origin offset->", self.origin_offset)

#     def change_image_size(self, image_size: int):
#         self.image_size = int(image_size)
#         self.origin_offset = (self.image_size // 2) * self.pixelsize
#         print("Image size   ->", self.image_size)
#         print("Origin offset->", self.origin_offset)

#     def __getitem__(self, index):
#         """
#         반환:
#           input_data: (seq_len*feature_size, H, W)  # 여기선 (4, 64, 64)
#           heatmap   : (H, W)
#           data      : (5,)
#           [dense 있을 때] dense_features: (H, W, 3)
#           free      : bool (장애물 없는 자유 경로 여부)
#         """
#         free = False

#         # 이 인덱스가 속한 파일 찾기
#         file_index = next(i for i, (start, end) in enumerate(self.file_indices) if start <= index < end)
#         row_index = index - self.file_indices[file_index][0]

#         seq_data = []
#         # intensities 컬럼은 읽지만 채널로 사용하지 않음(무시)
#         df = pd.read_csv(
#             self.file_paths[file_index],
#             skiprows=row_index,
#             nrows=self.seq_len,
#             header=None,
#             names=['lidar', 'intensities', 'x', 'y', 'vx', 'vy', 'yaw']
#         )
#         if len(df) == 0:
#             raise IndexError

#         for i in range(self.seq_len):
#             df.loc[i, 'lidar'] = df.loc[i, 'lidar'].replace('(', '').replace(')', '')
#             # intensities는 사용하지 않지만 CSV 포맷 유지 위해 정리만
#             df.loc[i, 'intensities'] = df.loc[i, 'intensities'].replace('(', '').replace(')', '')

#             row = df.iloc[i]
#             lidar_ranges = torch.tensor(
#                 np.fromstring(df.loc[i, 'lidar'], dtype=float, sep=', '),
#                 dtype=torch.float32
#             )
#             # occupancy + density만 생성
#             seq_data.append(self.preprocess(lidar_ranges))

#             if i == self.seq_len - 1:
#                 data = torch.tensor(row[2:].values.astype(float), dtype=torch.float32)
#                 self.data_for_plot = data.numpy().copy()
#                 heatmap = self.heatmap(data)

#         # (seq_len * feature_size, H, W) = (4, 64, 64)
#         input_data = torch.stack(seq_data).view(self.seq_len * self.feature_size,
#                                                 self.image_size, self.image_size)

#         # free-path 규칙: 상대가 뒤에 있거나 너무 멀면 free
#         if data[0] < 0 or np.sqrt(data[0] ** 2 + data[1] ** 2) > 6.4:
#             free = True

#         # 변환 적용
#         if self.transform:
#             input_data, heatmap, data = self.transform((input_data, heatmap, data))

#         if self.dense:
#             dense_features = self.populate_dense_features(data=data)
#             return (input_data,
#                     heatmap.view(self.image_size, self.image_size),
#                     data.view(5),
#                     dense_features,
#                     free)

#         return (input_data,
#                 heatmap.view(self.image_size, self.image_size),
#                 data.view(5),
#                 free)

#     def __len__(self):
#         if self.len is not None:
#             return self.len
#         # 일반 파일 단일경로로 주는 경우만 대비
#         with open(self.dataset_path, 'r') as f:
#             self.len = sum(1 for _ in csv.reader(f))
#         return self.len

#     @staticmethod
#     def gaussian_2d(x, y, x0, y0, sx, sy, A):
#         return A * np.exp(-((x - x0) ** 2 / (2 * sx ** 2) + (y - y0) ** 2 / (2 * sy ** 2)))

#     def populate_dense_features(self, data) -> torch.Tensor:
#         """
#         vx, vy, yaw 값을 heatmap 형태로 확장 (GT 위치 중심의 가우시안)
#         """
#         tensor = torch.zeros((self.image_size, self.image_size, 3), dtype=torch.float32)
#         x, y = np.meshgrid(np.arange(self.image_size), np.arange(self.image_size))
#         x0 = int((data[0] + self.origin_offset) / self.pixelsize)
#         y0 = int((data[1] + self.origin_offset) / self.pixelsize)

#         if self.consider_free_paths:
#             if data[0] < 0 or np.sqrt(data[0] ** 2 + data[1] ** 2) > 6.4:
#                 # 뒤/먼 경우 dense도 제로
#                 return tensor

#         for i in range(3):
#             tensor[:, :, i] = self.gaussian_2d(x, y, x0, y0, self.sx, self.sy, data[i + 2])
#         return tensor

#     def heatmap(self, data):
#         """GT 위치 가우시안 히트맵 생성"""
#         x, y = np.meshgrid(np.arange(self.image_size), np.arange(self.image_size))
#         x0 = int((data[0] + self.origin_offset) / self.pixelsize)
#         y0 = int((data[1] + self.origin_offset) / self.pixelsize)
#         hm = self.gaussian_2d(x, y, x0, y0, self.sx, self.sy, 1)

#         if self.consider_free_paths:
#             if data[0] < 0 or np.sqrt(data[0] ** 2 + data[1] ** 2) > 6.4:
#                 hm = np.zeros((self.image_size, self.image_size))

#         return torch.tensor(hm, dtype=torch.float32)

#     def preprocess(self, lidar_ranges: torch.Tensor):
#         """
#         라이다 극좌표 → 카르테시안 투영 → 격자화
#         채널:
#           [0] occupancy: 해당 픽셀에 하나라도 레이저가 들어오면 1
#           [1] density  : 해당 픽셀에 누적된 레이저 점 개수
#         """
#         input_data = torch.zeros((self.feature_size, self.image_size, self.image_size), dtype=torch.float32)

#         x = lidar_ranges * self.cos
#         y = lidar_ranges * self.sin
#         # x_idx = ((x + self.origin_offset) / self.pixelsize).astype(np.int64)
#         # y_idx = ((y + self.origin_offset) / self.pixelsize).astype(np.int64)
        
#         x_idx = ((x + self.origin_offset) / self.pixelsize).long()
#         y_idx = ((y + self.origin_offset) / self.pixelsize).long()

#         x_idx = torch.as_tensor(x_idx, dtype=torch.int64)
#         y_idx = torch.as_tensor(y_idx, dtype=torch.int64)

#         valid = (x_idx >= 0) & (x_idx < self.image_size) & (y_idx >= 0) & (y_idx < self.image_size)
#         x_idx = x_idx[valid]
#         y_idx = y_idx[valid]

#         # occupancy
#         input_data[0, y_idx, x_idx] = 1.0
#         # density (누적 카운트)
#         input_data[1, y_idx, x_idx] += 1.0

#         return input_data

#     def cartesian_to_pixel(self, x, y):
#         px = int(x / self.pixelsize + self.image_size / 2)
#         py = int(y / self.pixelsize + self.image_size / 2)
#         return px, py

#     def visualize(self, index, show_preprocessed=True, show_gt=True, show_raw=True):
#         """
#         시각화 유틸 (intensity 관련 플롯 제거)
#         """
#         config = [show_preprocessed, show_gt, show_raw]
#         plot_rows = sum(1 for c in config if c)
#         if plot_rows == 0:
#             print("No plots selected!")
#             return

#         fig, axs = plt.subplots(plot_rows + 1, 3, figsize=(10, 15))
#         sample = self.__getitem__(index)
#         # dense 모드 여부에 따라 반환 튜플 길이가 다름
#         if self.dense:
#             input_t, gt, data, _, free = sample
#         else:
#             input_t, gt, data, free = sample

#         if self.transform is not None and hasattr(self.transform, "transforms"):
#             transform_names = ", ".join([t.__class__.__name__ for t in self.transform.transforms])
#         else:
#             transform_names = "None"

#         # 정보 박스
#         for j in range(3):
#             axs[0, j].axis('off')
#         axs[0, 0].set_title('Dataset Info')
#         axs[0, 0].text(0, 0.3, f'Length: {self.len}'
#                                f'\nPath: {self.dataset_path}'
#                                f'\nTransforms: {transform_names}'
#                                f'\n\nIndex: {index}'
#                                f'\nFree track: {free}'
#                                f'\nPixel size: {self.pixelsize}'
#                                f'\nImage size: {self.image_size}'
#                                f'\nGaussian σ: {self.sx}', fontsize=10)

#         rowp = 1
#         if show_preprocessed:
#             axs[rowp, 0].set_title('Occupancy')
#             axs[rowp, 0].imshow(input_t[0], cmap='plasma')
#             axs[rowp, 1].imshow(input_t[1], cmap='plasma')
#             axs[rowp, 1].set_title('Density')
#             axs[rowp, 2].axis('off')
#             for i in range(2):
#                 axs[rowp, i].axis('off')
#                 axs[rowp, i].scatter(self.image_size // 2, self.image_size // 2,
#                                      label='Ego', color='g', s=10)
#                 px, py = self.cartesian_to_pixel(data[0], data[1])
#                 axs[rowp, i].scatter(px, py, label='GT Pos', color='r', s=10)
#                 axs[rowp, i].quiver(px, py, data[2], data[3], label='GT Vel', color='r', scale=20)
#                 yaw_deg = np.rad2deg(data[4])
#                 rect = patches.Rectangle((px - 2, py - 4), 8, 4, angle=yaw_deg, fill=False, color='r')
#                 axs[rowp, i].add_patch(rect)
#             axs[rowp, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
#             rowp += 1

#         if show_gt:
#             axs[rowp, 0].imshow(gt, cmap='plasma')
#             axs[rowp, 0].set_title('GT heatmap')
#             axs[rowp, 1].imshow(np.clip(input_t[0] + gt, 0, 1), cmap='plasma')
#             axs[rowp, 1].set_title('Occupancy + GT heatmap')
#             axs[rowp, 0].axis('off')
#             axs[rowp, 1].axis('off')
#             axs[rowp, 2].axis('off')
#             rowp += 1

#         if show_raw:
#             # 원시 라이다 범위만 간단히 표기 (intensity 플롯 제거)
#             # 직전 __getitem__에서 저장된 lidar_ranges는 없으므로 생략하거나 필요 시 별도 저장 로직 추가 가능
#             axs[rowp, 0].axis('off'); axs[rowp, 1].axis('off'); axs[rowp, 2].axis('off')
#             axs[rowp, 1].text(0.1, 0.5, "Raw lidar plot omitted (no intensity).\n"
#                                         "Add logging in __getitem__ if needed.", fontsize=10)


# class RandomRotation:
#     """
#     입력/GT를 동일 각도로 랜덤 회전.
#     - 채널 수에 의존하지 않도록 구현 (입력 shape에서 자동 추론)
#     """
#     def __init__(self, angle=45, image_size=64):
#         self.angle = angle
#         self.image_size = image_size

#     def __call__(self, sample):
#         input_t, heatmap, data = sample
#         angle = random.uniform(-self.angle, self.angle)
#         angle_rad = -math.radians(angle)

#         # 입력 각 채널 회전
#         rotated_list = []
#         for i in range(input_t.shape[0]):
#             pil_img = transforms.functional.to_pil_image(input_t[i])
#             rot_img = transforms.functional.rotate(pil_img, angle)
#             rotated_list.append(transforms.ToTensor()(rot_img))
#         input_rot = torch.stack(rotated_list, dim=0)

#         # heatmap 회전
#         hm_img = transforms.functional.to_pil_image(heatmap)
#         hm_rot = transforms.ToTensor()(transforms.functional.rotate(hm_img, angle))

#         # (x, y), (vx, vy), yaw 회전
#         R = torch.FloatTensor([[np.cos(angle_rad), -np.sin(angle_rad)],
#                                [np.sin(angle_rad),  np.cos(angle_rad)]])
#         data[0:2] = torch.matmul(R, data[0:2])
#         data[2:4] = torch.matmul(R.T, data[2:4])
#         data[4] = (data[4] - math.radians(angle)) % (2 * math.pi)
#         if data[4] > math.pi:
#             data[4] -= 2 * math.pi

#         return input_rot.view(input_rot.shape[0], self.image_size, self.image_size), \
#                hm_rot.view(self.image_size, self.image_size), \
#                data.view(5)


# class RandomFlip:
#     """
#     좌우 플립 (y축 대칭)
#     """
#     def __init__(self, p=0.5):
#         self.p = p

#     def __call__(self, sample):
#         input_t, heatmap, data = sample
#         if random.random() < self.p:
#             # 입력: 가로(=W) 플립 → tensor dim=2 기준
#             input_t = torch.flip(input_t, [2])
#             # heatmap은 세로/가로 중 선택 — 여기서는 X축 기준 뒤집힘을 고려하여 W축(가로)만 플립
#             heatmap = torch.flip(heatmap, [1])

#             # 좌표(y) 및 속도(vy) 부호 반전, yaw 반전
#             data[1] = -data[1]
#             data[3] = -data[3]
#             data[4] = -data[4]
#         return input_t, heatmap, data


# import torch
# import numpy as np
# import pandas as pd
# import csv
# from torch.utils.data import Dataset, DataLoader, random_split
# from torchvision import transforms
# import random
# import math
# import os
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches


# class CenterSpeedDataset(Dataset):
#     '''
#     Dataset class for the CenterSpeed dataset.
#     '''
#     def __init__(self, dataset_path, transform=None, dense=False):
#         self.dataset_path = dataset_path
#         self.transform = transform
#         self.use_heatmaps = True
#         self.dense = dense
#         self.consider_free_paths = True
#         self.pixelsize = 0.08#size of a pixel in meters
#         self.image_size = 64 #size of the image for preprocessing
#         self.feature_size = 3 #number of features in the preprocessed data
#         self.origin_offset = (self.image_size//2) * self.pixelsize
#         self.sx = self.sy = 5 #standard deviation of the gaussian peaks
#         self.len = None
#         self.seq_len = 2 #number of frames in a sequence
#         self.number_of_sets = None
#         self.cos = np.cos(np.arange(-2.356194496154785, 2.356194496154785 ,0.004363323096185923))
#         self.sin = np.sin(np.arange(-2.356194496154785, 2.356194496154785 ,0.004363323096185923))
#         self.setup()

#     def setup(self):
#         '''
#         Sets up the dataset by reading the files and determining the number of rows in each file.
#         '''
#         self.file_paths = [os.path.join(self.dataset_path, f) for f in os.listdir(self.dataset_path) if f.endswith('.csv')]
#         self.len = 0
#         self.file_indices = []
#         num_rows_per_file = []
#         for file_path in self.file_paths:
#                 num_rows = sum(1 for row in open(file_path))- 1 -(self.seq_len-1) #subtract 2 because of the header and the last row
#                 self.file_indices.append((self.len, self.len+num_rows))
#                 self.len += num_rows
#                 num_rows_per_file.append(num_rows)
#         for path in self.file_paths:
#             print("Reading the following files: ", path)
#             print("Number of entries: ", num_rows_per_file[self.file_paths.index(path)])

#         print("Number of rows: ", self.len)
#         print("File indices: ", self.file_indices)

#     def change_pixel_size(self, pixelsize):
#         '''
#         Changes the pixel size and the origin offset accordingly.

#         Args:
#             pixelsize (int): New pixel size in meters.
#         '''
#         self.pixelsize = pixelsize
#         self.origin_offset = (self.image_size//2) * self.pixelsize
#         print("Pixel size changed to: ", self.pixelsize)
#         print("Origin offset changed to: ", self.origin_offset)

#     def change_image_size(self, image_size):
#         '''
#         Changes the image size and the origin offset accordingly.

#         Args:
#             image_size (int): New image size in pixels.
#         '''
#         self.image_size = int(image_size)
#         self.origin_offset = (self.image_size//2) * self.pixelsize
#         print("Image size changed to: ", self.image_size)
#         print("Origin offset changed to: ", self.origin_offset)

#     def __getitem__(self, index):
#         '''
#         Returns the preprocessed data and the ground truth data for a given index.

#         Args:
#             index: Index of the data to be returned.

#         Returns:
#             input_data: Preprocessed data in the form of a tensor of size (3, 64, 64).
#             heatmap: Ground truth heatmap in the form of a tensor of size (64, 64).
#             data: Ground truth data in the form of a tensor of size (5).
#             free: Boolean indicating whether the path is free or not.
#             '''
#         free = False
#         # Determine which file the data should come from
#         file_index = next(i for i, (start, end) in enumerate(self.file_indices) if start <= index < end)
#         row_index = index - self.file_indices[file_index][0]

#         seq_data = []
#         df = pd.read_csv(self.file_paths[file_index], skiprows=row_index, nrows=self.seq_len, header=None, names=['lidar','intensities','x','y','vx','vy','yaw'])
#         if len(df) == 0:
#             raise IndexError
#         for i in range(self.seq_len):
#             df.loc[i, 'lidar'] = df.loc[i, 'lidar'].replace('(', '').replace(')', '')
#             df.loc[i, 'intensities'] = df.loc[i, 'intensities'].replace('(', '').replace(')', '')
#             row = df.iloc[i]
#             self.lidar_data = torch.tensor(np.fromstring(df.loc[i, 'lidar'], dtype=float, sep=', '), dtype=torch.float32)
#             intensities = torch.tensor(np.fromstring(df.loc[i, 'intensities'], dtype=float, sep=','), dtype=torch.float32)
#             try:
#                 self.intensities = (intensities - intensities.min()) / (intensities.max() - intensities.min())  # normalize intensities
#             except:
#                 print("Intensities: ", intensities)
#                 print("Row: ", row)
#                 print("Index: ", index)
#             seq_data.append(self.preprocess(self.lidar_data, self.intensities))
#             if i == self.seq_len - 1:
#                 data = torch.tensor(row[2:].values.astype(float), dtype=torch.float32)
#                 self.data_for_plot = data.numpy().copy()
#                 heatmap = self.heatmap(data)
#         input_data = torch.stack([item for item in seq_data]).view(self.seq_len*3,self.image_size,self.image_size)

#         if data[0] < 0 or np.sqrt(data[0]**2+ data[1]**2) > 3:
#             free = True

#         if self.transform:
#             input_data, heatmap, data = self.transform((input_data, heatmap, data))

#         if self.dense:
#             print(f'Using dense features with data: {data}')
#             dense_features = self.populate_dense_features(data=data)
#             return input_data.view(self.feature_size*self.seq_len, self.image_size, self.image_size), heatmap.view(self.image_size, self.image_size), data.view(5), dense_features, free

#         return input_data.view(self.feature_size*self.seq_len, self.image_size, self.image_size), heatmap.view(self.image_size, self.image_size), data.view(5), free


#     def __len__(self):
#         '''
#         Returns the length of the dataset.
#         '''
#         if self.len is not None:
#             return self.len
#         else:
#             with open(self.dataset_path, 'r') as f:
#                 self.len = sum(1 for row in csv.reader(f))
#                 return self.len

#     def gaussian_2d(self, x, y, x0, y0, sx, sy, A):
#         '''
#         2D Gaussian function.

#         Args:
#             x: x-coordinate
#             y: y-coordinate
#             x0: x-coordinate of the peak
#             y0: y-coordinate of the peak
#             sx: standard deviation in x
#             sy: standard deviation in y
#             A: amplitude'''
#         return A * np.exp(-((x - x0)**2 / (2 * sx**2) + (y - y0)**2 / (2 * sy**2)))

#     def populate_dense_features(self, data) -> torch.Tensor:
#         '''
#         Populates a tensor with dense speed and orientation values.

#         Args:
#             x: x-coordinate of the peak
#             y: y-coordinate of the peak
#             values: List of values to be populated in the tensor.
#         '''
#         tensor = torch.zeros((self.image_size, self.image_size, 3), dtype=torch.float32)
#         x,y = np.meshgrid(np.arange(self.image_size), np.arange(self.image_size))
#         x0 = int((data[0] + self.origin_offset) / self.pixelsize)
#         y0 = int((data[1] + self.origin_offset) / self.pixelsize)
#         print(f'Data length: {len(data)}')
#         for i in range(3):
#             tensor[:,:,i] = self.gaussian_2d(x, y, x0, y0, self.sx, self.sy, data[i+2])

#             if self.consider_free_paths:
#                 if data[0] < 0 or np.sqrt(data[0]**2+ data[1]**2) > 3:#the other car is behind us, no peak in the heatmap
#                     tensor = torch.zeros((self.image_size, self.image_size, 3), dtype=torch.float32)
#                     print(f'Car Behind, setting zero')
#                     return tensor


#         return tensor



#     def heatmap(self, data):
#         '''
#         Creates a heatmap from the ground truth data.
#         '''
#         self.heatmaps = torch.zeros(self.image_size, self.image_size)
#         x,y = np.meshgrid(np.arange(self.image_size), np.arange(self.image_size))
#         x0 = int((data[0] + self.origin_offset) / self.pixelsize)
#         y0 = int((data[1] + self.origin_offset) / self.pixelsize)
#         heatmap = self.gaussian_2d(x, y, x0, y0, self.sx, self.sy, 1)
#         if self.consider_free_paths:
#             if data[0] < 0 or np.sqrt(data[0]**2+ data[1]**2) > 3:#the other car is behind us, no peak in the heatmap
#                 heatmap = np.zeros((self.image_size, self.image_size))
#         heatmap = torch.tensor(heatmap, dtype=torch.float32)
#         return heatmap

#     def preprocess(self, lidar_data, intensities):
#         '''
#         Preprocesses the data. Convert polar coordinates to cartesian coordinates and discretize into an image.
#         Creates 3 feature maps: one for the occupancy, one for the intensity and one for the number of points in a pixel.

#         Args:
#             lidar_data: Lidar data in the form of a tensor of size (n).
#             intensities: Intensity data in the form of a tensor of size (n).

#         Returns:
#             input_data: Preprocessed data in the form of a tensor of size (3, 64, 64).

#         '''
#         self.use_heatmaps = True#use heatmaps for training after preprocessing
#         #preprocess the lidar data
#         input_data = torch.zeros((self.feature_size, self.image_size, self.image_size), dtype=torch.float32)
#         x = lidar_data * self.cos
#         y = lidar_data * self.sin
#         x_coord = ((x + self.origin_offset) / self.pixelsize)
#         y_coord = ((y + self.origin_offset) / self.pixelsize)
#         x_coord = x_coord.to(torch.int)
#         y_coord = y_coord.to(torch.int)
#         valid_indices = (x_coord >= 0) & (x_coord < self.image_size) & (y_coord >= 0) & (y_coord < self.image_size)
#         x_coord = x_coord[valid_indices]
#         y_coord = y_coord[valid_indices]
#         input_data[0,y_coord, x_coord] = 1 #set the pixel to occupied
#         input_data[1,y_coord, x_coord] = torch.maximum(input_data[ 1,y_coord,x_coord], intensities[valid_indices])#store the maximum intensity value in the pixel
#         input_data[2,y_coord, x_coord] +=1 #count the number of points in the pixel

#         return input_data

#     def cartesian_to_pixel(self, x, y):
#         '''
#         Converts cartesian coordinates to pixel coordinates.
#         '''
#         pixel_x = int(x / self.pixelsize + self.image_size / 2)
#         pixel_y = int(y / self.pixelsize + self.image_size / 2)
#         return pixel_x, pixel_y


#     def visualize(self, index, show_preprocessed=True, show_gt=True, show_raw=True):
#         '''
#         Visualizes the data for a given index.

#         Args:
#             index: Index of the data to be visualized.
#             show_preprocessed: Boolean indicating whether to show the preprocessed data.
#             show_gt: Boolean indicating whether to show the ground truth data.
#             show_raw: Boolean indicating whether to show the raw data.
#         '''
#         config = [show_preprocessed, show_gt, show_raw]
#         plot_rows = 0
#         for c in config:
#             if c:
#                 plot_rows += 1
#         if plot_rows == 0:
#             print("No plots selected!")
#             return

#         fig, axs = plt.subplots(plot_rows+1, 3, figsize=(10, 15))
#         input, gt, data, free = self.__getitem__(index)
#         if self.transform is not None:
#             transform_names = ', '.join([t.__class__.__name__ for t in self.transform.transforms])
#         else:
#             transform_names = 'None'

#         axs[0,0].axis('off')
#         axs[0,1].axis('off')
#         axs[0,2].axis('off')
#         axs[0,0].set_title('Dataset Info')
#         axs[0,0].text(0, 0.3, f'Length of dataset: {self.len}\
#                                 \nPath: {self.dataset_path}\
#                                 \nTransforms: {transform_names}\
#                                 \n\nIndex: {index}\
#                                 \nFree track: {free}\
#                                 \nPixel size: {self.pixelsize}\
#                                 \nImage size: {self.image_size}\
#                                 \nGaussian radius: {self.sx}', fontsize=10)

#         plot_nr = 1
#         if show_preprocessed:
#             axs[plot_nr, 0].set_title('Occupancy')
#             axs[plot_nr, 0].imshow(input[0], cmap='plasma')
#             axs[plot_nr, 1].imshow(input[1], cmap='plasma')
#             axs[plot_nr, 1].set_title('Intensity')
#             axs[plot_nr, 2].imshow(input[2], cmap='plasma')
#             axs[plot_nr, 2].set_title('Density')
#             for i in range(3):
#                 axs[plot_nr, i].axis('off')
#                 axs[plot_nr,i].scatter(self.image_size//2,self.image_size//2, label='Ego Position', color='g')
#                 x,y = self.cartesian_to_pixel(data[0],data[1])
#                 axs[plot_nr,i].scatter(x,y, label='GT Position', color='r')
#                 axs[plot_nr,i].quiver(x,y ,data[2],data[3], label='GT Velocity', color='r')
#                 yaw_degrees = np.rad2deg(data[4])
#                 rectangle = patches.Rectangle((x-2, y-4), 8, 4, angle=yaw_degrees, fill=False, color='r')
#                 axs[plot_nr, i].add_patch(rectangle)
#             axs[plot_nr, 2].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
#             plot_nr += 1

#         if show_gt:
#             axs[plot_nr, 0].imshow(gt, cmap='plasma')
#             axs[plot_nr, 0].set_title('GT heatmap')
#             axs[plot_nr, 1].imshow(np.clip(input[0]+gt, 0, 1), cmap='plasma')
#             axs[plot_nr, 1].set_title('Occupancy + GT heatmap')
#             axs[plot_nr, 0].axis('off')
#             axs[plot_nr, 1].axis('off')
#             axs[plot_nr, 2].axis('off')
#             plot_nr += 1

#         if show_raw:
#             axs[plot_nr, 0].plot(self.lidar_data)
#             axs[plot_nr, 0].set_title('Raw lidar ranges')
#             axs[plot_nr, 1].plot(self.intensities)
#             axs[plot_nr, 1].set_title('Raw lidar intensities')
#             x = self.lidar_data * self.cos
#             y = self.lidar_data * self.sin
#             axs[plot_nr,2].scatter(x, y, s=0.1, label='Scans', alpha=self.intensities)
#             axs[plot_nr,2].scatter(self.data_for_plot[0], self.data_for_plot[1], color='r', label='GT-Pos')
#             axs[plot_nr,2].text(self.data_for_plot[0], self.data_for_plot[1],'GT-Pos')
#             # Adjusting view, focusing on GT-position
#             dx = dy = 2
#             axs[plot_nr,2].set_xlim(self.data_for_plot[0] - dx, self.data_for_plot[0] + dx)
#             axs[plot_nr,2].set_ylim(self.data_for_plot[1] - dy,self.data_for_plot[1] + dy)
#             axs[plot_nr,2].set_xlabel('X coordinate')
#             axs[plot_nr,2].set_ylabel('Y coordinate')
#             axs[plot_nr, 2].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
#             axs[plot_nr, 2].set_title('Raw lidar data')





# class RandomRotation:
#     '''
#     Randomly rotates the input data and the ground truth data.
#     '''
#     def __init__(self, angle=45, image_size=64, feature_size=6):
#         self.angle = angle
#         self.image_size = image_size
#         self.feature_size = feature_size

#     def __call__(self, sample):
#         input, heatmap, data = sample
#         angle = random.uniform(-self.angle, self.angle)
#         angle_rad = -math.radians(angle)
#         #print("THis was rotated by: ", angle)
#         input_rotated = []
#         for i in range(input.shape[0]):
#             input_pil = transforms.functional.to_pil_image(input[i])
#             input_rotated_tensor = transforms.ToTensor()(transforms.functional.rotate(input_pil, angle))
#             input_rotated.append(input_rotated_tensor)
#         input = torch.stack(input_rotated, dim = 1)


#         heatmap_image = transforms.functional.to_pil_image(heatmap)
#         rotated_hm_image = transforms.functional.rotate(heatmap_image, angle)
#         heatmap = transforms.ToTensor()(rotated_hm_image)

#         rotation_matrix = torch.FloatTensor([[np.cos(angle_rad), -np.sin(angle_rad)],
#                                         [np.sin(angle_rad), np.cos(angle_rad)]])

#         # Apply the rotation
#         data[0:2] = torch.matmul(rotation_matrix, data[0:2])
#         data[2:4] = torch.matmul(rotation_matrix.T, data[2:4])
#         data[4] = (data[4] - math.radians(angle))% (2*math.pi)
#         if data[4] > math.pi:
#             data[4] -= 2*math.pi

#         return input.view(self.feature_size,self.image_size,self.image_size), heatmap.view(self.image_size,self.image_size), data.view(5)

# class RandomFlip:
#     '''
#     Randomly flips the input data and the ground truth data.
#     '''
#     def __init__(self, p=0.5):
#         self.p = p

#     def __call__(self, sample):
#         input, heatmap, data = sample
#         if random.random() < self.p:
#             #print("This was flipped")
#             input = torch.flip(input, [1])
#             heatmap = torch.flip(heatmap, [0])
#             data[1] = -data[1]
#             data[3] = -data[3]
#             data[4] = -data[4]
#         return input, heatmap, data


# ################OLD IMPLEMENTATIONS####################

# class LidarDatasetOD(Dataset):
#     '''V1, Not used anymore'''
#     def __init__(self, dataset_path):
#         self.dataset_path = dataset_path
#         self.use_heatmaps = True
#         self.pixelsize = 0.025#size of a pixel in meters, was 0.015 i think this bigger makes more sense for FOV
#         self.image_size = 256 #size of the image for preprocessing
#         self.feature_size = 3 #number of features in the preprocessed data
#         self.origin_offset = (self.image_size//2) * self.pixelsize
#         self.sx = self.sy = 5 #standard deviation of the gaussian peaks
#         self.len = None


#     def __getitem__(self, index):
#         df = pd.read_csv(self.dataset_path, skiprows=index-1, nrows=1, header=None, names=['lidar','intensities','x','y','vx','vy','yaw'])
#         if len(df) == 0:
#             raise IndexError
#         df.loc[0, 'lidar'] = df.loc[0, 'lidar'].replace('(', '').replace(')', '')
#         df.loc[0, 'intensities'] = df.loc[0, 'intensities'].replace('(', '').replace(')', '')
#         row = df.iloc[0]
#         lidar_data = torch.tensor(np.fromstring(df.loc[0, 'lidar'], dtype=float, sep=', '), dtype=torch.float32)
#         intensities = torch.tensor(np.fromstring(df.loc[0, 'intensities'], dtype=float, sep=','), dtype=torch.float32)
#         #print(len(intensities))
#         intensities = (intensities - intensities.min()) / (intensities.max() - intensities.min())  # normalize intensities
#         data = torch.tensor(row[2:].values.astype(float), dtype=torch.float32)

#         return self.preprocess(lidar_data, intensities, data)

#     def __len__(self):
#         if self.len is not None:
#             return self.len
#         else:
#             with open(self.dataset_path, 'r') as f:
#                 self.len = sum(1 for row in csv.reader(f))
#                 return self.len

#     def gaussian_2d(self, x, y, x0, y0, sx, sy, A):
#         '''
#         2D Gaussian function.

#         Args:
#             x: x-coordinate
#             y: y-coordinate
#             x0: x-coordinate of the peak
#             y0: y-coordinate of the peak
#             sx: standard deviation in x
#             sy: standard deviation in y
#             A: amplitude'''
#         return A * np.exp(-((x - x0)**2 / (2 * sx**2) + (y - y0)**2 / (2 * sy**2)))


#     def preprocess(self, lidar_data, intensities, data):
#         '''
#         Preprocesses the data. Convert polar coordinates to cartesian coordinates and discretize into a 256x256 grid.
#         Stores these grids in a new tensor.
#         Completely vectorized, efficient asf!
#         Does it make sense to put the origin in the middle of the grid?
#         Maybe it is better to put it in the bottom left corner? Or closer to the corner?
#         '''

#         self.use_heatmaps = True#use heatmaps for training after preprocessing
#         #preprocess the lidar data
#         input_data = torch.zeros((self.feature_size, self.image_size, self.image_size), dtype=torch.float32)
#         x = lidar_data * np.cos(np.arange(-2.356194496154785, 2.356194496154785 ,0.004363323096185923))
#         y = lidar_data * np.sin(np.arange(-2.356194496154785, 2.356194496154785 ,0.004363323096185923))
#         x_coord = ((x + self.origin_offset) / self.pixelsize)
#         y_coord = ((y + self.origin_offset) / self.pixelsize)
#         x_coord = x_coord.to(torch.int)
#         y_coord = y_coord.to(torch.int)
#         valid_indices = (x_coord >= 0) & (x_coord < self.image_size) & (y_coord >= 0) & (y_coord < self.image_size)
#         x_coord = x_coord[valid_indices]
#         y_coord = y_coord[valid_indices]
#         input_data[0,y_coord, x_coord] = 1 #set the pixel to occupied
#         input_data[1,y_coord, x_coord] = torch.maximum(input_data[ 1,y_coord,x_coord], intensities[valid_indices])#store the maximum intensity value in the pixel
#         input_data[2,y_coord, x_coord] +=1 #count the number of points in the pixel

#         #preprocess the gt's into heatmaps

#         self.heatmaps = torch.zeros(self.image_size, self.image_size)
#         x,y = np.meshgrid(np.arange(self.image_size), np.arange(self.image_size))

#         x0 = int((data[0] + self.origin_offset) / self.pixelsize)
#         y0 = int((data[1] + self.origin_offset) / self.pixelsize)
#         heatmap = self.gaussian_2d(x, y, x0, y0, self.sx, self.sy, 1)
#         #FIXME: i think this is the wrong place to do this because sometimes we can also see the opponent behind us!
#         if data[0] < 0:#the other car is behind us, no peak in the heatmap
#             heatmap = np.zeros((self.image_size, self.image_size))
#         heatmap = torch.tensor(heatmap, dtype=torch.float32)
#         return input_data, heatmap, data


# class LidarDatasetSeqOD(Dataset):
#     '''
#     V2, not used anymore
#     '''
#     def __init__(self, dataset_path, transform=None):
#         self.dataset_path = dataset_path
#         self.transform = transform
#         self.use_heatmaps = True
#         self.consider_free_paths = True
#         self.pixelsize = 0.025#size of a pixel in meters, was 0.015 i think this bigger makes more sense for FOV
#         self.image_size = 256 #size of the image for preprocessing
#         self.feature_size = 3 #number of features in the preprocessed data
#         self.origin_offset = (self.image_size//2) * self.pixelsize
#         self.sx = self.sy = 5 #standard deviation of the gaussian peaks
#         self.len = None
#         self.seq_len = 2 #number of frames in a sequence
#         self.number_of_sets = None
#         self.cos = np.cos(np.arange(-2.356194496154785, 2.356194496154785 ,0.004363323096185923))
#         self.sin = np.sin(np.arange(-2.356194496154785, 2.356194496154785 ,0.004363323096185923))
#         self.setup()

#     def setup(self):
#        df = pd.read_csv(self.dataset_path, header=None, names=['setid','lidar','intensities','x','y','vx','vy','yaw'])
#        self.number_of_sets = df.max()['setid']
#        print("Number of sets", self.number_of_sets)
#        self.len = len(df) - 1 - self.number_of_sets
#        print("Length of Dataset", self.len)
#        print("Dataset Setup!")

#     def change_pixel_size(self, pixelsize):
#         self.pixelsize = pixelsize
#         self.origin_offset = (self.image_size//2) * self.pixelsize
#         print("Pixel size changed to: ", self.pixelsize)
#         print("Origin offset changed to: ", self.origin_offset)


#     def __getitem__(self, index):
#         seq_data = []
#         df = pd.read_csv(self.dataset_path, skiprows=index-1, nrows=self.seq_len, header=None, names=['setid','lidar','intensities','x','y','vx','vy','yaw'])
#         if len(df) == 0:
#             raise IndexError
#         if df.iloc[0]['setid'] != df.iloc[-1]['setid']:
#             return self.__getitem__(index+1)
#         for i in range(self.seq_len):
#             df.loc[i, 'lidar'] = df.loc[i, 'lidar'].replace('(', '').replace(')', '')
#             df.loc[i, 'intensities'] = df.loc[i, 'intensities'].replace('(', '').replace(')', '')
#             row = df.iloc[i]
#             lidar_data = torch.tensor(np.fromstring(df.loc[i, 'lidar'], dtype=float, sep=', '), dtype=torch.float32)
#             intensities = torch.tensor(np.fromstring(df.loc[i, 'intensities'], dtype=float, sep=','), dtype=torch.float32)
#             intensities = (intensities - intensities.min()) / (intensities.max() - intensities.min())  # normalize intensities
#             seq_data.append(self.preprocess(lidar_data, intensities))
#             if i == self.seq_len - 1:
#                 data = torch.tensor(row[3:].values.astype(float), dtype=torch.float32)
#                 heatmap = self.heatmap(data)
#         input_data = torch.stack([item for item in seq_data]).view(self.seq_len*3,self.image_size,self.image_size)

#         if self.transform:
#             input_data, heatmap, data = self.transform((input_data, heatmap, data))

#         return input_data.view(self.feature_size*self.seq_len, self.image_size, self.image_size), heatmap.view(self.image_size, self.image_size), data.view(5)


#     def __len__(self):
#         if self.len is not None:
#             return self.len
#         else:
#             with open(self.dataset_path, 'r') as f:
#                 self.len = sum(1 for row in csv.reader(f))
#                 return self.len

#     def gaussian_2d(self, x, y, x0, y0, sx, sy, A):
#         '''
#         2D Gaussian function.

#         Args:
#             x: x-coordinate
#             y: y-coordinate
#             x0: x-coordinate of the peak
#             y0: y-coordinate of the peak
#             sx: standard deviation in x
#             sy: standard deviation in y
#             A: amplitude'''
#         return A * np.exp(-((x - x0)**2 / (2 * sx**2) + (y - y0)**2 / (2 * sy**2)))

#     def heatmap(self, data):
#         #preprocess the gt's into heatmaps

#         self.heatmaps = torch.zeros(self.image_size, self.image_size)
#         x,y = np.meshgrid(np.arange(self.image_size), np.arange(self.image_size))
#         x0 = int((data[0] + self.origin_offset) / self.pixelsize)
#         y0 = int((data[1] + self.origin_offset) / self.pixelsize)
#         heatmap = self.gaussian_2d(x, y, x0, y0, self.sx, self.sy, 1)
#         if self.consider_free_paths:
#             if data[0] < 0 or np.sqrt(data[0]**2+ data[1]**2) > 3:#the other car is behind us, no peak in the heatmap
#                 heatmap = np.zeros((self.image_size, self.image_size))
#         heatmap = torch.tensor(heatmap, dtype=torch.float32)
#         return heatmap

#     def preprocess(self, lidar_data, intensities):
#         '''
#         Preprocesses the data. Convert polar coordinates to cartesian coordinates and discretize into a 256x256 grid.
#         Stores these grids in a new tensor.
#         Completely vectorized, efficient asf!
#         Does it make sense to put the origin in the middle of the grid?
#         Maybe it is better to put it in the bottom left corner? Or closer to the corner?
#         '''

#         self.use_heatmaps = True#use heatmaps for training after preprocessing
#         #preprocess the lidar data
#         input_data = torch.zeros((self.feature_size, self.image_size, self.image_size), dtype=torch.float32)
#         x = lidar_data * self.cos
#         y = lidar_data * self.sin
#         x_coord = ((x + self.origin_offset) / self.pixelsize)
#         y_coord = ((y + self.origin_offset) / self.pixelsize)
#         x_coord = x_coord.to(torch.int).long()
#         y_coord = y_coord.to(torch.int).long()
#         valid_indices = (x_coord >= 0) & (x_coord < self.image_size) & (y_coord >= 0) & (y_coord < self.image_size)
#         x_coord = x_coord[valid_indices]
#         y_coord = y_coord[valid_indices]
#         input_data[0,y_coord, x_coord] = 1 #set the pixel to occupied
#         input_data[1,y_coord, x_coord] = torch.maximum(input_data[ 1,y_coord,x_coord], intensities[valid_indices])#store the maximum intensity value in the pixel
#         input_data[2,y_coord, x_coord] +=1 #count the number of points in the pixel

#         return input_data
