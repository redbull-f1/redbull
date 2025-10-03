import torch
import numpy as np
import pandas as pd
import csv
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import random
import math
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time

class CenterSpeedDataset(Dataset):
    
    # Dataset class for the CenterSpeed dataset.
    
    def __init__(self, dataset_path, transform=None, dense=False):
        self.dataset_path = dataset_path
        self.transform = transform
        self.use_heatmaps = True
        self.dense = dense
        self.consider_free_paths = True
        self.pixelsize = 0.1  # size of a pixel in meters
        self.image_size = 128
        self.feature_size = 2 # input ch에 occupancy, density
        self.origin_offset = (self.image_size//2) * self.pixelsize
        self.sx = self.sy = 5  # standard deviation of the gaussian peaks
        self.seq_len = 2       # number of frames in a sequence
        self.angle_min = -2.356194496154785
        self.angle_increment = 0.004363323096185923
        # self.samples = []  # List to hold all preprocessed samples in memory
        # self._load_all_data_in_memory()  # preload 방식은 주석처리

        # Lazy loading: 파일 목록 및 인덱스 구성
        import os
        self.file_paths = [os.path.join(self.dataset_path, f) for f in os.listdir(self.dataset_path) if f.endswith('.csv')] if os.path.isdir(self.dataset_path) else [self.dataset_path]
        self.file_indices = []  # (start_idx, end_idx, file_path)
        self._offsets = []      # 각 파일별 오프셋 리스트
        total = 0
        for p in self.file_paths:
            offsets = []
            with open(p, 'rb') as fp:
                pos = 0
                for line in fp:
                    offsets.append(pos)
                    pos += len(line)
            usable = max(0, len(offsets) - (self.seq_len - 1))
            self._offsets.append(offsets)
            self.file_indices.append((total, total + usable, p))
            total += usable
        self.len = total
        print(f"[Lazy] Indexed {len(self.file_paths)} files, total usable samples: {self.len}")

    def _load_all_data_in_memory(self):
        # Loads all CSVs into memory and preprocesses them
        if os.path.isdir(self.dataset_path):
            csv_files = [os.path.join(self.dataset_path, f) for f in os.listdir(self.dataset_path) if f.endswith('.csv')]
        else:
            csv_files = [self.dataset_path]
        count = 0
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file, header=None, names=['lidar', 'intensities', 'x', 'y', 'vx', 'vy', 'yaw'])
                for row_index in range(len(df) - (self.seq_len - 1)):
                    seq_data = []
                    free = False
                    for i in range(self.seq_len):
                        idx = row_index + i
                        lidar_str = df.loc[idx, 'lidar'].replace('(', '').replace(')', '')
                        intensities_str = df.loc[idx, 'intensities'].replace('(', '').replace(')', '')
                        lidar_data = torch.tensor(np.fromstring(lidar_str, dtype=float, sep=', '), dtype=torch.float32)
                        intensities = torch.tensor(np.fromstring(intensities_str, dtype=float, sep=','), dtype=torch.float32)
                        try:
                            intensities = torch.full_like(intensities, 0.5)
                        except Exception:
                            print("Intensities: ", intensities)
                        seq_data.append(self.preprocess(lidar_data, intensities))
                        if i == self.seq_len - 1:
                            row = df.iloc[idx]
                            data = torch.tensor(row[2:].values.astype(float), dtype=torch.float32)
                            heatmap = self.heatmap(data)
                    input_data = torch.stack(seq_data).view(self.seq_len * self.feature_size, self.image_size, self.image_size)
                    if data[0] < 0 or np.sqrt(data[0]**2 + data[1]**2) > 6.4:
                        free = True
                    # if self.transform:
                    #     input_data, heatmap, data = self.transform((input_data, heatmap, data))
                    # if self.dense:
                    #     dense_features = self.populate_dense_features(data=data)
                    #     self.samples.append((input_data, heatmap, data, dense_features, free))
                    # else:
                    #     self.samples.append((input_data, heatmap, data, free))
                    # count += 1
                    # ...existing code...
                    # (수정)
                    if self.transform:
                        input_data, heatmap, data = self.transform((input_data, heatmap, data))

                    if self.dense:
                        dense_features = self.populate_dense_features(data=data)
                        self.samples.append((input_data, heatmap, data, dense_features, free))
                    else:
                        self.samples.append((input_data, heatmap, data, free))

                    # ...existing code...
            except Exception as e:
                print(f"Error loading {csv_file}: {e}")
        self.len = len(self.samples)
        print(f"Loaded {self.len} samples into memory.")
        # CenterSpeedDataset __init__ 또는 _load_all_data_in_memory 내부에 추가
        print(f"image_size={self.image_size}, seq_len={self.seq_len}, feature_size={self.feature_size}")
        # input_data 생성 직후 shape 확인
        print(f"input_data shape: {input_data.shape}")



    # setup is no longer needed; replaced by _load_all_data_in_memory

    def change_pixel_size(self, pixelsize):
        '''
        Changes the pixel size and the origin offset accordingly.

        Args:
            pixelsize (int): New pixel size in meters.
        '''
        self.pixelsize = pixelsize
        self.origin_offset = (self.image_size//2) * self.pixelsize
        print("Pixel size changed to: ", self.pixelsize)
        print("Origin offset changed to: ", self.origin_offset)

    def change_image_size(self, image_size):
        '''
        Changes the image size and the origin offset accordingly.

        Args:
            image_size (int): New image size in pixels.
        '''
        self.image_size = int(image_size)
        self.origin_offset = (self.image_size//2) * self.pixelsize
        print("Image size changed to: ", self.image_size)
        print("Origin offset changed to: ", self.origin_offset)

    def __getitem__(self, index):
        # Lazy loading: index → 파일/시작 row 매핑
        import torch
        import numpy as np
        file_idx = next(i for i, (s, e, _) in enumerate(self.file_indices) if s <= index < e)
        start = index - self.file_indices[file_idx][0]
        csv_path = self.file_indices[file_idx][2]
        offsets = self._offsets[file_idx]

        # 시퀀스 데이터 읽기
        rows = []
        with open(csv_path, 'rb') as fp:
            for k in range(self.seq_len):
                fp.seek(offsets[start + k])
                line = fp.readline().decode('utf-8').strip()
                rows.append(line)

        seq = []
        for line in rows:
            cols = line.split(',')
            lidar_str = cols[0].replace('(', '').replace(')', '')
            intens_str = cols[1].replace('(', '').replace(')', '')
            lidar = torch.tensor(np.fromstring(lidar_str, dtype=float, sep=', '), dtype=torch.float32)
            intens = torch.tensor(np.fromstring(intens_str, dtype=float, sep=','), dtype=torch.float32)
            if intens.numel() == 0:
                intens = torch.zeros_like(lidar)
            intens = torch.full_like(lidar, 0.5)
            x = float(cols[2]); y = float(cols[3])
            vx = float(cols[4]); vy = float(cols[5])
            yaw = float(cols[6])
            gt = torch.tensor([x, y, vx, vy, yaw], dtype=torch.float32)
            seq.append((lidar, intens, gt))

        # 입력 생성
        seq_imgs = [self.preprocess(li, inten) for (li, inten, _) in seq]
        input_data = torch.stack(seq_imgs).view(self.seq_len * self.feature_size, self.image_size, self.image_size)
        data = seq[-1][2]
        heatmap = self.heatmap(data)
        if self.transform:
            input_data, heatmap, data = self.transform((input_data, heatmap, data))
        free = bool(data[0] < 0 or np.sqrt(float(data[0])**2 + float(data[1])**2) > 6.4)
        if self.dense:
            dense_features = self.populate_dense_features(data=data)
            return input_data, heatmap, data, dense_features, free
        return input_data, heatmap, data, free

    def __len__(self):
        return self.len

# ===================== DataLoader 최적화 예시 =====================
# 아래와 같이 DataLoader를 생성하면 lazy loading에서도 속도를 높일 수 있습니다.
#
# from torch.utils.data import DataLoader
# loader = DataLoader(
#     dataset, batch_size=32, shuffle=True,
#     num_workers=4,           # 워커 4개(코어수에 맞게 조정)
#     pin_memory=True,         # GPU 사용 시 권장
#     persistent_workers=True, # PyTorch 1.7+에서 사용
#     prefetch_factor=4        # 워커당 미리 4개 배치 준비
# )
# ===============================================================

    def gaussian_2d(self, x, y, x0, y0, sx, sy, A):
        '''
        2D Gaussian function.
        '''
        return A * np.exp(-((x - x0)**2 / (2 * sx**2) + (y - y0)**2 / (2 * sy**2)))

    def populate_dense_features(self, data) -> torch.Tensor:
        '''
        Populates a tensor with dense speed and orientation values.
        '''
        tensor = torch.zeros((self.image_size, self.image_size, 3), dtype=torch.float32)
        x, y = np.meshgrid(np.arange(self.image_size), np.arange(self.image_size))
        x0 = int((data[0] + self.origin_offset) / self.pixelsize)
        y0 = int((data[1] + self.origin_offset) / self.pixelsize)
        for i in range(3):
            tensor[:, :, i] = self.gaussian_2d(x, y, x0, y0, self.sx, self.sy, data[i + 2])

        ################################ 수정 필요 ################################
            if self.consider_free_paths:
                if data[0] < 0 or np.sqrt(data[0]**2 + data[1]**2) > 6.4:  # the other car is behind us, no peak in the heatmap
                    tensor = torch.zeros((self.image_size, self.image_size, 3), dtype=torch.float32)
                    # print(f'Car Behind, setting zero')
                    return tensor

        return tensor

    def heatmap(self, data):
        '''
        Creates a heatmap from the ground truth data.
        '''
        self.heatmaps = torch.zeros(self.image_size, self.image_size)
        x, y = np.meshgrid(np.arange(self.image_size), np.arange(self.image_size))
        x0 = int((data[0] + self.origin_offset) / self.pixelsize)
        y0 = int((data[1] + self.origin_offset) / self.pixelsize)
        heatmap = self.gaussian_2d(x, y, x0, y0, self.sx, self.sy, 1)

        ################################ 수정 필요 ################################
        if self.consider_free_paths:
            if data[0] < 0 or np.sqrt(data[0]**2 + data[1]**2) > 6.4:  # the other car is behind us, no peak in the heatmap
                heatmap = np.zeros((self.image_size, self.image_size))

        heatmap = torch.tensor(heatmap, dtype=torch.float32)
        return heatmap


    #input 데이터 이미지 처리 
    def preprocess(self, lidar_data, intensities):
        '''
        Preprocesses the data. Convert polar coordinates to cartesian coordinates and discretize into an image.
        Creates 3 feature maps: occupancy, intensity(max), density(count).
        '''
        self.use_heatmaps = True  # use heatmaps for training after preprocessing

        # --- 프레임 길이에 맞춰 각도/삼각값 동적 생성 ---
        N = lidar_data.shape[0]
        angles = self.angle_min + torch.arange(N, dtype=lidar_data.dtype, device=lidar_data.device) * self.angle_increment
        cos = torch.cos(angles)
        sin = torch.sin(angles)

        # preprocess the lidar data
        input_data = torch.zeros((self.feature_size, self.image_size, self.image_size), dtype=torch.float32)
        x = lidar_data * cos
        y = lidar_data * sin
        x_coord = ((x + self.origin_offset) / self.pixelsize)
        y_coord = ((y + self.origin_offset) / self.pixelsize)
        x_coord = x_coord.to(torch.int)
        y_coord = y_coord.to(torch.int)
        valid_indices = (x_coord >= 0) & (x_coord < self.image_size) & (y_coord >= 0) & (y_coord < self.image_size)
        x_coord = x_coord[valid_indices]
        y_coord = y_coord[valid_indices]
        
        input_data[0, y_coord, x_coord] = 1  # occupied
        
        # intensity feature 를 사용하지 않음
        # input_data[1, y_coord, x_coord] = torch.maximum(input_data[1, y_coord, x_coord], intensities[valid_indices])  # max intensity per pixel
        
        # input_data[2, y_coord, x_coord] += 1  # density
        input_data[1, y_coord, x_coord] += 1  # density

        return input_data

    def cartesian_to_pixel(self, x, y):
        '''
        Converts cartesian coordinates to pixel coordinates.
        '''
        pixel_x = int(x / self.pixelsize + self.image_size / 2)
        pixel_y = int(y / self.pixelsize + self.image_size / 2)
        return pixel_x, pixel_y

    def visualize(self, index, show_preprocessed=True, show_gt=True, show_raw=True):
        '''
        Visualizes the data for a given index.
        '''
        config = [show_preprocessed, show_gt, show_raw]
        plot_rows = sum(1 for c in config if c)
        if plot_rows == 0:
            print("No plots selected!")
            return

        # fig, axs = plt.subplots(plot_rows + 1, 3, figsize=(10, 15))
        fig, axs = plt.subplots(plot_rows + 1, 2, figsize=(10, 15))
        input, gt, data, free = self.__getitem__(index)
        if self.transform is not None:
            transform_names = ', '.join([t.__class__.__name__ for t in self.transform.transforms])
        else:
            transform_names = 'None'

        axs[0, 0].axis('off')
        axs[0, 1].axis('off')
        axs[0, 2].axis('off')
        axs[0, 0].set_title('Dataset Info')
        axs[0, 0].text(0, 0.3, f'Length of dataset: {self.len}\
                                \nPath: {self.dataset_path}\
                                \nTransforms: {transform_names}\
                                \n\nIndex: {index}\
                                \nFree track: {free}\
                                \nPixel size: {self.pixelsize}\
                                \nImage size: {self.image_size}\
                                \nGaussian radius: {self.sx}', fontsize=10)

        plot_nr = 1
        if show_preprocessed:
            axs[plot_nr, 0].set_title('Occupancy')
            axs[plot_nr, 0].imshow(input[0], cmap='plasma')
            # axs[plot_nr, 1].imshow(input[1], cmap='plasma')
            # axs[plot_nr, 1].set_title('Intensity')
            # axs[plot_nr, 2].imshow(input[2], cmap='plasma')
            # axs[plot_nr, 2].set_title('Density')
            axs[plot_nr, 1].imshow(input[1], cmap='plasma')
            axs[plot_nr, 1].set_title('Density')

            for i in range(self.feature_size):
                axs[plot_nr, i].axis('off')
                axs[plot_nr, i].scatter(self.image_size // 2, self.image_size // 2, label='Ego Position', color='g')
                xpix, ypix = self.cartesian_to_pixel(data[0], data[1])
                axs[plot_nr, i].scatter(xpix, ypix, label='GT Position', color='r')
                axs[plot_nr, i].quiver(xpix, ypix, data[2], data[3], label='GT Velocity', color='r')
                yaw_degrees = np.rad2deg(data[4])
                rectangle = patches.Rectangle((xpix - 2, ypix - 4), 8, 4, angle=yaw_degrees, fill=False, color='r')
                axs[plot_nr, i].add_patch(rectangle)


            # axs[plot_nr, 2].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
            axs[plot_nr, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
            plot_nr += 1

        if show_gt:
            axs[plot_nr, 0].imshow(gt, cmap='plasma')
            axs[plot_nr, 0].set_title('GT heatmap')
            axs[plot_nr, 1].imshow(np.clip(input[0] + gt, 0, 1), cmap='plasma')
            axs[plot_nr, 1].set_title('Occupancy + GT heatmap')
            axs[plot_nr, 0].axis('off')
            axs[plot_nr, 1].axis('off')
            # axs[plot_nr, 2].axis('off')
            plot_nr += 1

        if show_raw:
            axs[plot_nr, 0].plot(self.lidar_data)
            axs[plot_nr, 0].set_title('Raw lidar ranges')
            # axs[plot_nr, 1].plot(self.intensities)
            # axs[plot_nr, 1].set_title('Raw lidar intensities')

            # --- visualize에서도 동적 cos/sin 사용 ---
            N = self.lidar_data.shape[0]
            angles = self.angle_min + torch.arange(N, dtype=self.lidar_data.dtype, device=self.lidar_data.device) * self.angle_increment
            cos = torch.cos(angles)
            sin = torch.sin(angles)
            x_raw = self.lidar_data * cos
            y_raw = self.lidar_data * sin

            axs[plot_nr, 2].scatter(x_raw.cpu(), y_raw.cpu(), s=0.1, label='Scans', alpha=float(self.intensities.mean().item()) if isinstance(self.intensities, torch.Tensor) else 0.5)
            axs[plot_nr, 2].scatter(self.data_for_plot[0], self.data_for_plot[1], color='r', label='GT-Pos')
            axs[plot_nr, 2].text(self.data_for_plot[0], self.data_for_plot[1], 'GT-Pos')

            # Adjusting view, focusing on GT-position
            dx = dy = 2
            axs[plot_nr, 2].set_xlim(self.data_for_plot[0] - dx, self.data_for_plot[0] + dx)
            axs[plot_nr, 2].set_ylim(self.data_for_plot[1] - dy, self.data_for_plot[1] + dy)
            axs[plot_nr, 2].set_xlabel('X coordinate')
            axs[plot_nr, 2].set_ylabel('Y coordinate')
            axs[plot_nr, 2].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
            axs[plot_nr, 2].set_title('Raw lidar data')



class RandomRotation:
    '''
    Randomly rotates the input data and the ground truth data.
    '''
    def __init__(self, angle=45, image_size=64, feature_size=6):
        self.angle = angle
        self.image_size = image_size
        self.feature_size = feature_size

    def __call__(self, sample):
        input, heatmap, data = sample
        angle = random.uniform(-self.angle, self.angle)
        angle_rad = -math.radians(angle)
        input_rotated = []
        for i in range(input.shape[0]):
            input_pil = transforms.functional.to_pil_image(input[i])
            input_rotated_tensor = transforms.ToTensor()(transforms.functional.rotate(input_pil, angle))
            input_rotated.append(input_rotated_tensor)
        input = torch.stack(input_rotated, dim=1)

        heatmap_image = transforms.functional.to_pil_image(heatmap)
        rotated_hm_image = transforms.functional.rotate(heatmap_image, angle)
        heatmap = transforms.ToTensor()(rotated_hm_image)

        rotation_matrix = torch.FloatTensor([[np.cos(angle_rad), -np.sin(angle_rad)],
                                            [np.sin(angle_rad),  np.cos(angle_rad)]])

        # Apply the rotation
        data[0:2] = torch.matmul(rotation_matrix, data[0:2])
        data[2:4] = torch.matmul(rotation_matrix.T, data[2:4])
        data[4] = (data[4] - math.radians(angle)) % (2 * math.pi)
        if data[4] > math.pi:
            data[4] -= 2 * math.pi

        return input.view(self.feature_size, self.image_size, self.image_size), heatmap.view(self.image_size, self.image_size), data.view(5)


class RandomFlip:
    '''
    Randomly flips the input data and the ground truth data.
    '''
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        input, heatmap, data = sample
        if random.random() < self.p:
            input = torch.flip(input, [1])
            heatmap = torch.flip(heatmap, [0])
            data[1] = -data[1]
            data[3] = -data[3]
            data[4] = -data[4]
        return input, heatmap, data


################ OLD IMPLEMENTATIONS ####################

class LidarDatasetOD(Dataset):
    '''V1, Not used anymore'''
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.use_heatmaps = True
        self.pixelsize = 0.025  # size of a pixel in meters
        self.image_size = 256   # size of the image for preprocessing
        self.feature_size = 3   # number of features in the preprocessed data
        self.origin_offset = (self.image_size//2) * self.pixelsize
        self.sx = self.sy = 5   # standard deviation of the gaussian peaks
        self.len = None

        # 동적 계산에 필요한 각도 파라미터
        self.angle_min = -2.356194496154785
        self.angle_increment = 0.004363323096185923

    def __getitem__(self, index):
        df = pd.read_csv(self.dataset_path, skiprows=index-1, nrows=1, header=None,
                         names=['lidar', 'intensities', 'x', 'y', 'vx', 'vy', 'yaw'])
        if len(df) == 0:
            raise IndexError
        df.loc[0, 'lidar'] = df.loc[0, 'lidar'].replace('(', '').replace(')', '')
        df.loc[0, 'intensities'] = df.loc[0, 'intensities'].replace('(', '').replace(')', '')
        row = df.iloc[0]
        lidar_data = torch.tensor(np.fromstring(df.loc[0, 'lidar'], dtype=float, sep=', '), dtype=torch.float32)
        intensities = torch.tensor(np.fromstring(df.loc[0, 'intensities'], dtype=float, sep=','), dtype=torch.float32)
        intensities = (intensities - intensities.min()) / (intensities.max() - intensities.min())  # normalize intensities
        data = torch.tensor(row[2:].values.astype(float), dtype=torch.float32)
        return self.preprocess(lidar_data, intensities, data)

    def __len__(self):
        if self.len is not None:
            return self.len
        else:
            with open(self.dataset_path, 'r') as f:
                self.len = sum(1 for row in csv.reader(f))
                return self.len

    def gaussian_2d(self, x, y, x0, y0, sx, sy, A):
        return A * np.exp(-((x - x0)**2 / (2 * sx**2) + (y - y0)**2 / (2 * sy**2)))

    def preprocess(self, lidar_data, intensities, data):
        '''
        Preprocesses the data. Convert polar coordinates to cartesian coordinates and discretize into a 256x256 grid.
        '''
        self.use_heatmaps = True

        # --- 동적 각도 계산 ---
        N = lidar_data.shape[0]
        angles_np = self.angle_min + np.arange(N, dtype=np.float32) * self.angle_increment
        cos = torch.from_numpy(np.cos(angles_np)).to(lidar_data.dtype)
        sin = torch.from_numpy(np.sin(angles_np)).to(lidar_data.dtype)

        input_data = torch.zeros((self.feature_size, self.image_size, self.image_size), dtype=torch.float32)
        x = lidar_data * cos
        y = lidar_data * sin
        x_coord = ((x + self.origin_offset) / self.pixelsize)
        y_coord = ((y + self.origin_offset) / self.pixelsize)
        x_coord = x_coord.to(torch.int)
        y_coord = y_coord.to(torch.int)
        valid_indices = (x_coord >= 0) & (x_coord < self.image_size) & (y_coord >= 0) & (y_coord < self.image_size)
        x_coord = x_coord[valid_indices]
        y_coord = y_coord[valid_indices]
        input_data[0, y_coord, x_coord] = 1
        input_data[1, y_coord, x_coord] = torch.maximum(input_data[1, y_coord, x_coord], intensities[valid_indices])
        input_data[2, y_coord, x_coord] += 1

        # heatmap
        self.heatmaps = torch.zeros(self.image_size, self.image_size)
        X, Y = np.meshgrid(np.arange(self.image_size), np.arange(self.image_size))
        x0 = int((data[0] + self.origin_offset) / self.pixelsize)
        y0 = int((data[1] + self.origin_offset) / self.pixelsize)
        heatmap = self.gaussian_2d(X, Y, x0, y0, self.sx, self.sy, 1)
        if data[0] < 0:  # (구현 원문 유지)
            heatmap = np.zeros((self.image_size, self.image_size))
        heatmap = torch.tensor(heatmap, dtype=torch.float32)

        return input_data, heatmap, data


class LidarDatasetSeqOD(Dataset):
    '''
    V2, not used anymore
    '''
    def __init__(self, dataset_path, transform=None):
        self.dataset_path = dataset_path
        self.transform = transform
        self.use_heatmaps = True
        self.consider_free_paths = True
        self.pixelsize = 0.025  # size of a pixel in meters
        self.image_size = 256   # size of the image for preprocessing
        self.feature_size = 3   # number of features in the preprocessed data
        self.origin_offset = (self.image_size//2) * self.pixelsize
        self.sx = self.sy = 5   # standard deviation of the gaussian peaks
        self.len = None
        self.seq_len = 2        # number of frames in a sequence
        self.number_of_sets = None

        # 동적 각도 계산용 파라미터
        self.angle_min = -2.356194496154785
        self.angle_increment = 0.004363323096185923

        self.setup()

    def setup(self):
        df = pd.read_csv(self.dataset_path, header=None, names=['setid', 'lidar', 'intensities', 'x', 'y', 'vx', 'vy', 'yaw'])
        self.number_of_sets = df.max()['setid']
        print("Number of sets", self.number_of_sets)
        self.len = len(df) - 1 - self.number_of_sets
        print("Length of Dataset", self.len)
        print("Dataset Setup!")

    def change_pixel_size(self, pixelsize):
        self.pixelsize = pixelsize
        self.origin_offset = (self.image_size//2) * self.pixelsize
        print("Pixel size changed to: ", self.pixelsize)
        print("Origin offset changed to: ", self.origin_offset)

    def __getitem__(self, index):
        seq_data = []
        df = pd.read_csv(self.dataset_path, skiprows=index-1, nrows=self.seq_len, header=None,
                         names=['setid', 'lidar', 'intensities', 'x', 'y', 'vx', 'vy', 'yaw'])
        if len(df) == 0:
            raise IndexError
        if df.iloc[0]['setid'] != df.iloc[-1]['setid']:
            return self.__getitem__(index + 1)
        for i in range(self.seq_len):
            df.loc[i, 'lidar'] = df.loc[i, 'lidar'].replace('(', '').replace(')', '')
            df.loc[i, 'intensities'] = df.loc[i, 'intensities'].replace('(', '').replace(')', '')
            row = df.iloc[i]
            lidar_data = torch.tensor(np.fromstring(df.loc[i, 'lidar'], dtype=float, sep=', '), dtype=torch.float32)
            intensities = torch.tensor(np.fromstring(df.loc[i, 'intensities'], dtype=float, sep=','), dtype=torch.float32)
            intensities = (intensities - intensities.min()) / (intensities.max() - intensities.min())  # normalize intensities
            seq_data.append(self.preprocess(lidar_data, intensities))















# import os
# import math
# import random
# import csv
# import time
# import numpy as np
# import pandas as pd
# import torch
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
# from torchvision.transforms import functional as TF
# from torchvision.transforms.functional import InterpolationMode
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches


# # ----------------------------
# # Safe, shape-agnostic transforms
# # ----------------------------
# class RandomRotation:
#     def __init__(self, angle=45):
#         self.angle = angle

#     def __call__(self, sample):
#         input, heatmap, data = sample    # input: [C,H,W], heatmap: [H,W], data: [5]
#         ang = random.uniform(-self.angle, self.angle)
#         ang_rad = -math.radians(ang)     # 이미지 회전과 반대 부호로 좌표 회전

#         # 텐서 그대로 회전 (채널 유지, NEAREST로 occupancy/density 보존)
#         input = TF.rotate(input, ang, interpolation=InterpolationMode.NEAREST)
#         heatmap = TF.rotate(heatmap.unsqueeze(0), ang, interpolation=InterpolationMode.NEAREST).squeeze(0)

#         # GT 좌표/속도/요 회전
#         R = torch.tensor([[math.cos(ang_rad), -math.sin(ang_rad)],
#                           [math.sin(ang_rad),  math.cos(ang_rad)]],
#                          dtype=data.dtype, device=data.device if data.is_cuda else None)
#         data[:2]  = R @ data[:2]      # (x,y)
#         data[2:4] = R.T @ data[2:4]   # (vx,vy)
#         data[4]   = (data[4] - math.radians(ang)) % (2*math.pi)
#         if data[4] > math.pi:
#             data[4] -= 2*math.pi
#         return input, heatmap, data


# class RandomFlip:
#     def __init__(self, p=0.5, mode='horizontal'):  # 'vertical' 또는 'horizontal'
#         self.p = p
#         self.mode = mode

#     def __call__(self, sample):
#         input, heatmap, data = sample  # input: [C,H,W]
#         if random.random() >= self.p:
#             return input, heatmap, data

#         if self.mode == 'vertical':
#             input = torch.flip(input, dims=[1])  # H축
#             heatmap = torch.flip(heatmap, dims=[0])
#             data[1] = -data[1]   # y
#             data[3] = -data[3]   # vy
#             data[4] = -data[4]   # yaw
#         else:
#             input = torch.flip(input, dims=[2])  # W축
#             heatmap = torch.flip(heatmap, dims=[1])
#             data[0] = -data[0]   # x
#             data[2] = -data[2]   # vx
#             data[4] = ((math.pi - data[4] + math.pi) % (2*math.pi)) - math.pi  # (-pi,pi]

#         return input, heatmap, data


# # ----------------------------
# # Dataset with optional preload/lazy
# # ----------------------------
# class CenterSpeedDataset(Dataset):
#     """
#     CenterSpeed dataset:
#       - preload=True  : 모든 샘플을 메모리에 적재(작은 데이터만 권장)
#       - preload=False : lazy 로딩 + 빠른 바이트 오프셋 인덱싱(대용량 권장)
#     """
#     def __init__(self, dataset_path, transform=None, dense=False, preload=False, has_header=False):
#         self.dataset_path = dataset_path
#         self.transform = transform
#         self.use_heatmaps = True
#         self.dense = dense
#         self.consider_free_paths = True

#         # Grid 설정
#         self.pixelsize = 0.1
#         self.image_size = 128
#         self.feature_size = 2   # occupancy, density
#         self.origin_offset = (self.image_size // 2) * self.pixelsize

#         # Heatmap 가우시안 표준편차
#         self.sx = 2
#         self.sy = 2

#         # 시퀀스 길이
#         self.seq_len = 2

#         # 라이다 각도 파라미터
#         self.angle_min = -2.356194496154785
#         self.angle_increment = 0.004363323096185923

#         # CSV 포맷
#         self.has_header = has_header

#         # 로딩 모드
#         self.preload = preload
#         if self.preload:
#             self.samples = []
#             self._load_all_data_in_memory()
#         else:
#             self._setup_lazy()  # 파일 목록 + 바이트 오프셋 인덱스 구성

#     # ----------------------------
#     # Preload (기존 방식) - 작은 데이터 전용
#     # ----------------------------
#     def _load_all_data_in_memory(self):
#         if os.path.isdir(self.dataset_path):
#             csv_files = [os.path.join(self.dataset_path, f) for f in os.listdir(self.dataset_path) if f.endswith('.csv')]
#         else:
#             csv_files = [self.dataset_path]

#         for csv_file in csv_files:
#             try:
#                 df = pd.read_csv(
#                     csv_file,
#                     header=None if not self.has_header else 0,
#                     names=None if not self.has_header else ['lidar', 'intensities', 'x', 'y', 'vx', 'vy', 'yaw']
#                 )
#                 # 헤더 없는 케이스: 컬럼명 지정
#                 if not self.has_header:
#                     df.columns = ['lidar', 'intensities', 'x', 'y', 'vx', 'vy', 'yaw']

#                 for row_index in range(len(df) - (self.seq_len - 1)):
#                     seq_data = []
#                     for i in range(self.seq_len):
#                         idx = row_index + i
#                         lidar_str = str(df.loc[idx, 'lidar']).replace('(', '').replace(')', '')
#                         intens_str = str(df.loc[idx, 'intensities']).replace('(', '').replace(')', '')

#                         lidar = torch.tensor(np.fromstring(lidar_str, dtype=float, sep=', '), dtype=torch.float32)
#                         intens = torch.tensor(np.fromstring(intens_str, dtype=float, sep=','), dtype=torch.float32)
#                         if intens.numel() == 0:
#                             intens = torch.zeros_like(lidar)
#                         # 고정 0.5 사용(원 코드 유지)
#                         intens = torch.full_like(lidar, 0.5)

#                         seq_data.append(self.preprocess(lidar, intens))

#                     input_data = torch.stack(seq_data).view(self.seq_len * self.feature_size, self.image_size, self.image_size)

#                     # 마지막 프레임의 GT 사용
#                     last = df.iloc[row_index + self.seq_len - 1]
#                     data = torch.tensor([float(last['x']), float(last['y']), float(last['vx']), float(last['vy']), float(last['yaw'])], dtype=torch.float32)
#                     heatmap = self.heatmap(data)

#                     # transform 적용 후 free 판정
#                     if self.transform:
#                         input_data, heatmap, data = self.transform((input_data, heatmap, data))

#                     free = bool(data[0] < 0 or math.hypot(float(data[0]), float(data[1])) > 6.4)

#                     if self.dense:
#                         dense_features = self.populate_dense_features(data=data)
#                         self.samples.append((input_data, heatmap, data, dense_features, free))
#                     else:
#                         self.samples.append((input_data, heatmap, data, free))

#             except Exception as e:
#                 print(f"[Preload] Error loading {csv_file}: {e}")

#         self.len = len(self.samples)
#         print(f"[Preload] Loaded {self.len} samples into memory.")
#         print(f"image_size={self.image_size}, seq_len={self.seq_len}, feature_size={self.feature_size}")

#     # ----------------------------
#     # Lazy setup: 파일 목록 + 바이트 오프셋 인덱스
#     # ----------------------------
#     def _setup_lazy(self):
#         if os.path.isdir(self.dataset_path):
#             self.file_paths = [os.path.join(self.dataset_path, f) for f in os.listdir(self.dataset_path) if f.endswith('.csv')]
#         else:
#             self.file_paths = [self.dataset_path]

#         self._offsets = []
#         self.file_indices = []
#         total = 0

#         for p in self.file_paths:
#             offsets = []
#             with open(p, 'rb') as fp:
#                 pos = 0
#                 # header 처리
#                 if self.has_header:
#                     hdr = fp.readline()
#                     pos += len(hdr)
#                 for line in fp:
#                     offsets.append(pos)
#                     pos += len(line)
#             usable = max(0, len(offsets) - (self.seq_len - 1))
#             self._offsets.append(offsets)
#             self.file_indices.append((total, total + usable, p))
#             total += usable

#         self.len = total
#         print(f"[Lazy] Indexed {len(self.file_paths)} files, total usable samples: {self.len}")

#     # ----------------------------
#     # Lazy fast reader by byte offsets
#     # ----------------------------
#     def _read_seq_by_offset(self, csv_path, offsets, start_row):
#         """start_row에서 seq_len 줄을 오프셋으로 빠르게 읽어 파싱"""
#         rows = []
#         with open(csv_path, 'rb') as fp:
#             # header 스킵은 _setup_lazy에서 이미 처리함
#             for k in range(self.seq_len):
#                 fp.seek(offsets[start_row + k])
#                 line = fp.readline().decode('utf-8').strip()
#                 rows.append(line)

#         out = []
#         for line in rows:
#             cols = line.split(',')
#             # 기대 포맷: lidar, intensities, x, y, vx, vy, yaw
#             # lidar/intensities가 괄호 문자열이면 제거
#             lidar_str  = cols[0].replace('(', '').replace(')', '')
#             intens_str = cols[1].replace('(', '').replace(')', '')

#             lidar = torch.tensor(np.fromstring(lidar_str, dtype=float, sep=', '), dtype=torch.float32)
#             intens = torch.tensor(np.fromstring(intens_str, dtype=float, sep=','), dtype=torch.float32)
#             if intens.numel() == 0:
#                 intens = torch.zeros_like(lidar)
#             intens = torch.full_like(lidar, 0.5)

#             x  = float(cols[2]); y  = float(cols[3])
#             vx = float(cols[4]); vy = float(cols[5])
#             yaw = float(cols[6])
#             gt = torch.tensor([x, y, vx, vy, yaw], dtype=torch.float32)

#             out.append((lidar, intens, gt))
#         return out

#     # ----------------------------
#     # Public API
#     # ----------------------------
#     def __len__(self):
#         if hasattr(self, "samples"):
#             return len(self.samples)
#         return self.len

#     def __getitem__(self, index):
#         # Preload 경로
#         if hasattr(self, "samples"):
#             return self.samples[index]

#         # Lazy 경로: index → 파일 / 시작 로우 매핑
#         file_idx = next(i for i, (s, e, _) in enumerate(self.file_indices) if s <= index < e)
#         start = index - self.file_indices[file_idx][0]
#         csv_path = self.file_indices[file_idx][2]
#         offsets = self._offsets[file_idx]

#         seq = self._read_seq_by_offset(csv_path, offsets, start)

#         # 입력 생성
#         seq_imgs = [self.preprocess(li, inten) for (li, inten, _) in seq]
#         input_data = torch.stack(seq_imgs).view(self.seq_len * self.feature_size, self.image_size, self.image_size)

#         # 마지막 프레임의 GT로 heatmap 생성
#         data = seq[-1][2]
#         heatmap = self.heatmap(data)

#         # transform 적용 후 free 판정
#         if self.transform:
#             input_data, heatmap, data = self.transform((input_data, heatmap, data))

#         free = bool(data[0] < 0 or math.hypot(float(data[0]), float(data[1])) > 6.4)

#         if self.dense:
#             dense_features = self.populate_dense_features(data=data)
#             return input_data, heatmap, data, dense_features, free
#         return input_data, heatmap, data, free

#     # ----------------------------
#     # Utils
#     # ----------------------------
#     def change_pixel_size(self, pixelsize: float):
#         self.pixelsize = float(pixelsize)
#         self.origin_offset = (self.image_size // 2) * self.pixelsize
#         print("Pixel size changed to:", self.pixelsize, "origin_offset:", self.origin_offset)

#     def change_image_size(self, image_size: int):
#         # 주의: preload=False에서만 안전. preload=True로 이미 적재된 후 바꾸면 불일치
#         self.image_size = int(image_size)
#         self.origin_offset = (self.image_size // 2) * self.pixelsize
#         print("Image size changed to:", self.image_size, "origin_offset:", self.origin_offset)

#     def gaussian_2d(self, x, y, x0, y0, sx, sy, A):
#         return A * np.exp(-((x - x0) ** 2 / (2 * sx ** 2) + (y - y0) ** 2 / (2 * sy ** 2)))

#     def populate_dense_features(self, data) -> torch.Tensor:
#         """(H,W,3): vx, vy, yaw의 dense map"""
#         tensor = torch.zeros((self.image_size, self.image_size, 3), dtype=torch.float32)
#         xx, yy = np.meshgrid(np.arange(self.image_size), np.arange(self.image_size))
#         x0 = int((data[0] + self.origin_offset) / self.pixelsize)
#         y0 = int((data[1] + self.origin_offset) / self.pixelsize)

#         # 3채널: vx, vy, yaw
#         for i in range(3):
#             tensor[:, :, i] = self.gaussian_2d(xx, yy, x0, y0, self.sx, self.sy, data[i + 2])

#         if self.consider_free_paths:
#             if (data[0] < 0) or (math.hypot(float(data[0]), float(data[1])) > 6.4):
#                 tensor.zero_()

#         return tensor

#     def heatmap(self, data: torch.Tensor) -> torch.Tensor:
#         """(H,W) heatmap"""
#         xx, yy = np.meshgrid(np.arange(self.image_size), np.arange(self.image_size))
#         x0 = int((data[0] + self.origin_offset) / self.pixelsize)
#         y0 = int((data[1] + self.origin_offset) / self.pixelsize)
#         hm = self.gaussian_2d(xx, yy, x0, y0, self.sx, self.sy, 1.0)

#         if self.consider_free_paths:
#             if (data[0] < 0) or (math.hypot(float(data[0]), float(data[1])) > 6.4):
#                 hm = np.zeros((self.image_size, self.image_size), dtype=np.float32)

#         return torch.tensor(hm, dtype=torch.float32)

#     def preprocess(self, lidar_data: torch.Tensor, intensities: torch.Tensor) -> torch.Tensor:
#         """Polar → Cartesian → BEV(occupancy, density)"""
#         # 각도 벡터
#         N = lidar_data.shape[0]
#         angles = self.angle_min + torch.arange(N, dtype=lidar_data.dtype, device=lidar_data.device) * self.angle_increment
#         cos = torch.cos(angles)
#         sin = torch.sin(angles)

#         # 그리드
#         input_data = torch.zeros((self.feature_size, self.image_size, self.image_size), dtype=torch.float32)

#         x = lidar_data * cos
#         y = lidar_data * sin
#         x_pix = ((x + self.origin_offset) / self.pixelsize).to(torch.int)
#         y_pix = ((y + self.origin_offset) / self.pixelsize).to(torch.int)

#         valid = (x_pix >= 0) & (x_pix < self.image_size) & (y_pix >= 0) & (y_pix < self.image_size)
#         x_pix = x_pix[valid]
#         y_pix = y_pix[valid]

#         # occupancy
#         input_data[0, y_pix, x_pix] = 1
#         # density (count)
#         input_data[1, y_pix, x_pix] += 1

#         return input_data

#     def cartesian_to_pixel(self, x, y):
#         px = int(x / self.pixelsize + self.image_size / 2)
#         py = int(y / self.pixelsize + self.image_size / 2)
#         return px, py

#     def visualize(self, index, show_preprocessed=True, show_gt=True):
#         """2열 시각화(안전). lazy/preload 모두 동작."""
#         item = self.__getitem__(index)
#         if self.dense:
#             input, gt, data, dense_features, free = item
#         else:
#             input, gt, data, free = item

#         rows = int(show_preprocessed) + int(show_gt)
#         if rows == 0:
#             print("No plots selected!")
#             return

#         fig, axs = plt.subplots(rows + 1, 2, figsize=(10, 12))
#         # Info
#         for j in range(2):
#             axs[0, j].axis('off')
#         axs[0, 0].set_title('Dataset Info')
#         transform_names = 'None' if self.transform is None else ', '.join(
#             [t.__class__.__name__ for t in self.transform.transforms]
#         )
#         axs[0, 0].text(0, 0.3,
#                        f'Length: {len(self)}'
#                        f'\nPath: {self.dataset_path}'
#                        f'\nTransforms: {transform_names}'
#                        f'\nIndex: {index}'
#                        f'\nFree: {free}'
#                        f'\nPixel size: {self.pixelsize}'
#                        f'\nImage size: {self.image_size}'
#                        f'\nGaussian sigma: {self.sx}', fontsize=10)

#         r = 1
#         if show_preprocessed:
#             axs[r, 0].set_title('Occupancy'); axs[r, 0].imshow(input[0], cmap='plasma'); axs[r, 0].axis('off')
#             axs[r, 1].set_title('Density');   axs[r, 1].imshow(input[1], cmap='plasma'); axs[r, 1].axis('off')

#             # GT overlay marker
#             for c in range(2):
#                 axs[r, c].scatter(self.image_size // 2, self.image_size // 2, label='Ego', color='g', s=10)
#                 xpix, ypix = self.cartesian_to_pixel(data[0], data[1])
#                 axs[r, c].scatter(xpix, ypix, label='GT', color='r', s=10)
#                 axs[r, c].quiver(xpix, ypix, data[2], data[3], color='r', angles='xy', scale_units='xy', scale=1)
#             axs[r, 1].legend(loc='upper right')
#             r += 1

#         if show_gt:
#             axs[r, 0].set_title('GT heatmap'); axs[r, 0].imshow(gt, cmap='plasma'); axs[r, 0].axis('off')
#             axs[r, 1].set_title('Occupancy + GT'); axs[r, 1].imshow(np.clip(input[0] + gt, 0, 1), cmap='plasma'); axs[r, 1].axis('off')

#         plt.tight_layout()
#         plt.show()


# # ----------------------------
# # 사용 예시
# # ----------------------------
# if __name__ == "__main__":
#     transform = transforms.Compose([
#         RandomRotation(45),
#         RandomFlip(0.5, mode='horizontal')
#     ])

#     # 대용량 csv 권장: preload=False
#     ds = CenterSpeedDataset(
#         dataset_path="/home/harry/sim_ws/src/f1tenth_gym_ros/Train_GT",
#         transform=transform,
#         dense=True,           # 메모리 부담되면 False로 먼저 테스트
#         preload=False,        # True면 전부 메모리 적재(대용량 비권장)
#         has_header=False      # csv에 헤더가 있으면 True
#     )

#     print("dataset len:", len(ds))

#     # DataLoader 튜닝 예시
#     loader = DataLoader(
#         ds, batch_size=32, shuffle=True,
#         num_workers=max(1, (os.cpu_count() or 4) - 1),
#         pin_memory=True, persistent_workers=True, prefetch_factor=3
#     )

#     # 샘플 하나 시각화
#     # ds.visualize(0, show_preprocessed=True, show_gt=True)



# # import torch
# # import numpy as np
# # import pandas as pd
# # import csv
# # from torch.utils.data import Dataset, DataLoader, random_split
# # from torchvision import transforms
# # import random
# # import math
# # import os
# # import matplotlib.pyplot as plt
# # import matplotlib.patches as patches
# # import time




# # from torchvision.transforms import functional as TF
# # from torchvision.transforms.functional import InterpolationMode

# # class RandomRotation:
# #     def __init__(self, angle=45):
# #         self.angle = angle

# #     def __call__(self, sample):
# #         input, heatmap, data = sample    # input: [C,H,W], heatmap: [H,W], data: [5]
# #         ang = random.uniform(-self.angle, self.angle)
# #         ang_rad = -math.radians(ang)     # 좌표계 회전 보정(이미지 회전과 반대 부호)

# #         # 텐서 그대로 회전 (채널 유지)
# #         input = TF.rotate(input, ang, interpolation=InterpolationMode.NEAREST)
# #         heatmap = TF.rotate(heatmap.unsqueeze(0), ang, interpolation=InterpolationMode.NEAREST).squeeze(0)

# #         # GT 좌표/속도/요 회전
# #         rot = torch.tensor([[math.cos(ang_rad), -math.sin(ang_rad)],
# #                             [math.sin(ang_rad),  math.cos(ang_rad)]],
# #                            dtype=data.dtype, device=data.device if data.is_cuda else None)
# #         data[:2] = rot @ data[:2]          # (x,y)
# #         data[2:4] = rot.T @ data[2:4]      # (vx,vy)
# #         data[4] = (data[4] - math.radians(ang)) % (2*math.pi)
# #         if data[4] > math.pi:
# #             data[4] -= 2*math.pi
# #         return input, heatmap, data


# # class RandomFlip:
# #     def __init__(self, p=0.5, mode='horizontal'):
# #         self.p = p
# #         self.mode = mode  # 'vertical' = 상하반전, 'horizontal' = 좌우반전

# #     def __call__(self, sample):
# #         input, heatmap, data = sample  # input: [C,H,W]
# #         if random.random() >= self.p:
# #             return input, heatmap, data

# #         if self.mode == 'vertical':
# #             # 영상의 높이(H) 축 뒤집기
# #             input = torch.flip(input, dims=[1])
# #             heatmap = torch.flip(heatmap, dims=[0])
# #             data[1] = -data[1]   # y
# #             data[3] = -data[3]   # vy
# #             data[4] = -data[4]   # yaw
# #         else:
# #             # 영상의 너비(W) 축 뒤집기
# #             input = torch.flip(input, dims=[2])
# #             heatmap = torch.flip(heatmap, dims=[1])
# #             data[0] = -data[0]   # x
# #             data[2] = -data[2]   # vx
# #             data[4] = math.pi - data[4]
# #             # yaw 정규화(-pi,pi]
# #             data[4] = ((data[4] + math.pi) % (2*math.pi)) - math.pi

# #         return input, heatmap, data



# # class CenterSpeedDataset(Dataset):
    
# #     # Dataset class for the CenterSpeed dataset.
    
# #     def __init__(self, dataset_path, transform=None, dense=False):
# #         self.dataset_path = dataset_path
# #         self.transform = transform
# #         self.use_heatmaps = True
# #         self.dense = dense
# #         self.consider_free_paths = True
# #         self.pixelsize = 0.1  # size of a pixel in meters
# #         # self.image_size = 64   # size of the image for preprocessing
# #         self.image_size = 128
# #         self.feature_size = 2 #input ch에 occupancy, density
# #         self.origin_offset = (self.image_size//2) * self.pixelsize
# #         self.sx = self.sy = 5  # standard deviation of the gaussian peaks
# #         self.len = None
# #         self.seq_len = 2       # number of frames in a sequence
# #         self.number_of_sets = None
# #         self.angle_min = -2.356194496154785
# #         self.angle_increment = 0.004363323096185923
# #         self.samples = []  # List to hold all preprocessed samples in memory
# #         self._load_all_data_in_memory()







# #     def _load_all_data_in_memory(self):
# #         # Loads all CSVs into memory and preprocesses them
# #         if os.path.isdir(self.dataset_path):
# #             csv_files = [os.path.join(self.dataset_path, f) for f in os.listdir(self.dataset_path) if f.endswith('.csv')]
# #         else:
# #             csv_files = [self.dataset_path]
# #         count = 0
# #         for csv_file in csv_files:
# #             try:
# #                 df = pd.read_csv(csv_file, header=None, names=['lidar', 'intensities', 'x', 'y', 'vx', 'vy', 'yaw'])
# #                 for row_index in range(len(df) - (self.seq_len - 1)):
# #                     seq_data = []
# #                     free = False
# #                     for i in range(self.seq_len):
# #                         idx = row_index + i
# #                         lidar_str = df.loc[idx, 'lidar'].replace('(', '').replace(')', '')
# #                         intensities_str = df.loc[idx, 'intensities'].replace('(', '').replace(')', '')
# #                         lidar_data = torch.tensor(np.fromstring(lidar_str, dtype=float, sep=', '), dtype=torch.float32)
# #                         intensities = torch.tensor(np.fromstring(intensities_str, dtype=float, sep=','), dtype=torch.float32)
# #                         try:
# #                             intensities = torch.full_like(intensities, 0.5)
# #                         except Exception:
# #                             print("Intensities: ", intensities)
# #                         seq_data.append(self.preprocess(lidar_data, intensities))
# #                         if i == self.seq_len - 1:
# #                             row = df.iloc[idx]
# #                             data = torch.tensor(row[2:].values.astype(float), dtype=torch.float32)
# #                             heatmap = self.heatmap(data)
# #                     input_data = torch.stack(seq_data).view(self.seq_len * self.feature_size, self.image_size, self.image_size)
# #                     if data[0] < 0 or np.sqrt(data[0]**2 + data[1]**2) > 6.4:
# #                         free = True
                    
# #                     ##########################################
# #                     # # _load_all_data_in_memory() 내부에서 input_data/heatmap/data 만든 직후
# #                     # if self.transform:
# #                     #     input_data, heatmap, data = self.transform((input_data, heatmap, data))

# #                     # # ▶ 변환 후에 free 판정
# #                     # free = bool(data[0] < 0 or np.sqrt(float(data[0])**2 + float(data[1])**2) > 6.4)

# #                     # if self.dense:
# #                     #     dense_features = self.populate_dense_features(data=data)
# #                     #     self.samples.append((input_data, heatmap, data, dense_features, free))
# #                     # else:
# #                     #     self.samples.append((input_data, heatmap, data, free))

# #                     # # ...existing code...
# #                     # # (수정)
# #                     # if self.transform:
# #                     #     input_data, heatmap, data = self.transform((input_data, heatmap, data))

# #                     # if self.dense:
# #                     #     dense_features = self.populate_dense_features(data=data)
# #                     #     self.samples.append((input_data, heatmap, data, dense_features, free))
# #                     # else:
# #                     #     self.samples.append((input_data, heatmap, data, free))
# #                     # (input_data, heatmap, data) 생성 직후
# #                     if self.transform:
# #                         input_data, heatmap, data = self.transform((input_data, heatmap, data))

# #                     # 변환 후 좌표 기준으로 free 판정
# #                     free = bool(data[0] < 0 or np.sqrt(float(data[0])**2 + float(data[1])**2) > 6.4)

# #                     if self.dense:
# #                         dense_features = self.populate_dense_features(data=data)
# #                         self.samples.append((input_data, heatmap, data, dense_features, free))
# #                     else:
# #                         self.samples.append((input_data, heatmap, data, free))

# #                     # ...existing code...
# #             except Exception as e:
# #                 print(f"Error loading {csv_file}: {e}")
# #         self.len = len(self.samples)
# #         print(f"Loaded {self.len} samples into memory.")
# #         # CenterSpeedDataset __init__ 또는 _load_all_data_in_memory 내부에 추가
# #         print(f"image_size={self.image_size}, seq_len={self.seq_len}, feature_size={self.feature_size}")
# #         # input_data 생성 직후 shape 확인
# #         print(f"input_data shape: {input_data.shape}")



# #     # setup is no longer needed; replaced by _load_all_data_in_memory

# #     def change_pixel_size(self, pixelsize):
# #         '''
# #         Changes the pixel size and the origin offset accordingly.

# #         Args:
# #             pixelsize (int): New pixel size in meters.
# #         '''
# #         self.pixelsize = pixelsize
# #         self.origin_offset = (self.image_size//2) * self.pixelsize
# #         print("Pixel size changed to: ", self.pixelsize)
# #         print("Origin offset changed to: ", self.origin_offset)

# #     def change_image_size(self, image_size):
# #         '''
# #         Changes the image size and the origin offset accordingly.

# #         Args:
# #             image_size (int): New image size in pixels.
# #         '''
# #         self.image_size = int(image_size)
# #         self.origin_offset = (self.image_size//2) * self.pixelsize
# #         print("Image size changed to: ", self.image_size)
# #         print("Origin offset changed to: ", self.origin_offset)

# #     def __getitem__(self, index):
# #         # Return preprocessed sample from memory
# #         return self.samples[index]

# #     def __len__(self):
# #         return self.len

# #     def gaussian_2d(self, x, y, x0, y0, sx, sy, A):
# #         '''
# #         2D Gaussian function.
# #         '''
# #         return A * np.exp(-((x - x0)**2 / (2 * sx**2) + (y - y0)**2 / (2 * sy**2)))

# #     def populate_dense_features(self, data) -> torch.Tensor:
# #         '''
# #         Populates a tensor with dense speed and orientation values.
# #         '''
# #         tensor = torch.zeros((self.image_size, self.image_size, 3), dtype=torch.float32)
# #         x, y = np.meshgrid(np.arange(self.image_size), np.arange(self.image_size))
# #         x0 = int((data[0] + self.origin_offset) / self.pixelsize)
# #         y0 = int((data[1] + self.origin_offset) / self.pixelsize)
# #         for i in range(3):
# #             tensor[:, :, i] = self.gaussian_2d(x, y, x0, y0, self.sx, self.sy, data[i + 2])

# #         ################################ 수정 필요 ################################
# #             if self.consider_free_paths:
# #                 if data[0] < 0 or np.sqrt(data[0]**2 + data[1]**2) > 6.4:  # the other car is behind us, no peak in the heatmap
# #                     tensor = torch.zeros((self.image_size, self.image_size, 3), dtype=torch.float32)
# #                     # print(f'Car Behind, setting zero')
# #                     return tensor

# #         return tensor

# #     def heatmap(self, data):
# #         '''
# #         Creates a heatmap from the ground truth data.
# #         '''
# #         self.heatmaps = torch.zeros(self.image_size, self.image_size)
# #         x, y = np.meshgrid(np.arange(self.image_size), np.arange(self.image_size))
# #         x0 = int((data[0] + self.origin_offset) / self.pixelsize)
# #         y0 = int((data[1] + self.origin_offset) / self.pixelsize)
# #         heatmap = self.gaussian_2d(x, y, x0, y0, self.sx, self.sy, 1)

# #         ################################ 수정 필요 ################################
# #         if self.consider_free_paths:
# #             if data[0] < 0 or np.sqrt(data[0]**2 + data[1]**2) > 6.4:  # the other car is behind us, no peak in the heatmap
# #                 heatmap = np.zeros((self.image_size, self.image_size))

# #         heatmap = torch.tensor(heatmap, dtype=torch.float32)
# #         return heatmap


# #     #input 데이터 이미지 처리 
# #     def preprocess(self, lidar_data, intensities):
# #         '''
# #         Preprocesses the data. Convert polar coordinates to cartesian coordinates and discretize into an image.
# #         Creates 3 feature maps: occupancy, intensity(max), density(count).
# #         '''
# #         self.use_heatmaps = True  # use heatmaps for training after preprocessing

# #         # --- 프레임 길이에 맞춰 각도/삼각값 동적 생성 ---
# #         N = lidar_data.shape[0]
# #         angles = self.angle_min + torch.arange(N, dtype=lidar_data.dtype, device=lidar_data.device) * self.angle_increment
# #         cos = torch.cos(angles)
# #         sin = torch.sin(angles)

# #         # preprocess the lidar data
# #         input_data = torch.zeros((self.feature_size, self.image_size, self.image_size), dtype=torch.float32)
# #         x = lidar_data * cos
# #         y = lidar_data * sin
# #         x_coord = ((x + self.origin_offset) / self.pixelsize)
# #         y_coord = ((y + self.origin_offset) / self.pixelsize)
# #         x_coord = x_coord.to(torch.int)
# #         y_coord = y_coord.to(torch.int)
# #         valid_indices = (x_coord >= 0) & (x_coord < self.image_size) & (y_coord >= 0) & (y_coord < self.image_size)
# #         x_coord = x_coord[valid_indices]
# #         y_coord = y_coord[valid_indices]
        
# #         input_data[0, y_coord, x_coord] = 1  # occupied
        
# #         # intensity feature 를 사용하지 않음
# #         # input_data[1, y_coord, x_coord] = torch.maximum(input_data[1, y_coord, x_coord], intensities[valid_indices])  # max intensity per pixel
        
# #         # input_data[2, y_coord, x_coord] += 1  # density
# #         input_data[1, y_coord, x_coord] += 1  # density

# #         return input_data

# #     def cartesian_to_pixel(self, x, y):
# #         '''
# #         Converts cartesian coordinates to pixel coordinates.
# #         '''
# #         pixel_x = int(x / self.pixelsize + self.image_size / 2)
# #         pixel_y = int(y / self.pixelsize + self.image_size / 2)
# #         return pixel_x, pixel_y

# #     def visualize(self, index, show_preprocessed=True, show_gt=True, show_raw=True):
# #         '''
# #         Visualizes the data for a given index.
# #         '''
# #         config = [show_preprocessed, show_gt, show_raw]
# #         plot_rows = sum(1 for c in config if c)
# #         if plot_rows == 0:
# #             print("No plots selected!")
# #             return

# #         # fig, axs = plt.subplots(plot_rows + 1, 3, figsize=(10, 15))
# #         fig, axs = plt.subplots(plot_rows + 1, 2, figsize=(10, 15))
# #         input, gt, data, free = self.__getitem__(index)
# #         if self.transform is not None:
# #             transform_names = ', '.join([t.__class__.__name__ for t in self.transform.transforms])
# #         else:
# #             transform_names = 'None'

# #         axs[0, 0].axis('off')
# #         axs[0, 1].axis('off')
# #         axs[0, 2].axis('off')
# #         axs[0, 0].set_title('Dataset Info')
# #         axs[0, 0].text(0, 0.3, f'Length of dataset: {self.len}\
# #                                 \nPath: {self.dataset_path}\
# #                                 \nTransforms: {transform_names}\
# #                                 \n\nIndex: {index}\
# #                                 \nFree track: {free}\
# #                                 \nPixel size: {self.pixelsize}\
# #                                 \nImage size: {self.image_size}\
# #                                 \nGaussian radius: {self.sx}', fontsize=10)

# #         plot_nr = 1
# #         if show_preprocessed:
# #             axs[plot_nr, 0].set_title('Occupancy')
# #             axs[plot_nr, 0].imshow(input[0], cmap='plasma')
# #             # axs[plot_nr, 1].imshow(input[1], cmap='plasma')
# #             # axs[plot_nr, 1].set_title('Intensity')
# #             # axs[plot_nr, 2].imshow(input[2], cmap='plasma')
# #             # axs[plot_nr, 2].set_title('Density')
# #             axs[plot_nr, 1].imshow(input[1], cmap='plasma')
# #             axs[plot_nr, 1].set_title('Density')

# #             for i in range(self.feature_size):
# #                 axs[plot_nr, i].axis('off')
# #                 axs[plot_nr, i].scatter(self.image_size // 2, self.image_size // 2, label='Ego Position', color='g')
# #                 xpix, ypix = self.cartesian_to_pixel(data[0], data[1])
# #                 axs[plot_nr, i].scatter(xpix, ypix, label='GT Position', color='r')
# #                 axs[plot_nr, i].quiver(xpix, ypix, data[2], data[3], label='GT Velocity', color='r')
# #                 yaw_degrees = np.rad2deg(data[4])
# #                 rectangle = patches.Rectangle((xpix - 2, ypix - 4), 8, 4, angle=yaw_degrees, fill=False, color='r')
# #                 axs[plot_nr, i].add_patch(rectangle)


# #             # axs[plot_nr, 2].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
# #             axs[plot_nr, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
# #             plot_nr += 1

# #         if show_gt:
# #             axs[plot_nr, 0].imshow(gt, cmap='plasma')
# #             axs[plot_nr, 0].set_title('GT heatmap')
# #             axs[plot_nr, 1].imshow(np.clip(input[0] + gt, 0, 1), cmap='plasma')
# #             axs[plot_nr, 1].set_title('Occupancy + GT heatmap')
# #             axs[plot_nr, 0].axis('off')
# #             axs[plot_nr, 1].axis('off')
# #             # axs[plot_nr, 2].axis('off')
# #             plot_nr += 1

# #         if show_raw:
# #             axs[plot_nr, 0].plot(self.lidar_data)
# #             axs[plot_nr, 0].set_title('Raw lidar ranges')
# #             # axs[plot_nr, 1].plot(self.intensities)
# #             # axs[plot_nr, 1].set_title('Raw lidar intensities')

# #             # --- visualize에서도 동적 cos/sin 사용 ---
# #             N = self.lidar_data.shape[0]
# #             angles = self.angle_min + torch.arange(N, dtype=self.lidar_data.dtype, device=self.lidar_data.device) * self.angle_increment
# #             cos = torch.cos(angles)
# #             sin = torch.sin(angles)
# #             x_raw = self.lidar_data * cos
# #             y_raw = self.lidar_data * sin

# #             axs[plot_nr, 2].scatter(x_raw.cpu(), y_raw.cpu(), s=0.1, label='Scans', alpha=float(self.intensities.mean().item()) if isinstance(self.intensities, torch.Tensor) else 0.5)
# #             axs[plot_nr, 2].scatter(self.data_for_plot[0], self.data_for_plot[1], color='r', label='GT-Pos')
# #             axs[plot_nr, 2].text(self.data_for_plot[0], self.data_for_plot[1], 'GT-Pos')

# #             # Adjusting view, focusing on GT-position
# #             dx = dy = 2
# #             axs[plot_nr, 2].set_xlim(self.data_for_plot[0] - dx, self.data_for_plot[0] + dx)
# #             axs[plot_nr, 2].set_ylim(self.data_for_plot[1] - dy, self.data_for_plot[1] + dy)
# #             axs[plot_nr, 2].set_xlabel('X coordinate')
# #             axs[plot_nr, 2].set_ylabel('Y coordinate')
# #             axs[plot_nr, 2].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
# #             axs[plot_nr, 2].set_title('Raw lidar data')


# class RandomRotation_before:
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
#         input_rotated = []
#         for i in range(input.shape[0]):
#             input_pil = transforms.functional.to_pil_image(input[i])
#             input_rotated_tensor = transforms.ToTensor()(transforms.functional.rotate(input_pil, angle))
#             input_rotated.append(input_rotated_tensor)
#         input = torch.stack(input_rotated, dim=1)

#         heatmap_image = transforms.functional.to_pil_image(heatmap)
#         rotated_hm_image = transforms.functional.rotate(heatmap_image, angle)
#         heatmap = transforms.ToTensor()(rotated_hm_image)

#         rotation_matrix = torch.FloatTensor([[np.cos(angle_rad), -np.sin(angle_rad)],
#                                             [np.sin(angle_rad),  np.cos(angle_rad)]])

#         # Apply the rotation
#         data[0:2] = torch.matmul(rotation_matrix, data[0:2])
#         data[2:4] = torch.matmul(rotation_matrix.T, data[2:4])
#         data[4] = (data[4] - math.radians(angle)) % (2 * math.pi)
#         if data[4] > math.pi:
#             data[4] -= 2 * math.pi

#         return input.view(self.feature_size, self.image_size, self.image_size), heatmap.view(self.image_size, self.image_size), data.view(5)


# class RandomFlip_before:
#     '''
#     Randomly flips the input data and the ground truth data.
#     '''
#     def __init__(self, p=0.5):
#         self.p = p

#     def __call__(self, sample):
#         input, heatmap, data = sample
#         if random.random() < self.p:
#             input = torch.flip(input, [1])
#             heatmap = torch.flip(heatmap, [0])
#             data[1] = -data[1]
#             data[3] = -data[3]
#             data[4] = -data[4]
#         return input, heatmap, data


# ################ OLD IMPLEMENTATIONS ####################

# class LidarDatasetOD(Dataset):
#     '''V1, Not used anymore'''
#     def __init__(self, dataset_path):
#         self.dataset_path = dataset_path
#         self.use_heatmaps = True
#         self.pixelsize = 0.025  # size of a pixel in meters
#         self.image_size = 256   # size of the image for preprocessing
#         self.feature_size = 3   # number of features in the preprocessed data
#         self.origin_offset = (self.image_size//2) * self.pixelsize
#         self.sx = self.sy = 5   # standard deviation of the gaussian peaks
#         self.len = None

#         # 동적 계산에 필요한 각도 파라미터
#         self.angle_min = -2.356194496154785
#         self.angle_increment = 0.004363323096185923

#     def __getitem__(self, index):
#         df = pd.read_csv(self.dataset_path, skiprows=index-1, nrows=1, header=None,
#                          names=['lidar', 'intensities', 'x', 'y', 'vx', 'vy', 'yaw'])
#         if len(df) == 0:
#             raise IndexError
#         df.loc[0, 'lidar'] = df.loc[0, 'lidar'].replace('(', '').replace(')', '')
#         df.loc[0, 'intensities'] = df.loc[0, 'intensities'].replace('(', '').replace(')', '')
#         row = df.iloc[0]
#         lidar_data = torch.tensor(np.fromstring(df.loc[0, 'lidar'], dtype=float, sep=', '), dtype=torch.float32)
#         intensities = torch.tensor(np.fromstring(df.loc[0, 'intensities'], dtype=float, sep=','), dtype=torch.float32)
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
#         return A * np.exp(-((x - x0)**2 / (2 * sx**2) + (y - y0)**2 / (2 * sy**2)))

#     def preprocess(self, lidar_data, intensities, data):
#         '''
#         Preprocesses the data. Convert polar coordinates to cartesian coordinates and discretize into a 256x256 grid.
#         '''
#         self.use_heatmaps = True

#         # --- 동적 각도 계산 ---
#         N = lidar_data.shape[0]
#         angles_np = self.angle_min + np.arange(N, dtype=np.float32) * self.angle_increment
#         cos = torch.from_numpy(np.cos(angles_np)).to(lidar_data.dtype)
#         sin = torch.from_numpy(np.sin(angles_np)).to(lidar_data.dtype)

#         input_data = torch.zeros((self.feature_size, self.image_size, self.image_size), dtype=torch.float32)
#         x = lidar_data * cos
#         y = lidar_data * sin
#         x_coord = ((x + self.origin_offset) / self.pixelsize)
#         y_coord = ((y + self.origin_offset) / self.pixelsize)
#         x_coord = x_coord.to(torch.int)
#         y_coord = y_coord.to(torch.int)
#         valid_indices = (x_coord >= 0) & (x_coord < self.image_size) & (y_coord >= 0) & (y_coord < self.image_size)
#         x_coord = x_coord[valid_indices]
#         y_coord = y_coord[valid_indices]
#         input_data[0, y_coord, x_coord] = 1
#         input_data[1, y_coord, x_coord] = torch.maximum(input_data[1, y_coord, x_coord], intensities[valid_indices])
#         input_data[2, y_coord, x_coord] += 1

#         # heatmap
#         self.heatmaps = torch.zeros(self.image_size, self.image_size)
#         X, Y = np.meshgrid(np.arange(self.image_size), np.arange(self.image_size))
#         x0 = int((data[0] + self.origin_offset) / self.pixelsize)
#         y0 = int((data[1] + self.origin_offset) / self.pixelsize)
#         heatmap = self.gaussian_2d(X, Y, x0, y0, self.sx, self.sy, 1)
#         if data[0] < 0:  # (구현 원문 유지)
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
#         self.pixelsize = 0.025  # size of a pixel in meters
#         self.image_size = 256   # size of the image for preprocessing
#         self.feature_size = 3   # number of features in the preprocessed data
#         self.origin_offset = (self.image_size//2) * self.pixelsize
#         self.sx = self.sy = 5   # standard deviation of the gaussian peaks
#         self.len = None
#         self.seq_len = 2        # number of frames in a sequence
#         self.number_of_sets = None

#         # 동적 각도 계산용 파라미터
#         self.angle_min = -2.356194496154785
#         self.angle_increment = 0.004363323096185923

#         self.setup()

#     def setup(self):
#         df = pd.read_csv(self.dataset_path, header=None, names=['setid', 'lidar', 'intensities', 'x', 'y', 'vx', 'vy', 'yaw'])
#         self.number_of_sets = df.max()['setid']
#         print("Number of sets", self.number_of_sets)
#         self.len = len(df) - 1 - self.number_of_sets
#         print("Length of Dataset", self.len)
#         print("Dataset Setup!")

#     def change_pixel_size(self, pixelsize):
#         self.pixelsize = pixelsize
#         self.origin_offset = (self.image_size//2) * self.pixelsize
#         print("Pixel size changed to: ", self.pixelsize)
#         print("Origin offset changed to: ", self.origin_offset)

#     def __getitem__(self, index):
#         seq_data = []
#         df = pd.read_csv(self.dataset_path, skiprows=index-1, nrows=self.seq_len, header=None,
#                          names=['setid', 'lidar', 'intensities', 'x', 'y', 'vx', 'vy', 'yaw'])
#         if len(df) == 0:
#             raise IndexError
#         if df.iloc[0]['setid'] != df.iloc[-1]['setid']:
#             return self.__getitem__(index + 1)
#         for i in range(self.seq_len):
#             df.loc[i, 'lidar'] = df.loc[i, 'lidar'].replace('(', '').replace(')', '')
#             df.loc[i, 'intensities'] = df.loc[i, 'intensities'].replace('(', '').replace(')', '')
#             row = df.iloc[i]
#             lidar_data = torch.tensor(np.fromstring(df.loc[i, 'lidar'], dtype=float, sep=', '), dtype=torch.float32)
#             intensities = torch.tensor(np.fromstring(df.loc[i, 'intensities'], dtype=float, sep=','), dtype=torch.float32)
#             intensities = (intensities - intensities.min()) / (intensities.max() - intensities.min())  # normalize intensities
#             seq_data.append(self.preprocess(lidar_data, intensities))





# # import torch
# # import numpy as np
# # import pandas as pd
# # import csv
# # from torch.utils.data import Dataset, DataLoader, random_split
# # from torchvision import transforms
# # import random
# # import math
# # import os
# # import matplotlib.pyplot as plt
# # import matplotlib.patches as patches


# # class CenterSpeedDataset(Dataset):
# #     '''
# #     Dataset class for the CenterSpeed dataset.
# #     '''
# #     def __init__(self, dataset_path, transform=None, dense=False):
# #         self.dataset_path = dataset_path
# #         self.transform = transform
# #         self.use_heatmaps = True
# #         self.dense = dense
# #         self.consider_free_paths = True
# #         self.pixelsize = 0.08#size of a pixel in meters
# #         self.image_size = 64 #size of the image for preprocessing
# #         self.feature_size = 3 #number of features in the preprocessed data
# #         self.origin_offset = (self.image_size//2) * self.pixelsize
# #         self.sx = self.sy = 5 #standard deviation of the gaussian peaks
# #         self.len = None
# #         self.seq_len = 2 #number of frames in a sequence
# #         self.number_of_sets = None
# #         self.cos = np.cos(np.arange(-2.356194496154785, 2.356194496154785 ,0.004363323096185923))
# #         self.sin = np.sin(np.arange(-2.356194496154785, 2.356194496154785 ,0.004363323096185923))
# #         self.setup()

# #     def setup(self):
# #         '''
# #         Sets up the dataset by reading the files and determining the number of rows in each file.
# #         '''
# #         self.file_paths = [os.path.join(self.dataset_path, f) for f in os.listdir(self.dataset_path) if f.endswith('.csv')]
# #         self.len = 0
# #         self.file_indices = []
# #         num_rows_per_file = []
# #         for file_path in self.file_paths:
# #                 num_rows = sum(1 for row in open(file_path))- 1 -(self.seq_len-1) #subtract 2 because of the header and the last row
# #                 self.file_indices.append((self.len, self.len+num_rows))
# #                 self.len += num_rows
# #                 num_rows_per_file.append(num_rows)
# #         for path in self.file_paths:
# #             print("Reading the following files: ", path)
# #             print("Number of entries: ", num_rows_per_file[self.file_paths.index(path)])

# #         print("Number of rows: ", self.len)
# #         print("File indices: ", self.file_indices)

# #     def change_pixel_size(self, pixelsize):
# #         '''
# #         Changes the pixel size and the origin offset accordingly.

# #         Args:
# #             pixelsize (int): New pixel size in meters.
# #         '''
# #         self.pixelsize = pixelsize
# #         self.origin_offset = (self.image_size//2) * self.pixelsize
# #         print("Pixel size changed to: ", self.pixelsize)
# #         print("Origin offset changed to: ", self.origin_offset)

# #     def change_image_size(self, image_size):
# #         '''
# #         Changes the image size and the origin offset accordingly.

# #         Args:
# #             image_size (int): New image size in pixels.
# #         '''
# #         self.image_size = int(image_size)
# #         self.origin_offset = (self.image_size//2) * self.pixelsize
# #         print("Image size changed to: ", self.image_size)
# #         print("Origin offset changed to: ", self.origin_offset)

# #     def __getitem__(self, index):
# #         '''
# #         Returns the preprocessed data and the ground truth data for a given index.

# #         Args:
# #             index: Index of the data to be returned.

# #         Returns:
# #             input_data: Preprocessed data in the form of a tensor of size (3, 64, 64).
# #             heatmap: Ground truth heatmap in the form of a tensor of size (64, 64).
# #             data: Ground truth data in the form of a tensor of size (5).
# #             free: Boolean indicating whether the path is free or not.
# #             '''
# #         free = False
# #         # Determine which file the data should come from
# #         file_index = next(i for i, (start, end) in enumerate(self.file_indices) if start <= index < end)
# #         row_index = index - self.file_indices[file_index][0]

# #         seq_data = []
# #         df = pd.read_csv(self.file_paths[file_index], skiprows=row_index, nrows=self.seq_len, header=None, names=['lidar','intensities','x','y','vx','vy','yaw'])
# #         if len(df) == 0:
# #             raise IndexError
# #         for i in range(self.seq_len):
# #             df.loc[i, 'lidar'] = df.loc[i, 'lidar'].replace('(', '').replace(')', '')
# #             df.loc[i, 'intensities'] = df.loc[i, 'intensities'].replace('(', '').replace(')', '')
# #             row = df.iloc[i]
# #             self.lidar_data = torch.tensor(np.fromstring(df.loc[i, 'lidar'], dtype=float, sep=', '), dtype=torch.float32)
# #             intensities = torch.tensor(np.fromstring(df.loc[i, 'intensities'], dtype=float, sep=','), dtype=torch.float32)
# #             try:
# #                 self.intensities = (intensities - intensities.min()) / (intensities.max() - intensities.min())  # normalize intensities
# #             except:
# #                 print("Intensities: ", intensities)
# #                 print("Row: ", row)
# #                 print("Index: ", index)
# #             seq_data.append(self.preprocess(self.lidar_data, self.intensities))
# #             if i == self.seq_len - 1:
# #                 data = torch.tensor(row[2:].values.astype(float), dtype=torch.float32)
# #                 self.data_for_plot = data.numpy().copy()
# #                 heatmap = self.heatmap(data)
# #         input_data = torch.stack([item for item in seq_data]).view(self.seq_len*3,self.image_size,self.image_size)

# #         if data[0] < 0 or np.sqrt(data[0]**2+ data[1]**2) > 3:
# #             free = True

# #         if self.transform:
# #             input_data, heatmap, data = self.transform((input_data, heatmap, data))

# #         if self.dense:
# #             print(f'Using dense features with data: {data}')
# #             dense_features = self.populate_dense_features(data=data)
# #             return input_data.view(self.feature_size*self.seq_len, self.image_size, self.image_size), heatmap.view(self.image_size, self.image_size), data.view(5), dense_features, free

# #         return input_data.view(self.feature_size*self.seq_len, self.image_size, self.image_size), heatmap.view(self.image_size, self.image_size), data.view(5), free


# #     def __len__(self):
# #         '''
# #         Returns the length of the dataset.
# #         '''
# #         if self.len is not None:
# #             return self.len
# #         else:
# #             with open(self.dataset_path, 'r') as f:
# #                 self.len = sum(1 for row in csv.reader(f))
# #                 return self.len

# #     def gaussian_2d(self, x, y, x0, y0, sx, sy, A):
# #         '''
# #         2D Gaussian function.

# #         Args:
# #             x: x-coordinate
# #             y: y-coordinate
# #             x0: x-coordinate of the peak
# #             y0: y-coordinate of the peak
# #             sx: standard deviation in x
# #             sy: standard deviation in y
# #             A: amplitude'''
# #         return A * np.exp(-((x - x0)**2 / (2 * sx**2) + (y - y0)**2 / (2 * sy**2)))

# #     def populate_dense_features(self, data) -> torch.Tensor:
# #         '''
# #         Populates a tensor with dense speed and orientation values.

# #         Args:
# #             x: x-coordinate of the peak
# #             y: y-coordinate of the peak
# #             values: List of values to be populated in the tensor.
# #         '''
# #         tensor = torch.zeros((self.image_size, self.image_size, 3), dtype=torch.float32)
# #         x,y = np.meshgrid(np.arange(self.image_size), np.arange(self.image_size))
# #         x0 = int((data[0] + self.origin_offset) / self.pixelsize)
# #         y0 = int((data[1] + self.origin_offset) / self.pixelsize)
# #         print(f'Data length: {len(data)}')
# #         for i in range(3):
# #             tensor[:,:,i] = self.gaussian_2d(x, y, x0, y0, self.sx, self.sy, data[i+2])

# #             if self.consider_free_paths:
# #                 if data[0] < 0 or np.sqrt(data[0]**2+ data[1]**2) > 3:#the other car is behind us, no peak in the heatmap
# #                     tensor = torch.zeros((self.image_size, self.image_size, 3), dtype=torch.float32)
# #                     print(f'Car Behind, setting zero')
# #                     return tensor


# #         return tensor



# #     def heatmap(self, data):
# #         '''
# #         Creates a heatmap from the ground truth data.
# #         '''
# #         self.heatmaps = torch.zeros(self.image_size, self.image_size)
# #         x,y = np.meshgrid(np.arange(self.image_size), np.arange(self.image_size))
# #         x0 = int((data[0] + self.origin_offset) / self.pixelsize)
# #         y0 = int((data[1] + self.origin_offset) / self.pixelsize)
# #         heatmap = self.gaussian_2d(x, y, x0, y0, self.sx, self.sy, 1)
# #         if self.consider_free_paths:
# #             if data[0] < 0 or np.sqrt(data[0]**2+ data[1]**2) > 3:#the other car is behind us, no peak in the heatmap
# #                 heatmap = np.zeros((self.image_size, self.image_size))
# #         heatmap = torch.tensor(heatmap, dtype=torch.float32)
# #         return heatmap

# #     def preprocess(self, lidar_data, intensities):
# #         '''
# #         Preprocesses the data. Convert polar coordinates to cartesian coordinates and discretize into an image.
# #         Creates 3 feature maps: one for the occupancy, one for the intensity and one for the number of points in a pixel.

# #         Args:
# #             lidar_data: Lidar data in the form of a tensor of size (n).
# #             intensities: Intensity data in the form of a tensor of size (n).

# #         Returns:
# #             input_data: Preprocessed data in the form of a tensor of size (3, 64, 64).

# #         '''
# #         self.use_heatmaps = True#use heatmaps for training after preprocessing
# #         #preprocess the lidar data
# #         input_data = torch.zeros((self.feature_size, self.image_size, self.image_size), dtype=torch.float32)
# #         x = lidar_data * self.cos
# #         y = lidar_data * self.sin
# #         x_coord = ((x + self.origin_offset) / self.pixelsize)
# #         y_coord = ((y + self.origin_offset) / self.pixelsize)
# #         x_coord = x_coord.to(torch.int)
# #         y_coord = y_coord.to(torch.int)
# #         valid_indices = (x_coord >= 0) & (x_coord < self.image_size) & (y_coord >= 0) & (y_coord < self.image_size)
# #         x_coord = x_coord[valid_indices]
# #         y_coord = y_coord[valid_indices]
# #         input_data[0,y_coord, x_coord] = 1 #set the pixel to occupied
# #         input_data[1,y_coord, x_coord] = torch.maximum(input_data[ 1,y_coord,x_coord], intensities[valid_indices])#store the maximum intensity value in the pixel
# #         input_data[2,y_coord, x_coord] +=1 #count the number of points in the pixel

# #         return input_data

# #     def cartesian_to_pixel(self, x, y):
# #         '''
# #         Converts cartesian coordinates to pixel coordinates.
# #         '''
# #         pixel_x = int(x / self.pixelsize + self.image_size / 2)
# #         pixel_y = int(y / self.pixelsize + self.image_size / 2)
# #         return pixel_x, pixel_y


# #     def visualize(self, index, show_preprocessed=True, show_gt=True, show_raw=True):
# #         '''
# #         Visualizes the data for a given index.

# #         Args:
# #             index: Index of the data to be visualized.
# #             show_preprocessed: Boolean indicating whether to show the preprocessed data.
# #             show_gt: Boolean indicating whether to show the ground truth data.
# #             show_raw: Boolean indicating whether to show the raw data.
# #         '''
# #         config = [show_preprocessed, show_gt, show_raw]
# #         plot_rows = 0
# #         for c in config:
# #             if c:
# #                 plot_rows += 1
# #         if plot_rows == 0:
# #             print("No plots selected!")
# #             return

# #         fig, axs = plt.subplots(plot_rows+1, 3, figsize=(10, 15))
# #         input, gt, data, free = self.__getitem__(index)
# #         if self.transform is not None:
# #             transform_names = ', '.join([t.__class__.__name__ for t in self.transform.transforms])
# #         else:
# #             transform_names = 'None'

# #         axs[0,0].axis('off')
# #         axs[0,1].axis('off')
# #         axs[0,2].axis('off')
# #         axs[0,0].set_title('Dataset Info')
# #         axs[0,0].text(0, 0.3, f'Length of dataset: {self.len}\
# #                                 \nPath: {self.dataset_path}\
# #                                 \nTransforms: {transform_names}\
# #                                 \n\nIndex: {index}\
# #                                 \nFree track: {free}\
# #                                 \nPixel size: {self.pixelsize}\
# #                                 \nImage size: {self.image_size}\
# #                                 \nGaussian radius: {self.sx}', fontsize=10)

# #         plot_nr = 1
# #         if show_preprocessed:
# #             axs[plot_nr, 0].set_title('Occupancy')
# #             axs[plot_nr, 0].imshow(input[0], cmap='plasma')
# #             axs[plot_nr, 1].imshow(input[1], cmap='plasma')
# #             axs[plot_nr, 1].set_title('Intensity')
# #             axs[plot_nr, 2].imshow(input[2], cmap='plasma')
# #             axs[plot_nr, 2].set_title('Density')
# #             for i in range(3):
# #                 axs[plot_nr, i].axis('off')
# #                 axs[plot_nr,i].scatter(self.image_size//2,self.image_size//2, label='Ego Position', color='g')
# #                 x,y = self.cartesian_to_pixel(data[0],data[1])
# #                 axs[plot_nr,i].scatter(x,y, label='GT Position', color='r')
# #                 axs[plot_nr,i].quiver(x,y ,data[2],data[3], label='GT Velocity', color='r')
# #                 yaw_degrees = np.rad2deg(data[4])
# #                 rectangle = patches.Rectangle((x-2, y-4), 8, 4, angle=yaw_degrees, fill=False, color='r')
# #                 axs[plot_nr, i].add_patch(rectangle)
# #             axs[plot_nr, 2].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
# #             plot_nr += 1

# #         if show_gt:
# #             axs[plot_nr, 0].imshow(gt, cmap='plasma')
# #             axs[plot_nr, 0].set_title('GT heatmap')
# #             axs[plot_nr, 1].imshow(np.clip(input[0]+gt, 0, 1), cmap='plasma')
# #             axs[plot_nr, 1].set_title('Occupancy + GT heatmap')
# #             axs[plot_nr, 0].axis('off')
# #             axs[plot_nr, 1].axis('off')
# #             axs[plot_nr, 2].axis('off')
# #             plot_nr += 1

# #         if show_raw:
# #             axs[plot_nr, 0].plot(self.lidar_data)
# #             axs[plot_nr, 0].set_title('Raw lidar ranges')
# #             axs[plot_nr, 1].plot(self.intensities)
# #             axs[plot_nr, 1].set_title('Raw lidar intensities')
# #             x = self.lidar_data * self.cos
# #             y = self.lidar_data * self.sin
# #             axs[plot_nr,2].scatter(x, y, s=0.1, label='Scans', alpha=self.intensities)
# #             axs[plot_nr,2].scatter(self.data_for_plot[0], self.data_for_plot[1], color='r', label='GT-Pos')
# #             axs[plot_nr,2].text(self.data_for_plot[0], self.data_for_plot[1],'GT-Pos')
# #             # Adjusting view, focusing on GT-position
# #             dx = dy = 2
# #             axs[plot_nr,2].set_xlim(self.data_for_plot[0] - dx, self.data_for_plot[0] + dx)
# #             axs[plot_nr,2].set_ylim(self.data_for_plot[1] - dy,self.data_for_plot[1] + dy)
# #             axs[plot_nr,2].set_xlabel('X coordinate')
# #             axs[plot_nr,2].set_ylabel('Y coordinate')
# #             axs[plot_nr, 2].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
# #             axs[plot_nr, 2].set_title('Raw lidar data')





# # class RandomRotation:
# #     '''
# #     Randomly rotates the input data and the ground truth data.
# #     '''
# #     def __init__(self, angle=45, image_size=64, feature_size=6):
# #         self.angle = angle
# #         self.image_size = image_size
# #         self.feature_size = feature_size

# #     def __call__(self, sample):
# #         input, heatmap, data = sample
# #         angle = random.uniform(-self.angle, self.angle)
# #         angle_rad = -math.radians(angle)
# #         #print("THis was rotated by: ", angle)
# #         input_rotated = []
# #         for i in range(input.shape[0]):
# #             input_pil = transforms.functional.to_pil_image(input[i])
# #             input_rotated_tensor = transforms.ToTensor()(transforms.functional.rotate(input_pil, angle))
# #             input_rotated.append(input_rotated_tensor)
# #         input = torch.stack(input_rotated, dim = 1)


# #         heatmap_image = transforms.functional.to_pil_image(heatmap)
# #         rotated_hm_image = transforms.functional.rotate(heatmap_image, angle)
# #         heatmap = transforms.ToTensor()(rotated_hm_image)

# #         rotation_matrix = torch.FloatTensor([[np.cos(angle_rad), -np.sin(angle_rad)],
# #                                         [np.sin(angle_rad), np.cos(angle_rad)]])

# #         # Apply the rotation
# #         data[0:2] = torch.matmul(rotation_matrix, data[0:2])
# #         data[2:4] = torch.matmul(rotation_matrix.T, data[2:4])
# #         data[4] = (data[4] - math.radians(angle))% (2*math.pi)
# #         if data[4] > math.pi:
# #             data[4] -= 2*math.pi

# #         return input.view(self.feature_size,self.image_size,self.image_size), heatmap.view(self.image_size,self.image_size), data.view(5)

# # class RandomFlip:
# #     '''
# #     Randomly flips the input data and the ground truth data.
# #     '''
# #     def __init__(self, p=0.5):
# #         self.p = p

# #     def __call__(self, sample):
# #         input, heatmap, data = sample
# #         if random.random() < self.p:
# #             #print("This was flipped")
# #             input = torch.flip(input, [1])
# #             heatmap = torch.flip(heatmap, [0])
# #             data[1] = -data[1]
# #             data[3] = -data[3]
# #             data[4] = -data[4]
# #         return input, heatmap, data


# # ################OLD IMPLEMENTATIONS####################

# # class LidarDatasetOD(Dataset):
# #     '''V1, Not used anymore'''
# #     def __init__(self, dataset_path):
# #         self.dataset_path = dataset_path
# #         self.use_heatmaps = True
# #         self.pixelsize = 0.025#size of a pixel in meters, was 0.015 i think this bigger makes more sense for FOV
# #         self.image_size = 256 #size of the image for preprocessing
# #         self.feature_size = 3 #number of features in the preprocessed data
# #         self.origin_offset = (self.image_size//2) * self.pixelsize
# #         self.sx = self.sy = 5 #standard deviation of the gaussian peaks
# #         self.len = None


# #     def __getitem__(self, index):
# #         df = pd.read_csv(self.dataset_path, skiprows=index-1, nrows=1, header=None, names=['lidar','intensities','x','y','vx','vy','yaw'])
# #         if len(df) == 0:
# #             raise IndexError
# #         df.loc[0, 'lidar'] = df.loc[0, 'lidar'].replace('(', '').replace(')', '')
# #         df.loc[0, 'intensities'] = df.loc[0, 'intensities'].replace('(', '').replace(')', '')
# #         row = df.iloc[0]
# #         lidar_data = torch.tensor(np.fromstring(df.loc[0, 'lidar'], dtype=float, sep=', '), dtype=torch.float32)
# #         intensities = torch.tensor(np.fromstring(df.loc[0, 'intensities'], dtype=float, sep=','), dtype=torch.float32)
# #         #print(len(intensities))
# #         intensities = (intensities - intensities.min()) / (intensities.max() - intensities.min())  # normalize intensities
# #         data = torch.tensor(row[2:].values.astype(float), dtype=torch.float32)

# #         return self.preprocess(lidar_data, intensities, data)

# #     def __len__(self):
# #         if self.len is not None:
# #             return self.len
# #         else:
# #             with open(self.dataset_path, 'r') as f:
# #                 self.len = sum(1 for row in csv.reader(f))
# #                 return self.len

# #     def gaussian_2d(self, x, y, x0, y0, sx, sy, A):
# #         '''
# #         2D Gaussian function.

# #         Args:
# #             x: x-coordinate
# #             y: y-coordinate
# #             x0: x-coordinate of the peak
# #             y0: y-coordinate of the peak
# #             sx: standard deviation in x
# #             sy: standard deviation in y
# #             A: amplitude'''
# #         return A * np.exp(-((x - x0)**2 / (2 * sx**2) + (y - y0)**2 / (2 * sy**2)))


# #     def preprocess(self, lidar_data, intensities, data):
# #         '''
# #         Preprocesses the data. Convert polar coordinates to cartesian coordinates and discretize into a 256x256 grid.
# #         Stores these grids in a new tensor.
# #         Completely vectorized, efficient asf!
# #         Does it make sense to put the origin in the middle of the grid?
# #         Maybe it is better to put it in the bottom left corner? Or closer to the corner?
# #         '''

# #         self.use_heatmaps = True#use heatmaps for training after preprocessing
# #         #preprocess the lidar data
# #         input_data = torch.zeros((self.feature_size, self.image_size, self.image_size), dtype=torch.float32)
# #         x = lidar_data * np.cos(np.arange(-2.356194496154785, 2.356194496154785 ,0.004363323096185923))
# #         y = lidar_data * np.sin(np.arange(-2.356194496154785, 2.356194496154785 ,0.004363323096185923))
# #         x_coord = ((x + self.origin_offset) / self.pixelsize)
# #         y_coord = ((y + self.origin_offset) / self.pixelsize)
# #         x_coord = x_coord.to(torch.int)
# #         y_coord = y_coord.to(torch.int)
# #         valid_indices = (x_coord >= 0) & (x_coord < self.image_size) & (y_coord >= 0) & (y_coord < self.image_size)
# #         x_coord = x_coord[valid_indices]
# #         y_coord = y_coord[valid_indices]
# #         input_data[0,y_coord, x_coord] = 1 #set the pixel to occupied
# #         input_data[1,y_coord, x_coord] = torch.maximum(input_data[ 1,y_coord,x_coord], intensities[valid_indices])#store the maximum intensity value in the pixel
# #         input_data[2,y_coord, x_coord] +=1 #count the number of points in the pixel

# #         #preprocess the gt's into heatmaps

# #         self.heatmaps = torch.zeros(self.image_size, self.image_size)
# #         x,y = np.meshgrid(np.arange(self.image_size), np.arange(self.image_size))

# #         x0 = int((data[0] + self.origin_offset) / self.pixelsize)
# #         y0 = int((data[1] + self.origin_offset) / self.pixelsize)
# #         heatmap = self.gaussian_2d(x, y, x0, y0, self.sx, self.sy, 1)
# #         #FIXME: i think this is the wrong place to do this because sometimes we can also see the opponent behind us!
# #         if data[0] < 0:#the other car is behind us, no peak in the heatmap
# #             heatmap = np.zeros((self.image_size, self.image_size))
# #         heatmap = torch.tensor(heatmap, dtype=torch.float32)
# #         return input_data, heatmap, data


# # class LidarDatasetSeqOD(Dataset):
# #     '''
# #     V2, not used anymore
# #     '''
# #     def __init__(self, dataset_path, transform=None):
# #         self.dataset_path = dataset_path
# #         self.transform = transform
# #         self.use_heatmaps = True
# #         self.consider_free_paths = True
# #         self.pixelsize = 0.025#size of a pixel in meters, was 0.015 i think this bigger makes more sense for FOV
# #         self.image_size = 256 #size of the image for preprocessing
# #         self.feature_size = 3 #number of features in the preprocessed data
# #         self.origin_offset = (self.image_size//2) * self.pixelsize
# #         self.sx = self.sy = 5 #standard deviation of the gaussian peaks
# #         self.len = None
# #         self.seq_len = 2 #number of frames in a sequence
# #         self.number_of_sets = None
# #         self.cos = np.cos(np.arange(-2.356194496154785, 2.356194496154785 ,0.004363323096185923))
# #         self.sin = np.sin(np.arange(-2.356194496154785, 2.356194496154785 ,0.004363323096185923))
# #         self.setup()

# #     def setup(self):
# #        df = pd.read_csv(self.dataset_path, header=None, names=['setid','lidar','intensities','x','y','vx','vy','yaw'])
# #        self.number_of_sets = df.max()['setid']
# #        print("Number of sets", self.number_of_sets)
# #        self.len = len(df) - 1 - self.number_of_sets
# #        print("Length of Dataset", self.len)
# #        print("Dataset Setup!")

# #     def change_pixel_size(self, pixelsize):
# #         self.pixelsize = pixelsize
# #         self.origin_offset = (self.image_size//2) * self.pixelsize
# #         print("Pixel size changed to: ", self.pixelsize)
# #         print("Origin offset changed to: ", self.origin_offset)


# #     def __getitem__(self, index):
# #         seq_data = []
# #         df = pd.read_csv(self.dataset_path, skiprows=index-1, nrows=self.seq_len, header=None, names=['setid','lidar','intensities','x','y','vx','vy','yaw'])
# #         if len(df) == 0:
# #             raise IndexError
# #         if df.iloc[0]['setid'] != df.iloc[-1]['setid']:
# #             return self.__getitem__(index+1)
# #         for i in range(self.seq_len):
# #             df.loc[i, 'lidar'] = df.loc[i, 'lidar'].replace('(', '').replace(')', '')
# #             df.loc[i, 'intensities'] = df.loc[i, 'intensities'].replace('(', '').replace(')', '')
# #             row = df.iloc[i]
# #             lidar_data = torch.tensor(np.fromstring(df.loc[i, 'lidar'], dtype=float, sep=', '), dtype=torch.float32)
# #             intensities = torch.tensor(np.fromstring(df.loc[i, 'intensities'], dtype=float, sep=','), dtype=torch.float32)
# #             intensities = (intensities - intensities.min()) / (intensities.max() - intensities.min())  # normalize intensities
# #             seq_data.append(self.preprocess(lidar_data, intensities))
# #             if i == self.seq_len - 1:
# #                 data = torch.tensor(row[3:].values.astype(float), dtype=torch.float32)
# #                 heatmap = self.heatmap(data)
# #         input_data = torch.stack([item for item in seq_data]).view(self.seq_len*3,self.image_size,self.image_size)

# #         if self.transform:
# #             input_data, heatmap, data = self.transform((input_data, heatmap, data))

# #         return input_data.view(self.feature_size*self.seq_len, self.image_size, self.image_size), heatmap.view(self.image_size, self.image_size), data.view(5)


# #     def __len__(self):
# #         if self.len is not None:
# #             return self.len
# #         else:
# #             with open(self.dataset_path, 'r') as f:
# #                 self.len = sum(1 for row in csv.reader(f))
# #                 return self.len

# #     def gaussian_2d(self, x, y, x0, y0, sx, sy, A):
# #         '''
# #         2D Gaussian function.

# #         Args:
# #             x: x-coordinate
# #             y: y-coordinate
# #             x0: x-coordinate of the peak
# #             y0: y-coordinate of the peak
# #             sx: standard deviation in x
# #             sy: standard deviation in y
# #             A: amplitude'''
# #         return A * np.exp(-((x - x0)**2 / (2 * sx**2) + (y - y0)**2 / (2 * sy**2)))

# #     def heatmap(self, data):
# #         #preprocess the gt's into heatmaps

# #         self.heatmaps = torch.zeros(self.image_size, self.image_size)
# #         x,y = np.meshgrid(np.arange(self.image_size), np.arange(self.image_size))
# #         x0 = int((data[0] + self.origin_offset) / self.pixelsize)
# #         y0 = int((data[1] + self.origin_offset) / self.pixelsize)
# #         heatmap = self.gaussian_2d(x, y, x0, y0, self.sx, self.sy, 1)
# #         if self.consider_free_paths:
# #             if data[0] < 0 or np.sqrt(data[0]**2+ data[1]**2) > 3:#the other car is behind us, no peak in the heatmap
# #                 heatmap = np.zeros((self.image_size, self.image_size))
# #         heatmap = torch.tensor(heatmap, dtype=torch.float32)
# #         return heatmap

# #     def preprocess(self, lidar_data, intensities):
# #         '''
# #         Preprocesses the data. Convert polar coordinates to cartesian coordinates and discretize into a 256x256 grid.
# #         Stores these grids in a new tensor.
# #         Completely vectorized, efficient asf!
# #         Does it make sense to put the origin in the middle of the grid?
# #         Maybe it is better to put it in the bottom left corner? Or closer to the corner?
# #         '''

# #         self.use_heatmaps = True#use heatmaps for training after preprocessing
# #         #preprocess the lidar data
# #         input_data = torch.zeros((self.feature_size, self.image_size, self.image_size), dtype=torch.float32)
# #         x = lidar_data * self.cos
# #         y = lidar_data * self.sin
# #         x_coord = ((x + self.origin_offset) / self.pixelsize)
# #         y_coord = ((y + self.origin_offset) / self.pixelsize)
# #         x_coord = x_coord.to(torch.int).long()
# #         y_coord = y_coord.to(torch.int).long()
# #         valid_indices = (x_coord >= 0) & (x_coord < self.image_size) & (y_coord >= 0) & (y_coord < self.image_size)
# #         x_coord = x_coord[valid_indices]
# #         y_coord = y_coord[valid_indices]
# #         input_data[0,y_coord, x_coord] = 1 #set the pixel to occupied
# #         input_data[1,y_coord, x_coord] = torch.maximum(input_data[ 1,y_coord,x_coord], intensities[valid_indices])#store the maximum intensity value in the pixel
# #         input_data[2,y_coord, x_coord] +=1 #count the number of points in the pixel

# #         return input_data
