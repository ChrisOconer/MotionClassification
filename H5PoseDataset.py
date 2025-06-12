import torch
from torch.utils.data import Dataset
import h5py
import numpy as np


class H5PoseDataset(Dataset):
    """从H5文件加载姿态数据的数据集，支持倒序增强"""

    def __init__(self, h5_path, group='train', segment_length=300,
                 augment=False, normalize=True, reverse_prob=0.5, mask_ratio=0.0, noise_scale=0.01,num_joints = 23, **kwargs):
        """
        参数:
            h5_path: H5文件路径
            group: 数据集组 ('train' 或 'validation')
            segment_length: 每个样本的固定长度
            augment: 是否应用数据增强（仅在group='train'时有效）
            normalize: 是否归一化数据
            reverse_prob: 序列倒序的概率（仅在augment=True时有效）
        """
        super().__init__()
        self.h5_path = h5_path
        self.group = group
        self.segment_length = segment_length
        self.augment = augment if group == 'train' else False
        self.normalize = normalize
        self.reverse_prob = reverse_prob
        self.mask_ratio = mask_ratio
        self.noise_scale = noise_scale
        self.num_joints = num_joints
        self.file = None  # 确保file属性被初始化

        # 加载样本元数据
        try:
            with h5py.File(h5_path, 'r') as f:
                self.samples = list(f[group].keys())
                self.labels = [f[f'{group}/{s}/label'][()].item() for s in self.samples]
                self.frame_counts = [f[f'{group}/{s}/data'].shape[0] for s in self.samples]
        except Exception as e:
            print(f"Error loading metadata from {h5_path}: {e}")
            self.samples = []
            self.labels = []
            self.frame_counts = []

        # 生成片段元数据
        self.segment_meta = self._generate_segment_metadata()

        self._filter_nan_inf_samples()

    def _filter_nan_inf_samples(self):
        valid_segment_meta = []
        with h5py.File(self.h5_path, 'r') as f:
            for meta in self.segment_meta:
                sample_id = meta['sample_id']
                data = f[f'{self.group}/{sample_id}/data'][
                    meta['start_frame']:meta['end_frame']
                ]
                if not (np.isnan(data).any() or np.isinf(data).any()):
                    valid_segment_meta.append(meta)
        self.segment_meta = valid_segment_meta

    def _generate_segment_metadata(self):
        """生成片段元数据（起始帧、标签等）"""
        segment_meta = []
        for i, sample_id in enumerate(self.samples):
            total_frames = self.frame_counts[i]
            label = self.labels[i]

            # 滑动窗口生成片段
            for start_idx in range(0, total_frames, self.segment_length):
                end_idx = start_idx + self.segment_length
                if end_idx > total_frames:
                    continue  # 跳过不足一帧的片段
                segment_meta.append({
                    'sample_id': sample_id,
                    'start_frame': start_idx,
                    'end_frame': end_idx,
                    'label': label
                })
        return segment_meta

    def __len__(self):
        return len(self.segment_meta)

    def __getitem__(self, idx):
        # 在每个worker中第一次调用时打开文件
        if self.file is None:
            try:
                self.file = h5py.File(self.h5_path, 'r')
            except Exception as e:
                print(f"Error opening file {self.h5_path}: {e}")
                # 返回默认值或抛出异常
                return torch.zeros(self.segment_length, self.num_joints, 3), 0

        meta = self.segment_meta[idx]
        sample_id = meta['sample_id']

        # 读取片段数据
        try:
            data = self.file[f'{self.group}/{sample_id}/data'][
                   meta['start_frame']:meta['end_frame']
                   ]
            if data.size == 0:  # 检查数据是否为空
                raise ValueError(f"Empty data for sample {sample_id}")
            if np.isnan(data).any() or np.isinf(data).any():
                print(f"Data for sample {sample_id} contains Nan or Inf values.")
            data = data[:, :self.num_joints, :]
        except Exception as e:
            print(f"Error loading data: {e}")
            # 返回默认值或抛出异常
            return torch.zeros(self.segment_length, self.num_joints, 3), 0

        label = meta['label']

        # 转换为PyTorch张量
        data = torch.tensor(data, dtype=torch.float32)

        # 数据增强（仅训练集）
        if self.augment:
            data = self._apply_augmentations(data)

        # 归一化
        if self.normalize:
            data = self._normalize(data)

        return data, label

    def _apply_augmentations(self, data):
        if self.augment:
            # 随机旋转（3D适用）
            if data.shape[-1] == 3:  # 假设是3D坐标
                data = self._random_rotation_3d(data)
            # 时间掩码（随机遮挡部分时间步）
            if self.mask_ratio > 0:
                data = self._temporal_masking(data,self.mask_ratio)
            # 原有倒序和噪声
            if np.random.random() < self.reverse_prob:
                data = torch.flip(data, dims=[0])

            data = self._add_gaussian_noise(data)

            if np.random.random() < 0.3:
                data = self._random_scale(data)

            # if np.random.random() < 0.3:
            #     data = self._time_warp(data)
        return data

    def _random_rotation_3d(self, data):
        """3D随机旋转"""
        angles = torch.rand(3) * 2 * np.pi
        Rx = torch.tensor([
            [1, 0, 0],
            [0, torch.cos(angles[0]), -torch.sin(angles[0])],
            [0, torch.sin(angles[0]), torch.cos(angles[0])]
        ], device=data.device)
        Ry = torch.tensor([
            [torch.cos(angles[1]), 0, torch.sin(angles[1])],
            [0, 1, 0],
            [-torch.sin(angles[1]), 0, torch.cos(angles[1])]
        ], device=data.device)
        Rz = torch.tensor([
            [torch.cos(angles[2]), -torch.sin(angles[2]), 0],
            [torch.sin(angles[2]), torch.cos(angles[2]), 0],
            [0, 0, 1]
        ], device=data.device)
        R = Rz @ Ry @ Rx
        return torch.einsum('ij,tkj->tki', R, data)

    def _temporal_masking(self, data, mask_ratio=0.1):
        """时间掩码：随机遮挡部分时间步"""
        T = data.size(0)
        num_mask = int(T * mask_ratio)
        if num_mask > 0:
            mask_indices = np.random.choice(T, num_mask, replace=False)
            data[mask_indices] = 0
        return data

    def _random_scale(self, data):
        """随机缩放序列"""
        if data is None or data.numel() == 0:
            return data  # 或返回默认值
        scale = np.random.uniform(0.9, 1.1)
        return data * scale

    def _time_warp(self, data):
        T = data.size(0)
        indices = torch.linspace(0, T-1, T).float()
        warp = (torch.rand(T, device=data.device) * 2 -1)*0.1*T
        indices = indices+ warp
        indices = torch.clamp(indices,0,T-1)

        indices_floor = indices.floor().long()
        indices_ceil = torch.min(indices_floor+1, torch.tensor(T-1,device=data.device))
        alpha = indices- indices_floor.float()
        data_warped = data[indices_floor] * (1-alpha).view(-1,1,1)+ data[indices_ceil]*alpha.view(-1,1,1)
        return data_warped

    def _add_gaussian_noise(self, data):
        """添加高斯噪声"""
        if data is None or data.numel() == 0:
            return data  # 或返回默认值
        noise = torch.randn_like(data) * 0.01
        return data + noise

    def _normalize(self, data):
        """以根关节（第0个关节）为中心归一化"""
        if data is None or data.numel() == 0:
            return data  # 或返回默认值
        if data.shape[1] >= 1:  # 确保存在根关节
            root_joint = data[:, 0:1, :]  # [T, 1, C]
            data = data - root_joint  # 中心化
        return data

    def __del__(self):
        """对象销毁时关闭HDF5文件"""
        # 检查属性是否存在
        if hasattr(self, 'file') and self.file is not None:
            try:
                self.file.close()
            except Exception as e:
                print(f"Error closing file: {e}")
