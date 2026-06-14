import h5py
import torch
import numpy as np
from torch.utils.data import Dataset

class NYUv2_Dataset(Dataset):
    def __init__(self, file_path='/data/zmy/MMPareto_ICML2024/data/NYUv2'):
        # 使用 h5py 读取 Matlab v7.3 格式数据
        self.data = h5py.File(file_path, 'r')
        self.images = self.data['images']  # [1449, 3, 640, 480]
        self.depths = self.data['depths']  # [1449, 640, 480]
        self.labels = self.data['labels']  # [1449, 640, 480]

    def __getitem__(self, index):
        # 1. 图像处理 (RGB)
        # 原维度是 (3, 640, 480), 需要转置为常用的 (3, 480, 640)
        img = np.transpose(self.images[index], (0, 2, 1))
        img = torch.from_numpy(img).float() / 255.0

        # 2. 深度图处理 (Depth)
        # 转置并增加通道维度 [1, 480, 640]
        depth = np.transpose(self.depths[index], (1, 0))
        depth = torch.from_numpy(depth).float().unsqueeze(0) 

        # 3. 语义分割标签处理 (Label Mapping)
        # 这一步最关键，直接关系到你的 CUDA 报错是否会再次出现
        raw_label = np.transpose(self.labels[index], (1, 0))
        raw_label = raw_label.astype(np.int64)
        
        # 针对 13 类任务的简易映射逻辑：
        # NYUv2 原始标签若包含 0 (未标注)，减 1 后会变成 -1，这里统一映射为 255 (ignore_index)
        temp_label = raw_label.copy()
        temp_label[temp_label > 13] = 0  # 超过 13 类的设为背景/忽略
        mapped_label = temp_label - 1    # 将 1~13 映射到 0~12
        mapped_label[mapped_label < 0] = 255 # 处理原本为 0 的标签
        
        label = torch.from_numpy(mapped_label).long()

        # 确保返回的变量名与上面定义的一致
        return img, label, depth

    def __len__(self):
        return self.images.shape[0]