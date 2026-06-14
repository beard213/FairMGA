import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms # 引入 transform
from PIL import Image

class CGMNISTDataset(Dataset):
    def __init__(self, mode='train', root='./data'):
        self.root = root
        self.mode = mode
        self.data_dir = '/data/zmy/MMPareto_ICML2024/data/colored_mnist'
        file_name = 'mnist_10color_jitter_var_0.020.npy'
        data_path = os.path.join(self.data_dir, file_name)
        
        if os.path.exists(data_path):
            raw_data = np.load(data_path, allow_pickle=True, encoding='latin1').item()
            if mode == 'train':
                self.images = raw_data['train_image']
                self.labels = raw_data['train_label']
            else:
                self.images = raw_data['test_image']
                self.labels = raw_data['test_label']
        else:
            raise FileNotFoundError(f"找不到文件 {file_name}")

        # --- 新增：定义数据增强 ---
        if self.mode == 'train':
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomGrayscale(p=0.4), # 40% 的概率变黑白，强迫学形状
                transforms.ColorJitter(brightness=0.3, contrast=0.3), # 颜色扰动
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img = self.images[index] # 假设是 (H, W, C)
        label = self.labels[index]
        
        # 使用 transform 处理
        img_tensor = self.transform(img)
            
        # 适配 MMPareto 结构
        # 技巧：为了让两个模态有差异，可以让 spec 变成纯灰度图
        spec = transforms.Grayscale(num_output_channels=3)(transforms.ToPILImage()(img))
        spec = transforms.ToTensor()(spec)
        spec = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(spec)
        
        return spec, img_tensor, int(label)