import csv
import math
import os
import random
import copy
import numpy as np
import torch
import torch.nn.functional
import torchaudio
from PIL import Image
from scipy import signal
from torch.utils.data import Dataset
from torchvision import transforms

    


class AVDataset_CD(Dataset):
  def __init__(self, mode='train'):
    classes = []
    self.data = []
    data2class = {}

    self.mode=mode
    # 改为你的数据集路径
    base_path = '/data/zmy/MMPareto_ICML2024/data/CREMAD'
    self.visual_path = os.path.join(base_path, 'VideoFrames')
    self.audio_path  = os.path.join(base_path, 'audio')
    self.stat_path   = os.path.join(base_path, 'stat.csv')
    self.train_txt   = os.path.join(base_path, 'train.csv')
    self.test_txt    = os.path.join(base_path, 'test.csv')
    if mode == 'train':
        csv_file = self.train_txt
    else:
        csv_file = self.test_txt

    
    with open(self.stat_path, encoding='UTF-8-sig') as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                classes.append(row[0])
    
    with open(csv_file) as f:
      csv_reader = csv.reader(f)
      for item in csv_reader:
        audio_file = os.path.join(self.audio_path, item[0] + '.pt')
        visual_dir = os.path.join(self.visual_path, item[0])
        if item[1] in classes and os.path.exists(audio_file) and os.path.exists(visual_dir):
            self.data.append(item[0])
            data2class[item[0]] = item[1]

    print('data load over')
    print(len(self.data))
    
    self.classes = sorted(classes)

    self.data2class = data2class
    self._init_atransform()
    print('# of files = %d ' % len(self.data))
    print('# of classes = %d' % len(self.classes))

    #Audio
    self.class_num = len(self.classes)

  def _init_atransform(self):
    self.aid_transform = transforms.Compose([transforms.ToTensor()])

  def __len__(self):
    return len(self.data)

  
  def __getitem__(self, idx):
    datum = self.data[idx]

    # Audio
    fbank = torch.load(os.path.join(self.audio_path, datum + '.pt')).unsqueeze(0)

    # Visual
    if self.mode == 'train':
        transf = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        transf = transforms.Compose([
            transforms.Resize(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    folder_path = os.path.join(self.visual_path, datum)
    files = sorted(os.listdir(folder_path))
    file_num = len(files)

    pick_num = 6
    seg = max(1, file_num // pick_num)   # ✅ 关键修复

    image_arr = []

    for i in range(pick_num):
        if self.mode == 'train':
            start = i * seg
            end = min((i + 1) * seg - 1, file_num - 1)

            if start > end:
                idx = random.randint(0, file_num - 1)
            else:
                idx = random.randint(start, end)
        else:
            idx = min(i * seg + seg // 2, file_num - 1)

        img_path = os.path.join(folder_path, files[idx])
        image_arr.append(transf(Image.open(img_path).convert('RGB')).unsqueeze(0))

    images = torch.cat(image_arr)

    return fbank, images, self.classes.index(self.data2class[datum]) 
    import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image
import torch

class AVDataset_KS(Dataset):
    def __init__(self, mode='train'):
        # 请根据你服务器上 KS 数据集的实际路径修改以下三行
        self.data_path = "/data/zmy/datasets/KineticSound/" 
        self.mode = mode
        
        # 加载对应的索引文件 (通常是 csv 或 pkl)
        if mode == 'train':
            self.list_path = os.path.join(self.data_path, 'train.txt')
        else:
            self.list_path = os.path.join(self.data_path, 'test.txt')

        # 解析文件列表
        with open(self.list_path, 'r') as f:
            self.file_list = [line.strip().split() for line in f.readlines()]
            
        print(f"KS {mode} 数据加载完成: 共 {len(self.file_list)} 条数据")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # 假设文件列表格式为: video_id label
        item = self.file_list[idx]
        file_id = item[0]
        label = int(item[1])

        # 1. 加载音频 (通常是预处理好的 .npy 或 .pt 频谱图)
        audio_path = os.path.join(self.data_path, 'audio', f"{file_id}.npy")
        audio = np.load(audio_path)
        audio = torch.from_numpy(audio).float() # [1, 257, 422] 类似维度

        # 2. 加载视频序列 (KS 通常取 3 帧或 6 帧)
        visual_frames = []
        for i in range(1, 4): # 示例：取前 3 帧
            frame_path = os.path.join(self.data_path, 'visual', file_id, f"frame_{i:04d}.jpg")
            frame = Image.open(frame_path).convert('RGB')
            frame = frame.resize((224, 224))
            visual_frames.append(np.array(frame))
        
        visual = np.stack(visual_frames, axis=0) # [T, H, W, C]
        visual = torch.from_numpy(visual).permute(3, 0, 1, 2).float() # [C, T, H, W]

        return audio, visual, label 