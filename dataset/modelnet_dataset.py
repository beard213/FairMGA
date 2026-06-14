import os
import numpy as np
import torch
from torch.utils.data import Dataset

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

class ModelNet40(Dataset):
    def __init__(self, root, split='train', num_points=1024):
        self.root = root
        self.num_points = num_points
        self.categories = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.categories)}
        
        self.datapath = []
        for cls in self.categories:
            cls_folder = os.path.join(root, cls, split)
            if not os.path.exists(cls_folder): continue
            for f in os.listdir(cls_folder):
                if f.endswith('.off'):
                    self.datapath.append((os.path.join(cls_folder, f), self.class_to_idx[cls]))

    def __getitem__(self, index):
        path, label = self.datapath[index]
        try:
            with open(path, 'r') as f:
                # 过滤掉空行和注释
                lines = [line.strip() for line in f.readlines() if line.strip() and not line.startswith('#')]
            
            # 健壮的 Header 解析
            if lines[0].startswith('OFF'):
                if len(lines[0]) > 3: # 格式：OFF 1234 567 0
                    header = lines[0][3:].split()
                    start_idx = 1
                else: # 格式：OFF \n 1234 567 0
                    header = lines[1].split()
                    start_idx = 2
            else:
                header = lines[0].split()
                start_idx = 1
            
            n_verts = int(header[0])
            verts = []
            
            # 读取顶点数据
            for i in range(n_verts):
                # 关键：[:3] 确保只取 XYZ，忽略可能存在的颜色信息
                # lines[start_idx + i] 确保偏移量正确
                v = [float(x) for x in lines[start_idx + i].split()[:3]]
                verts.append(v)
            
            verts = np.array(verts, dtype=np.float32)

            # 采样逻辑
            if len(verts) >= self.num_points:
                idx = np.random.choice(len(verts), self.num_points, replace=False)
            else:
                idx = np.random.choice(len(verts), self.num_points, replace=True)
            
            point_set = verts[idx, :]
            point_set = pc_normalize(point_set)
            
            # 返回 [3, 1024] 形状以适配 Conv1d
            return torch.from_numpy(point_set).transpose(1, 0).float(), torch.tensor(label).long()

        except Exception as e:
            # 如果某个文件损坏，随机读取另一个
            # print(f"Error loading {path}: {e}")
            return self.__getitem__(np.random.randint(0, len(self.datapath)))

    def __len__(self):
        return len(self.datapath)