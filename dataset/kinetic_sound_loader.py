import os
import glob
import pickle
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class KineticSoundDataset(Dataset):
    def __init__(self, mode='train', data_root='/data/zmy/MMPareto_ICML2024/data/KineticSound', n_frames=4):
        self.data_root = data_root
        self.mode = mode
        self.n_frames = n_frames
        self.samples = []

        # --- 1. 建立音频索引 (针对 .pkl 特征文件) ---
        print(f"🔍 正在建立音频文件索引...")
        audio_dir = os.path.join(data_root, 'audio_spec')
        audio_index = {}
        for root, _, files in os.walk(audio_dir):
            for f in files:
                if f.endswith('.pkl'):
                    # 关键点：统一取不带后缀的文件名作为 Key
                    name_key = os.path.splitext(f)[0] 
                    audio_index[name_key] = os.path.join(root, f)
        print(f"✅ 索引完成，共找到 {len(audio_index)} 个 .pkl 文件")

# --- 2. 加载数据列表 ---
        txt_file = os.path.join(data_root, 'my_train.txt' if mode=='train' else 'my_test.txt')
        
        # 获取该模式下磁盘上实际存在的文件夹列表，建立一个快速索引
        actual_vis_dirs = os.listdir(os.path.join(data_root, 'visual', mode))
        vis_index = {d: d for d in actual_vis_dirs} 

        with open(txt_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                
                # 1. 兼容逗号和空格
                parts = line.replace(',', ' ').split()
                if len(parts) < 3: continue # absolute文件通常有3部分：音频,视频,标签
                
                # 2. 提取 ID (建议取 parts[1] 即视频路径部分，因为它不带 .wav 后缀，更稳)
                vis_path_in_txt = parts[1]
                name_id = os.path.basename(vis_path_in_txt) # 直接取文件夹名
                
                try:
                    label = int(parts[-1]) 
                except:
                    continue

                # --- 关键：路径容错处理 ---
                # 3. 确定音频 .pkl 的实际路径 (利用你之前建立的 audio_index)
                spec_path = audio_index.get(name_id)
                if spec_path is None:
                    continue # 找不到特征文件，跳过

                # 4. 确定视频帧的实际路径
                # 注意：txt 里的路径是 /data/zmy/OGM-GE/... 
                # 我们要把它映射到你当前的 /data/zmy/MMPareto_ICML2024/data/KineticSound/visual 目录下
                vis_dir = os.path.join(self.data_root, 'visual', self.mode, name_id)
                
                # 容错：如果 mode 是 test 但文件夹在 train 里（或者反过来），尝试交叉查找
                if not os.path.exists(vis_dir):
                    other_mode = 'test' if self.mode == 'train' else 'train'
                    vis_dir = os.path.join(self.data_root, 'visual', other_mode, name_id)
                
                if not os.path.exists(vis_dir):
                    continue # 还是找不到，跳过

                # 5. 读取图像帧逻辑保持不变...
                imgs = sorted(glob.glob(os.path.join(vis_dir, '*.jpg')) + 
                              glob.glob(os.path.join(vis_dir, '*.png')))
                
                if len(imgs) == 0: continue
                # ... 限制帧数并 append ...
                
                # 对齐帧数...
                if len(imgs) < n_frames:
                    imgs = imgs + [imgs[-1]] * (n_frames - len(imgs))
                else:
                    imgs = imgs[:n_frames]

                self.samples.append((spec_path, imgs, label))
        print(f"🚀 最终有效样本数: {len(self.samples)}")
        from collections import Counter

        labels = [s[2] for s in self.samples]
        print("📊 label 范围:", min(labels), max(labels))
        print("📊 类别数:", len(set(labels)))
        print("📊 分布:", Counter(labels))
        
        self.img_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        spec_path, imgs, label = self.samples[idx]

        try:
            # 🔹 读取 audio_spec
            with open(spec_path, 'rb') as f:
                spec = pickle.load(f)
            
            # 修复 UserWarning: 使用 detach().clone()
            if not isinstance(spec, torch.Tensor):
                spec = torch.from_numpy(spec).float()
            else:
                spec = spec.detach().clone().float()

            # 🔹 读取 visual 帧
            frames = []
            for img_path in imgs:
                img = Image.open(img_path).convert('RGB')
                img = self.img_transform(img)
                frames.append(img)
            
            # 将多帧拼成 (C, T, H, W) 或 (T, C, H, W)
            # 你的模型 models.py 里有 permute，通常期待是 (T, C, H, W)
            visual_tensor = torch.stack(frames, dim=0) 

            return spec, visual_tensor, label

        except Exception as e:
            print(f"❌ 读取样本失败 {spec_path}: {e}")
            # 返回一个全 0 样本防止训练崩溃，或者再次递归（不推荐）
            return torch.zeros(1, 128, 128), torch.zeros(self.n_frames, 3, 224, 224), label