import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from .backbone import resnet18  # 保留你原来的其他类时可留着
from .mbt import MBTClassifier

class ConcatFusion(nn.Module):
    def __init__(self, input_dim=1024+512, output_dim=100):
        super(ConcatFusion, self).__init__()
        self.fc_out = nn.Linear(input_dim, output_dim)

    def forward(self, out):
        # output = torch.cat((x, y), dim=1)
        output = self.fc_out(out)
        return output


class CGClassifier(nn.Module):
    def __init__(self, args):
        super(CGClassifier, self).__init__()
        # 换成强大的 ResNet18
        from .backbone import resnet18
        
        self.audio_net = resnet18(modality='visual', pretrained=True)
        self.visual_net = resnet18(modality='visual', pretrained=True)
        
        # ResNet18 输出通常是 512 维
        self.head_audio = nn.Linear(512, args.n_classes)
        self.head_visual = nn.Linear(512, args.n_classes)
        self.fusion = nn.Linear(1024, args.n_classes) # 512 + 512

    def forward(self, audio, visual):
        # 如果输入是 28x28，ResNet 需要插值到 224 (或者 64 以上)
        if visual.shape[-1] < 64:
            visual = F.interpolate(visual, size=(64, 64), mode='bilinear')
            audio = F.interpolate(audio, size=(64, 64), mode='bilinear')

        feat_a = self.audio_net(audio) 
        feat_a = F.adaptive_avg_pool2d(feat_a, 1).flatten(1)
        
        feat_v = self.visual_net(visual)
        feat_v = F.adaptive_avg_pool2d(feat_v, 1).flatten(1)

        return self.fusion(torch.cat([feat_a, feat_v], 1)), self.head_audio(feat_a), self.head_visual(feat_v)

class PointNetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1, self.bn2, self.bn3 = nn.BatchNorm1d(64), nn.BatchNorm1d(128), nn.BatchNorm1d(1024)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return torch.max(x, 2)[0] # Global Max Pooling

class ModelNetModel(nn.Module):
    def __init__(self, n_classes=40):
        super().__init__()
        self.encoder = PointNetEncoder()
        self.cls_head = nn.Linear(1024, n_classes)
        # 为了跑你的 MMPareto，我们加一个辅助任务：旋转角度预测
        self.rot_head = nn.Linear(1024, 4) 

    def forward(self, x):
        feat = self.encoder(x)
        return self.cls_head(feat), self.rot_head(feat)
        
class NYUMultiTaskModel(nn.Module):
    def __init__(self, n_classes=13): # 常用 13 类分割
        super().__init__()
        self.encoder = resnet18(modality='visual', pretrained=True)
        # 移除 resnet 末尾的 avgpool 和 fc，获取特征图
        
        # 简单的上采样解码器
        self.seg_head = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.Upsample(scale_factor=32, mode='bilinear'),
            nn.Conv2d(256, n_classes, 1)
        )
        self.depth_head = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.Upsample(scale_factor=32, mode='bilinear'),
            nn.Conv2d(256, 1, 1)
        )

    def forward(self, x):
        feat = self.encoder(x) # 假设输出是 [B, 512, H/32, W/32]
        seg_logits = self.seg_head(feat)
        depth_pred = self.depth_head(feat)
        return seg_logits, depth_pred
        
class RGBClassifier(nn.Module):
    def __init__(self, args):
        super(RGBClassifier, self).__init__()

        n_classes = 101

        self.visual_net = resnet18(modality='visual')
        self.visual_net.load_state_dict(torch.load('/home/yake_wei/models/resnet18.pth'), strict=False)
        self.fc = nn.Linear(512, n_classes)

    def forward(self, visual):
        B = visual.size()[0]
        v = self.visual_net(visual)

        (_, C, H, W) = v.size()
        v = v.view(B, -1, C, H, W)
        v = v.permute(0, 2, 1, 3, 4)

        v = F.adaptive_avg_pool3d(v, 1)

        v = torch.flatten(v, 1)

        out = self.fc(v)

        return out

class FlowClassifier(nn.Module):
    def __init__(self, args):
        super(FlowClassifier, self).__init__()

        n_classes = 101

        self.flow_net = resnet18(modality='flow')
        state = torch.load('/home/yake_wei/models/resnet18.pth')
        del state['conv1.weight']
        self.flow_net.load_state_dict(state, strict=False)
        self.fc = nn.Linear(512, n_classes)

    def forward(self, flow):
        B = flow.size()[0]
        v = self.flow_net(flow)

        (_, C, H, W) = v.size()
        v = v.view(B, -1, C, H, W)
        v = v.permute(0, 2, 1, 3, 4)

        v = F.adaptive_avg_pool3d(v, 1)

        v = torch.flatten(v, 1)

        out = self.fc(v)

        return out



class AVClassifier(nn.Module):
    def __init__(self, args):
        super(AVClassifier, self).__init__()

        self.n_classes = args.n_classes
        self.dataset = args.dataset

        # 开启预训练提升特征提取能力
        self.audio_net = resnet18(modality='audio', pretrained=True)
        self.visual_net = resnet18(modality='visual', pretrained=True)

        # 预测头
        self.head = nn.Linear(1024, self.n_classes)
        self.head_audio = nn.Linear(512, self.n_classes)
        self.head_visual = nn.Linear(512, self.n_classes)

    def forward(self, audio, visual):
            # 针对 CGMNIST 小尺寸输入的优化：如果输入分辨率低于 64，则插值放大
        if visual.shape[-1] < 64:
            visual = F.interpolate(visual, size=(224, 224), mode='bilinear', align_corners=False)
        if audio.shape[-1] < 64:
            # 假设 CGMNIST 的音频模态也是 2D 频谱图
            audio = F.interpolate(audio, size=(224, 224), mode='bilinear', align_corners=False)
        # 1. 音频分支
        a = self.audio_net(audio) 
        a = F.adaptive_avg_pool2d(a, 1)
        a = torch.flatten(a, 1) # [B, 512]
        B = a.size(0)

        # 2. 视频分支：自适应处理 4D(图像) 或 5D(视频)
        if len(visual.shape) == 5:
            # 输入为 [B, T, C, H, W] (针对 KS)
            B, T, C, H, W = visual.shape
            visual = visual.view(B * T, C, H, W) # 压平时间维给 2D ResNet
            
            v = self.visual_net(visual) # [B*T, 512, 7, 7]
            
            # 维度还原并进行时间维池化
            (_, C_feat, H_feat, W_feat) = v.size()
            v = v.view(B, T, C_feat, H_feat, W_feat) # [B, T, 512, 7, 7]
            v = v.permute(0, 2, 1, 3, 4)             # [B, 512, T, 7, 7]
            v = F.adaptive_avg_pool3d(v, 1)          # 在 T, H, W 上做全局平均池化
        else:
            # 输入为 [B, C, H, W] (针对 CREMA-D)
            v = self.visual_net(visual)
            v = F.adaptive_avg_pool2d(v, 1)

        v = torch.flatten(v, 1) # [B, 512]

        # 3. 生成各分支预测
        out_audio = self.head_audio(a)
        out_visual = self.head_visual(v)
        
        # 特征融合
        feat_combined = torch.cat((a, v), dim=1) 
        out_joint = self.head(feat_combined)

        return out_joint, out_audio, out_visual
class AVClassifier_OGM(nn.Module):
    def __init__(self, args):
        super(AVClassifier_OGM, self).__init__()

        if args.dataset == 'VGGSound':
            n_classes = 309
        elif args.dataset == 'KineticSound':
            n_classes = 31
        elif args.dataset == 'CREMAD':
            n_classes = 6
        elif args.dataset == 'AVE':
            n_classes = 28
        else:
            raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))


        self.dataset = args.dataset

        self.audio_net = resnet18(modality='audio')
        self.visual_net = resnet18(modality='visual')

        self.head = nn.Linear(1024, n_classes)
        self.head_audio = nn.Linear(512, n_classes)
        self.head_visual = nn.Linear(512, n_classes)



    def forward(self, audio, visual):
        if self.dataset != 'CREMAD':
            visual = visual.permute(0, 2, 1, 3, 4).contiguous()
        a = self.audio_net(audio)
        v = self.visual_net(visual)

        (_, C, H, W) = v.size()
        B = a.size()[0]
        v = v.view(B, -1, C, H, W)
        v = v.permute(0, 2, 1, 3, 4)

        a = F.adaptive_avg_pool2d(a, 1)
        v = F.adaptive_avg_pool3d(v, 1)

        a = torch.flatten(a, 1)
        v = torch.flatten(v, 1)


        out = torch.cat((a,v),1)
        out = self.head(out)


        return a,v,out
