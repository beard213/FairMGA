import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

def fix_input(x):
    # [B, T, C, H, W] - 5D 视频数据
    if x.dim() == 5:
        return x
    # [H, W]
    if x.dim() == 2:
        x = x.unsqueeze(0).unsqueeze(0)
    # [C, H, W] 或 [B, H, W]
    elif x.dim() == 3:
        if x.shape[0] in [1, 3]:
            x = x.unsqueeze(0)   # → [1,C,H,W]
        else:
            x = x.unsqueeze(1)   # → [B,1,H,W]
    return x

class MBTClassifier(nn.Module):
    def __init__(self, num_classes=6, pretrained=True):
        super().__init__()
        # 使用预训练的 ViT
        self.audio_backbone = timm.create_model('vit_base_patch16_224', pretrained=pretrained, num_classes=0)
        self.visual_backbone = timm.create_model('vit_base_patch16_224', pretrained=pretrained, num_classes=0)

        dim = self.audio_backbone.embed_dim
        self.head = nn.Linear(dim * 2, num_classes)
        self.head_a = nn.Linear(dim, num_classes)
        self.head_v = nn.Linear(dim, num_classes)

    def forward(self, audio, visual):
        # 1. 兜底处理
        audio = fix_input(audio)
        visual = fix_input(visual)

        # ===== ✅ 处理 5D 视频输入 [B, T, C, H, W] =====
        if visual.dim() == 5:
            B, T, C, H, W = visual.shape
            # 将 Batch 和 Time 维度合并，一起送入 backbone
            visual = visual.view(B * T, C, H, W) 
        else:
            B = visual.shape[0]
            T = 1

        # 2. 统一尺寸为 224x224
        if audio.shape[-1] != 224:
            audio = F.interpolate(audio, size=(224, 224), mode='bilinear', align_corners=False)
        if visual.shape[-1] != 224:
            visual = F.interpolate(visual, size=(224, 224), mode='bilinear', align_corners=False)

        # 3. 统一为 3 通道 (ViT 要求)
        if audio.shape[1] == 1:
            audio = audio.repeat(1, 3, 1, 1)
        if visual.shape[1] == 1:
            visual = visual.repeat(1, 3, 1, 1)

        # 4. 特征提取
        a_feat = self.audio_backbone(audio) # [B, 768]
        
        v_feat_raw = self.visual_backbone(visual) # [B*T, 768]
        # ===== ✅ 时间维度聚合 (Temporal Pooling) =====
        if T > 1:
            v_feat = v_feat_raw.view(B, T, -1).mean(1) # 将 6 帧的特征取平均 -> [B, 768]
        else:
            v_feat = v_feat_raw

        # 5. 融合与输出
        fused = torch.cat([a_feat, v_feat], dim=1) # [B, 1536]

        out = self.head(fused)
        out_a = self.head_a(a_feat)
        out_v = self.head_v(v_feat)

        return out, out_a, out_v