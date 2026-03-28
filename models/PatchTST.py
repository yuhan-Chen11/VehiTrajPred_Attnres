"""
实验3: PatchTST
核心思想: Patch切分后接Transformer编码器进行序列混合
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.attnres import AttnRes


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# ==================== Patch 模块 ====================
class Patching(nn.Module):
    """
    PatchTST的Patch操作
    将长序列切分成多个patch,降低序列长度

    输入: [B, T, C]
    输出: [B, num_patches, patch_len * C]
    """
    def __init__(self, patch_len=5, stride=4):
        super(Patching, self).__init__()
        self.patch_len = patch_len
        self.stride = stride

    def forward(self, x):
        """
        x: [B, T, C]
        """
        B, T, C = x.shape

        # 使用unfold进行patch切分
        # unfold(dimension, size, step)
        x = x.permute(0, 2, 1)  # [B, C, T]

        # 计算需要的padding
        num_patches = (T - self.patch_len) // self.stride + 1
        last_patch_start = (num_patches - 1) * self.stride
        padding_needed = max(0, self.patch_len - (T - last_patch_start))

        if padding_needed > 0:
            x = nn.functional.pad(x, (0, padding_needed), mode='replicate')

        # Unfold操作
        x = x.unfold(dimension=2, size=self.patch_len, step=self.stride)  # [B, C, num_patches, patch_len]

        # 重排维度
        x = x.permute(0, 2, 1, 3)  # [B, num_patches, C, patch_len]
        x = x.reshape(B, -1, C * self.patch_len)  # [B, num_patches, C * patch_len]

        return x


# ==================== PatchTST 创新残差块 ====================
class PatchMixBlock(nn.Module):
    """
    Patch切分 + Transformer编码 + 重建创新块
    输入: [B, T, input_dim]
    输出: [B, T, d_model]
    """
    def __init__(
        self,
        input_dim,
        d_model,
        patch_len=5,
        stride=4,
        n_heads=8,
        num_layers=2,
        d_ff=512,
        dropout=0.1,
    ):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.patching = Patching(patch_len=patch_len, stride=stride)
        self.patch_proj = nn.Linear(patch_len * input_dim, d_model)
        self.patch_reconstruct = nn.Linear(d_model, d_model * patch_len)

        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        self.attnres = AttnRes(d_model)

    def forward(self, x):
        T_in = x.size(1)
        x_patched = self.patching(x)
        x_patched = self.patch_proj(x_patched)
        

        mixed = self.dropout(self.encoder(self.pos_encoder(x_patched)))
        states = [x_patched, mixed]
        res_out = self.attnres(states)
        x_patched = self.norm(res_out)
        
        
        return self._reconstruct_sequence(x_patched, T_in)

    def _reconstruct_sequence(self, patch_features, target_len):
        B, num_patches, _ = patch_features.shape
        decoded = self.patch_reconstruct(patch_features)
        decoded = decoded.view(B, num_patches, self.patch_len, -1)

        total_len = (num_patches - 1) * self.stride + self.patch_len
        device = patch_features.device
        dtype = patch_features.dtype
        recon = torch.zeros(B, total_len, decoded.size(-1), device=device, dtype=dtype)
        counts = torch.zeros(B, total_len, 1, device=device, dtype=dtype)

        for idx in range(num_patches):
            start = idx * self.stride
            end = start + self.patch_len
            recon[:, start:end, :] += decoded[:, idx, :, :]
            counts[:, start:end, :] += 1

        recon = recon / counts.clamp_min(1.0)

        if total_len > target_len:
            recon = recon[:, :target_len, :]
        elif total_len < target_len:
            pad = target_len - total_len
            recon = torch.cat([recon, recon[:, -1:, :].repeat(1, pad, 1)], dim=1)

        return recon


class TemporalResampler(nn.Module):
    """将序列重采样到预测长度, 并进行轻量非线性映射。"""
    def __init__(self, d_model, pred_len, dropout=0.1):
        super().__init__()
        self.pred_len = pred_len
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x):
        if x.size(1) != self.pred_len:
            x = F.interpolate(
                x.permute(0, 2, 1),
                size=self.pred_len,
                mode='linear',
                align_corners=False
            ).permute(0, 2, 1)
        return self.mlp(x)


# ==================== 主模型: PatchTSTAttnRes ====================
class PatchTSTAttnRes(nn.Module):
    """
    流程:
    1. 输入数据 [B, T_in, C_in]
    2. Patch残差创新块输出特征
    3. 序列重采样得到预测特征
    4. 输出投影生成预测结果
    """
    def __init__(self,
                 input_dim=58,
                 d_model=128,
                 n_heads=8,
                 num_layers=2,
                 d_ff=512,
                 dropout=0.1,
                 pred_len=25,
                 patch_len=5,
                 stride=4):
        super(PatchTSTAttnRes, self).__init__()

        self.pred_len = pred_len
        self.d_model = d_model
        self.patch_len = patch_len
        self.stride = stride

        # Patch + Transformer 编码块
        self.patch_block = PatchMixBlock(
            input_dim=input_dim,
            d_model=d_model,
            patch_len=patch_len,
            stride=stride,
            n_heads=n_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            dropout=dropout
        )

        # 输出投影
        self.output_proj = nn.Linear(d_model, 5)

        # 预测序列重采样头
        self.pred_head = TemporalResampler(d_model, pred_len, dropout=dropout)

        # 分类头
        self.lane_cly = nn.Linear(d_model, 3)
        self.longitudinal_cly = nn.Linear(d_model, 3)


    def forward(self, batch):
        """
        batch: dict with keys 'I' and 'S'
        """
        I = batch['I']
        S = batch['S']
        x = torch.cat([I, S], dim=-1)  # [B, T, input_dim]
        T_in = x.size(1)

        # 步骤1: Patch + Transformer 编码块
        memory = self.patch_block(x)  # [B, T_in, d_model]

        # 分类任务
        lane = self.lane_cly(memory[:, 0])
        longitudinal = self.longitudinal_cly(memory[:, 0])

        # 步骤2: 序列重采样预测
        pred_features = self.pred_head(memory)  # [B, pred_len, d_model]
        raw_out = self.output_proj(pred_features)  # [B, pred_len, 5]
        mu = raw_out[..., :2]
        sigma = F.softplus(raw_out[..., 2:4]) + 1e-6
        rho = torch.tanh(raw_out[..., 4:])
        out = torch.cat([mu, sigma, rho], dim=-1)

        return out, lane, longitudinal

# ==================== 配置类 ====================
class Exp3Config:
    """实验3配置"""
    def __init__(self):
        # 数据配置
        self.batch_size = 32
        self.seq_len = 15
        self.pred_len = 25
        self.nbr_cnt = 13

        # 输入维度
        self.state_dim = 6
        self.interaction_dim = self.nbr_cnt * 4
        self.input_dim = self.state_dim + self.interaction_dim

        # 模型配置
        self.d_model = 128
        self.n_heads = 8
        self.num_layers = 2
        self.d_ff = 512
        self.dropout = 0.1

        # PatchTST配置
        self.patch_len = 5
        self.stride = 4


# ==================== 测试代码 ====================
if __name__ == "__main__":
    print("=" * 80)
    print("实验3: PatchTST ")
    print("=" * 80)

    # 配置
    config = Exp3Config()

    # 创建模型
    model = PatchTSTAttnRes(
        input_dim=config.input_dim,
        d_model=config.d_model,
        n_heads=config.n_heads,
        num_layers=config.num_layers,
        d_ff=config.d_ff,
        dropout=config.dropout,
        pred_len=config.pred_len,
        patch_len=config.patch_len,
        stride=config.stride
    )

    # 测试数据
    batch = {
        'S': torch.randn(config.batch_size, config.seq_len, config.state_dim),
        'I': torch.randn(config.batch_size, config.seq_len, config.interaction_dim),
    }

    print(f"\n输入数据:")
    print(f"  S (状态): {batch['S'].shape}")
    print(f"  I (交互): {batch['I'].shape}")

    # 前向传播
    with torch.no_grad():
        out, lane, longitudinal = model(batch)

    print(f"\n输出数据:")
    print(f"  轨迹预测: {out.shape}")
    print(f"  车道分类: {lane.shape}")
    print(f"  纵向分类: {longitudinal.shape}")

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n模型参数:")
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")

    # 计算patch信息
    num_patches = (config.seq_len - config.patch_len) // config.stride + 1
    print(f"\nPatch信息:")
    print(f"  原始序列长度: {config.seq_len}")
    print(f"  Patch长度: {config.patch_len}")
    print(f"  步长: {config.stride}")
    print(f"  Patch数量: {num_patches}")
    print(f"  序列压缩比: {config.seq_len / num_patches:.2f}x")


