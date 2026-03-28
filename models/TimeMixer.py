"""
实验2: TimeMixer 
核心思想: 使用TimeMixer的多尺度混合替换Transformer解码结构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.attnres import AttnRes


# ==================== DLinear分解模块 (TimeMixer需要) ====================
class MovingAverage(nn.Module):
    """移动平均提取趋势分量"""
    def __init__(self, kernel_size, stride=1):
        super(MovingAverage, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = x.permute(0, 2, 1)
        x = self.avg(x)
        x = x.permute(0, 2, 1)
        return x


class SeriesDecomp(nn.Module):
    """季节-趋势分解"""
    def __init__(self, kernel_size=25):
        super(SeriesDecomp, self).__init__()
        self.moving_avg = MovingAverage(kernel_size, stride=1)

    def forward(self, x):
        trend = self.moving_avg(x)
        seasonal = x - trend
        return seasonal, trend


# ==================== TimeMixer 多尺度混合模块 ====================
class MultiScaleSeasonMixing(nn.Module):
    """
    TimeMixer的Bottom-up季节性混合：逐层降低时间分辨率，捕获低频信息后再映射回原尺度。
    """
    def __init__(self, pred_len, d_model, num_scales=2):
        super(MultiScaleSeasonMixing, self).__init__()

        self.num_scales = max(1, num_scales)
        self.downsample_layers = nn.ModuleList([
            nn.AvgPool1d(kernel_size=2, stride=2, ceil_mode=True)
            for _ in range(self.num_scales - 1)
        ])
        self.mixing_layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
            for _ in range(self.num_scales)
        ])

    def forward(self, seasonal):
        """
        seasonal: [B, T, d_model]
        """
        base_len = seasonal.size(1)
        current = seasonal
        mixed = self.mixing_layers[0](current)

        for scale in range(1, self.num_scales):
            pooled = self.downsample_layers[scale - 1](current.permute(0, 2, 1))
            pooled = pooled.permute(0, 2, 1)
            current = pooled
            low_freq = self.mixing_layers[scale](current)
            upsampled = F.interpolate(
                low_freq.permute(0, 2, 1),
                size=base_len,
                mode='linear',
                align_corners=False
            ).permute(0, 2, 1)
            mixed = mixed + upsampled

        return mixed / self.num_scales


class MultiScaleTrendMixing(nn.Module):
    """
    TimeMixer的Top-down趋势混合：从最粗尺度开始，逐层上采样并与高频细节融合。
    """
    def __init__(self, pred_len, d_model, num_scales=2):
        super(MultiScaleTrendMixing, self).__init__()

        self.num_scales = max(1, num_scales)
        self.downsample_layers = nn.ModuleList([
            nn.AvgPool1d(kernel_size=2, stride=2, ceil_mode=True)
            for _ in range(self.num_scales - 1)
        ])
        self.mixing_layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
            for _ in range(self.num_scales)
        ])

    def forward(self, trend):
        """
        trend: [B, T, d_model]
        """
        # 构建多尺度金字塔
        pyramid = [trend]
        current = trend
        for ds in self.downsample_layers:
            current = ds(current.permute(0, 2, 1)).permute(0, 2, 1)
            pyramid.append(current)

        # 自顶向下融合：先处理最粗尺度，再逐步上采样
        current_repr = self.mixing_layers[-1](pyramid[-1])
        for scale in range(self.num_scales - 2, -1, -1):
            target_len = pyramid[scale].size(1)
            upsampled = F.interpolate(
                current_repr.permute(0, 2, 1),
                size=target_len,
                mode='linear',
                align_corners=False
            ).permute(0, 2, 1)
            refined = pyramid[scale] + upsampled
            current_repr = self.mixing_layers[scale](refined)

        return current_repr


# ==================== TimeMixer 创新残差块 ====================
class TimeMixerBlock(nn.Module):
    """
    TimeMixer的多尺度混合残差块。
    输入: [B, T, d_model]
    输出: [B, T, d_model]
    """
    def __init__(self, pred_len, d_model, kernel_size=25, num_scales=2, dropout=0.1):
        super().__init__()
        self.decomp = SeriesDecomp(kernel_size=kernel_size)
        self.season_mixing = MultiScaleSeasonMixing(pred_len, d_model, num_scales)
        self.trend_mixing = MultiScaleTrendMixing(pred_len, d_model, num_scales)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        seasonal, trend = self.decomp(x)
        mixed = self.season_mixing(seasonal) + self.trend_mixing(trend)
        return self.norm(x + self.dropout(mixed))


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


# ==================== 主模型: TimeMixerAttnres模型 ====================
class TimeMixerAttnres(nn.Module):
    """
    流程:
    1. 输入数据 [B, T_in, C_in]
    2. 序列重采样获得预测特征
    3. TimeMixer多尺度残差混合
    4. 输出投影生成预测结果
    """
    def __init__(self,
                 input_dim=58,
                 d_model=128,
                 dropout=0.1,
                 pred_len=25,
                 kernel_size=25,
                 num_scales=2):
        super(TimeMixerAttnres, self).__init__()

        self.pred_len = pred_len
        self.d_model = d_model

        # 输入/输出投影
        self.input_proj = nn.Linear(input_dim, d_model)
        self.output_proj = nn.Linear(d_model, 5)


        # TimeMixer创新块
        self.time_mixer = TimeMixerBlock(
            pred_len=pred_len,
            d_model=d_model,
            kernel_size=kernel_size,
            num_scales=num_scales,
            dropout=dropout
        )

        #attnres
        self.attnres = AttnRes(d_model)
        self.norm = nn.LayerNorm(d_model)


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
        # 步骤1: 输入投影
        memory = self.input_proj(x)           # [B, T, d_model]

        # 分类任务
        lane = self.lane_cly(memory[:, 0])
        longitudinal = self.longitudinal_cly(memory[:, 0])

        # 步骤2: TimeMixer（在原始时间尺度上）
        mixed_memory = self.time_mixer(memory)

        # 步骤3: 两路预测
        pred_raw = self.pred_head(memory)
        pred_mixed = self.pred_head(mixed_memory)

        # 步骤4：AttnRes融合
        states = [pred_raw, pred_mixed]
        out = self.attnres(states)
        out = self.norm(out + pred_raw)


        # 步骤5: 输出投影并参数化 (mu, sigma, rho)
        raw_out = self.output_proj(out)             # [B, pred_len, 5]
        mu = raw_out[..., :2]
        sigma = F.softplus(raw_out[..., 2:4]) + 1e-6
        rho = torch.tanh(raw_out[..., 4:])
        out = torch.cat([mu, sigma, rho], dim=-1)

        return out, lane, longitudinal


# ==================== 配置类 ====================
class Exp2Config:
    """实验2配置"""
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
        self.dropout = 0.1

        # TimeMixer配置
        self.kernel_size = 25
        self.num_scales = 2


# ==================== 测试代码 ====================
if __name__ == "__main__":
    print("=" * 80)
    print("实验2: TimeMixer")
    print("=" * 80)

    # 配置
    config = Exp2Config()

    # 创建模型
    model = TimeMixerAttnres(
        input_dim=config.input_dim,
        d_model=config.d_model,
        dropout=config.dropout,
        pred_len=config.pred_len,
        kernel_size=config.kernel_size,
        num_scales=config.num_scales
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

    print("\n" + "=" * 80)
    print("实验设计说明:")
    print("=" * 80)
    print("""
核心创新:
- TimeMixer的多尺度混合作为后处理
- Bottom-up处理季节性特征 (高频->低频)
- Top-down处理趋势特征 (低频->高频)

优势:
- 通过多尺度混合细化预测特征
- 捕获不同时间尺度的模式

对比基线:
- 简化基线: 直接线性预测
- 本模型: 多尺度混合,更好地建模多时间尺度特征
    """)
    print("=" * 80)
