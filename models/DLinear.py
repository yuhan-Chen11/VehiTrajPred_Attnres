"""
实验1: DLinear 
核心思想: 使用DLinear季节-趋势分解进行残差融合建模
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.attnres import AttnRes



# ==================== DLinear 模块 ====================
class MovingAverage(nn.Module):
    """移动平均提取趋势分量"""
    def __init__(self, kernel_size, stride=1):
        super(MovingAverage, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x, kernel_size=None):
        # 根据当前序列长度自适应 kernel，避免 kernel 大于序列长度
        k = kernel_size or self.kernel_size
        k = int(max(1, k))
        k = min(k, x.size(1))

        pad_left = (k - 1) // 2
        pad_right = (k - 1) - pad_left
        first = x[:, 0:1, :]
        last = x[:, -1:, :]
        parts = []
        if pad_left > 0:
            parts.append(first.repeat(1, pad_left, 1))
        parts.append(x)
        if pad_right > 0:
            parts.append(last.repeat(1, pad_right, 1))
        x_padded = torch.cat(parts, dim=1) if len(parts) > 1 else x
        x_padded = x_padded.permute(0, 2, 1)

        if k == self.kernel_size:
            x_avg = self.avg(x_padded)
        else:
            x_avg = F.avg_pool1d(x_padded, kernel_size=k, stride=self.stride, padding=0)

        x_avg = x_avg.permute(0, 2, 1)
        return x_avg


class SeriesDecomp(nn.Module):
    """
    DLinear的季节-趋势分解模块
    输入: [B, T, C]
    输出: seasonal [B, T, C], trend [B, T, C]
    """
    def __init__(self, kernel_size=25):
        super(SeriesDecomp, self).__init__()
        self.kernel_size = kernel_size
        self.moving_avg = MovingAverage(kernel_size, stride=1)

    def forward(self, x):
        effective_kernel = min(self.kernel_size, x.size(1))
        trend = self.moving_avg(x, kernel_size=effective_kernel)
        seasonal = x - trend
        return seasonal, trend


# ==================== DLinear 残差创新块 ====================
class DLinearBlock(nn.Module):
    """
    将 DLinear 分解与残差融合封装为一个创新块。
    输入: [B, T, input_dim]
    输出: [B, T, d_model]
    """
    def __init__(self, input_dim, d_model, kernel_size=25, dropout=0.1):
        super().__init__()
        self.decomp = SeriesDecomp(kernel_size=kernel_size)
        self.seasonal_proj = nn.Linear(input_dim, d_model)
        self.trend_proj = nn.Linear(input_dim, d_model)
        self.residual_proj = nn.Linear(input_dim, d_model)

        self.fusion = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

        self.attnres = AttnRes(d_model)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        seasonal, trend = self.decomp(x)
        seasonal = self.seasonal_proj(seasonal)
        trend = self.trend_proj(trend)
        fused = self.fusion(torch.cat([seasonal, trend], dim=-1))
        residual = self.residual_proj(x)
        out = self.attnres([residual, self.dropout(fused)])
        return self.norm(out)


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


# ==================== 主模型: DLinearAttnres模型 ====================
class DLinearAttnres(nn.Module):
    """
    流程:
    1. 输入数据 [B, T_in, C_in]
    2. DLinear残差创新块
    3. 序列重采样得到预测特征
    4. 输出投影生成预测结果
    """
    def __init__(self,
                 input_dim=58,          # 输入特征维度
                 d_model=128,           # 模型隐藏维度
                 dropout=0.1,
                 pred_len=25,           # 预测长度
                 kernel_size=25):        # DLinear的移动平均kernel
        super(DLinearAttnres, self).__init__()

        self.pred_len = pred_len
        self.d_model = d_model

        # DLinear创新块
        self.dlinear_block = DLinearBlock(
            input_dim=input_dim,
            d_model=d_model,
            kernel_size=kernel_size,
            dropout=dropout
        )

        # 预测序列重采样头
        self.pred_head = TemporalResampler(d_model, pred_len, dropout=dropout)

        # 输出投影
        self.output_proj = nn.Linear(d_model, 5)

        # 分类头
        self.lane_cly = nn.Linear(d_model, 3)
        self.longitudinal_cly = nn.Linear(d_model, 3)

    def forward(self, batch):
        """
        batch: dict with keys 'I' and 'S'
        - I: [B, T, interaction_dim]
        - S: [B, T, state_dim]

        return:
        - out: [B, pred_len, 5]
        - lane: [B, 3]
        - longitudinal: [B, 3]
        """
        I = batch['I']
        S = batch['S']
        x = torch.cat([I, S], dim=-1)  # [B, T, input_dim]
        # 步骤1: DLinear残差创新块
        memory = self.dlinear_block(x)  # [B, T, d_model]

        # 分类任务
        lane = self.lane_cly(memory[:, 0])           # [B, 3]
        longitudinal = self.longitudinal_cly(memory[:, 0])  # [B, 3]

        # 步骤2: 预测序列重采样
        pred_features = self.pred_head(memory)       # [B, pred_len, d_model]
        raw_out = self.output_proj(pred_features)    # [B, pred_len, 5]
        mu = raw_out[..., :2]
        sigma = F.softplus(raw_out[..., 2:4]) + 1e-6
        rho = torch.tanh(raw_out[..., 4:])
        out = torch.cat([mu, sigma, rho], dim=-1)

        return out, lane, longitudinal


# ==================== 配置类 ====================
class Exp1Config:
    """实验1配置"""
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

        # DLinear配置
        self.kernel_size = 25


# ==================== 测试代码 ====================
if __name__ == "__main__":
    print("=" * 80)
    print("实验1: DLinear")
    print("=" * 80)

    # 配置
    config = Exp1Config()

    # 创建模型
    model = DLinearAttnres(
        input_dim=config.input_dim,
        d_model=config.d_model,
        dropout=config.dropout,
        pred_len=config.pred_len,
        kernel_size=config.kernel_size
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
- DLinear的季节-趋势分解作为预处理
- 季节/趋势残差融合块
- 轻量重采样生成预测特征

优势:
- 显式建模周期性和长期趋势
- 残差融合提高稳定性

对比基线:
- 直接线性预测: 直接在原始序列上回归
- 本模型: 先分解再融合,更好地理解时间序列结构
    """)
    print("=" * 80)
