"""
实验4: FEDformer
核心思想: 在Transformer结构中使用傅里叶增强注意力进行频域混合
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


# ==================== FEDformer 傅里叶注意力模块 ====================
class FourierAttention(nn.Module):
    """
    FEDformer的傅里叶域注意力机制
    在频域进行注意力计算,降低复杂度到O(N log N)
    """
    def __init__(self, d_model, n_heads=8, modes=32, mode_select='random'):
        super(FourierAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.modes = modes
        self.mode_select = mode_select
        self.d_keys = d_model // n_heads

        self.scale = 1. / (self.d_keys ** 0.5)

    def forward(self, q, k, v, mask=None):
        """
        输入: q, k, v [B, L, H, E] (B=batch, L=length, H=heads, E=d_keys)
        输出: [B, L_q, H, E]
        """
        B, L_q, H, E = q.shape
        _, L_k, _, _ = k.shape
        _, L_v, _, _ = v.shape

        # 转换到频域
        q_ft = torch.fft.rfft(q, dim=1, norm='ortho')
        k_ft = torch.fft.rfft(k, dim=1, norm='ortho')
        v_ft = torch.fft.rfft(v, dim=1, norm='ortho')

        # 选择关键频率模式
        freq_len_q = q_ft.shape[1]
        freq_len_k = k_ft.shape[1]
        freq_len_v = v_ft.shape[1]

        modes_q = min(self.modes, freq_len_q)
        modes_k = min(self.modes, freq_len_k)
        modes_v = min(self.modes, freq_len_v)
        modes_common = min(modes_q, modes_k, modes_v)

        index = list(range(0, modes_common))

        # 频域注意力计算
        q_ft_selected = q_ft[:, index, :, :]  # [B, M, H, E]
        k_ft_selected = k_ft[:, index, :, :]
        v_ft_selected = v_ft[:, index, :, :]

        # 跨频率相关性：沿特征维累加并在频率维上归一化
        attn_scores = torch.einsum('bmhe,bmhe->bmh', q_ft_selected, k_ft_selected.conj()).real
        attn_weights = torch.softmax(attn_scores * self.scale, dim=1)  # normalize over frequency dimension

        # 逐频加权
        out_ft = attn_weights.unsqueeze(-1) * v_ft_selected

        # 重构完整频谱
        out_ft_full = torch.zeros_like(q_ft)
        out_ft_full[:, index, :, :] = out_ft

        # 逆傅里叶变换回时域
        out = torch.fft.irfft(out_ft_full, n=L_q, dim=1, norm='ortho')

        return out


# ==================== 傅里叶混合创新块 ====================
class FourierMixBlock(nn.Module):
    """使用傅里叶注意力的Transformer编码层"""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, modes=32):
        super().__init__()

        self.fourier_attn = FourierAttention(d_model, n_heads, modes)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

        self.attnres = AttnRes(d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.n_heads = n_heads
        self.d_model = d_model

    def forward(self, x):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, -1)
        k = self.k_proj(x).view(B, T, self.n_heads, -1)
        v = self.v_proj(x).view(B, T, self.n_heads, -1)

        attn_out = self.fourier_attn(q, k, v)
        attn_out = attn_out.reshape(B, T, self.d_model)
        attn_out = self.out_proj(attn_out)
        x = self.norm1(self.attnres([x, self.dropout(attn_out)]))

        x = self.norm2(self.attnres([x, self.dropout(attn_out)]))
        return x


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


# ==================== 主模型: FEDformerAttnres模型 ====================
class FEDformerAttnres(nn.Module):
    """
    流程:
    1. 输入数据 [B, T_in, C_in]
    2. 傅里叶注意力Transformer编码层
    3. 序列重采样得到预测特征
    4. 输出投影生成预测结果
    """
    def __init__(self,
                 input_dim=58,
                 d_model=128,
                 n_heads=8,
                 num_blocks=4,
                 d_ff=512,
                 dropout=0.1,
                 pred_len=25,
                 modes=32):
        super(FEDformerAttnres, self).__init__()

        self.pred_len = pred_len
        self.d_model = d_model

        # 输入/输出投影
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.output_proj = nn.Linear(d_model, 5)

        # 傅里叶注意力Transformer编码层
        self.fourier_blocks = nn.ModuleList([
            FourierMixBlock(d_model, n_heads, d_ff, dropout, modes)
            for _ in range(num_blocks)
        ])

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
        x = self.input_proj(x)  # [B, T, d_model]
        x = self.pos_encoder(x)

        # 步骤2: 傅里叶注意力Transformer编码层
        for block in self.fourier_blocks:
            x = block(x)

        memory = x

        # 分类任务
        lane = self.lane_cly(memory[:, 0])
        longitudinal = self.longitudinal_cly(memory[:, 0])

        # 步骤3: 序列重采样预测
        pred_features = self.pred_head(memory)  # [B, pred_len, d_model]

        # 步骤4: 输出投影并参数化 (mu, sigma, rho)
        raw_out = self.output_proj(pred_features)  # [B, pred_len, 5]
        mu = raw_out[..., :2]
        sigma = F.softplus(raw_out[..., 2:4]) + 1e-6
        rho = torch.tanh(raw_out[..., 4:])
        out = torch.cat([mu, sigma, rho], dim=-1)

        return out, lane, longitudinal


# ==================== 配置类 ====================
class Exp4Config:
    """实验4配置"""
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
        self.num_blocks = 4
        self.d_ff = 512
        self.dropout = 0.1

        # FEDformer配置
        self.modes = 32


# ==================== 测试代码 ====================
if __name__ == "__main__":
    print("=" * 80)
    print("实验4: FEDformer")
    print("=" * 80)

    # 配置
    config = Exp4Config()

    # 创建模型
    model = FEDformerAttnres(
        input_dim=config.input_dim,
        d_model=config.d_model,
        n_heads=config.n_heads,
        num_blocks=config.num_blocks,
        d_ff=config.d_ff,
        dropout=config.dropout,
        pred_len=config.pred_len,
        modes=config.modes
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


