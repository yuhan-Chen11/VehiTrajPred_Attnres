import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # RMSNorm 公式
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class AttnRes(nn.Module):
    """
    Attention Residuals
    用层间注意力替代传统残差 x + F(x)
    输入: x_list: list[(B, T, D)]
    输出: 加权融合 (B, T, D)
    """

    def __init__(self, dim):
        super().__init__()
        self.query = nn.Parameter(torch.randn(dim))
        self.norm = RMSNorm(dim)  

    def forward(self, x_list):
        """
        x_list: 每一层输出 [x0, x1, ..., xt]，shape=(B, T, D)
        """
        # 对每个层特征做 Norm（K = Norm(xs)）
        k_list = [self.norm(x) for x in x_list]

        # 计算每个层的注意力分数 (B, T)
        scores = []
        for k in k_list:
            # (B, T, D) @ (D,) -> (B, T)
            score = torch.einsum('btd,d->bt', k, self.query)
            scores.append(score)

        # (N, B, T)
        scores = torch.stack(scores, dim=0)
        # 对层维度 softmax，得到每层权重 (N, B, T)
        weights = F.softmax(scores, dim=0)

        # 加权融合：sum( w_{s} * x_s )
        out = 0
        for w, x in zip(weights, x_list):
            # w: (B,T), x: (B,T,D) → 广播相乘
            out = out + w.unsqueeze(-1) * x

        return out