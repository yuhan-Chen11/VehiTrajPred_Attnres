import torch
import torch.nn as nn
import torch.nn.functional as F


class BlockAttnRes(nn.Module):
    def __init__(self, dim, num_layers, num_blocks=8):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        
        # 关键设计1: Embedding层单独作为第0块
        # 文档："首先Embedding层单独作为一个Block"
        self.block_sizes = [1]  # 第0块: Embedding层
        
        # 关键设计2: 剩余层均匀分配到其他块
        remaining_layers = num_layers - 1
        layers_per_block = remaining_layers // (num_blocks - 1)
        for _ in range(num_blocks - 1):
            self.block_sizes.append(layers_per_block)
        
        # 分配余数
        remainder = remaining_layers % (num_blocks - 1)
        for i in range(remainder):
            self.block_sizes[i + 1] += 1
        
        # 关键设计3: 每个块一个静态查询向量
        self.queries = nn.Parameter(torch.randn(num_blocks, dim) * 0.02)
    
    def forward(self, x_list):
        """
        严格遵循文档描述的实现：
        1. Embedding层(第0个x)单独作为一块
        2. 块内通过求和做压缩
        3. 块间做注意力融合
        4. 输出最终融合结果
        """
        B, T, D = x_list[0].shape
        
        # 1. 块内求和压缩
        block_reps = []
        start = 0
        for block_size in self.block_sizes:
            end = start + block_size
            if end <= len(x_list):
                # 文档："Block内通过求和做压缩"
                block_sum = torch.stack(x_list[start:end], dim=0).sum(dim=0)  # (B, T, D)
                block_reps.append(block_sum)
            start = end
        
        block_representations = torch.stack(block_reps, dim=1)  # (B, num_blocks, T, D)
        
        # 2. 块间注意力
        # 沿时间维度平均得到块键表示
        block_keys = block_representations.mean(dim=2)  # (B, num_blocks, D)
        scores = torch.einsum('bd,bsd->bs', self.queries[0], block_keys)  # 简化计算
        weights = F.softmax(scores, dim=-1)  # (B, num_blocks)
        
        # 3. 块间加权融合
        out = torch.einsum('bs,bstd->btd', weights, block_representations)
        
        return out