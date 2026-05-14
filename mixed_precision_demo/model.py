"""二分类模型定义。"""
from __future__ import annotations

import torch.nn as nn


class SimpleModel(nn.Module):
    """简单的全连接二分类模型。

    结构: Linear(10→64) → ReLU → Linear(64→2)
    """

    def __init__(self, input_dim: int = 10, output_dim: int = 2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x):
        return self.fc(x)
