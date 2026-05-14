"""模拟用户-物品交互数据集。"""
from __future__ import annotations

import torch
from torch.utils.data import Dataset


class UserItemDataset(Dataset):
    """随机生成用户-物品交互对。

    每个样本是一个 (user_id, item_id) 对，表示一次正向交互。
    """

    def __init__(self, num_users: int, num_items: int, num_samples: int):
        self.num_users = num_users
        self.num_items = num_items
        self.num_samples = num_samples
        self.data = [
            (
                torch.randint(0, num_users, (1,)).item(),
                torch.randint(0, num_items, (1,)).item(),
            )
            for _ in range(num_samples)
        ]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[int, int]:
        return self.data[idx]
