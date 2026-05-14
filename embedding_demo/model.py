"""用户与物品嵌入生成模型。"""
from __future__ import annotations

import torch.nn as nn


class EmbeddingModel(nn.Module):
    """通过 Embedding 层学习用户和物品的稠密向量表示。

    前向传播返回三样东西：
        - dot_product: 用户向量与物品向量的逐元素乘积之和（表示相关性得分）
        - user_vector:  用户嵌入向量
        - item_vector:  物品嵌入向量
    """

    def __init__(self, num_users: int, num_items: int, embed_dim: int):
        super().__init__()
        self.user_embed = nn.Embedding(num_users, embed_dim)
        self.item_embed = nn.Embedding(num_items, embed_dim)

    def forward(self, user_ids, item_ids):
        user_vector = self.user_embed(user_ids)
        item_vector = self.item_embed(item_ids)
        dot_product = (user_vector * item_vector).sum(dim=1)
        return dot_product, user_vector, item_vector
