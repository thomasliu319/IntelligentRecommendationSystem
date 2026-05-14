"""训练 EmbeddingModel 并返回用户/物品嵌入。"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import UserItemDataset
from model import EmbeddingModel

# ---------- 超参数 ----------
NUM_USERS = 100
NUM_ITEMS = 200
EMBED_DIM = 8
NUM_SAMPLES = 1000
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 0.01


def train() -> EmbeddingModel:
    """使用 MSE 损失训练嵌入模型，使正样本的得分逼近 1.0。"""
    dataset = UserItemDataset(NUM_USERS, NUM_ITEMS, NUM_SAMPLES)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = EmbeddingModel(NUM_USERS, NUM_ITEMS, EMBED_DIM)
    criterion: nn.Module = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("开始训练模型：")

    for epoch in range(EPOCHS):
        total_loss = 0.0

        for user_ids, item_ids in dataloader:
            # 所有交互视为正向交互，目标得分为 1
            predicted_scores, _, _ = model(user_ids, item_ids)
            loss = criterion(predicted_scores, torch.ones(len(user_ids)))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"第 {epoch + 1} 轮训练，平均损失：{avg_loss:.4f}")

    model.eval()
    return model
