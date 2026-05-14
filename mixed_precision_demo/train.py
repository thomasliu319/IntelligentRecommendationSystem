"""混合精度训练与数据并行 —— 核心训练流程。"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset

from model import SimpleModel


def create_dataloader(
    num_samples: int = 10000,
    input_dim: int = 10,
    batch_size: int = 256,
) -> DataLoader:
    """创建模拟二分类数据集。

    Args:
        num_samples: 样本总数。
        input_dim: 每个样本的特征维度。
        batch_size: 批大小。

    Returns:
        一个 shuffle 的 DataLoader。
    """
    X = torch.rand(num_samples, input_dim)
    y = torch.randint(0, 2, (num_samples,))
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def train(
    epochs: int = 5,
    lr: float = 0.001,
) -> SimpleModel:
    """使用混合精度（AMP）与数据并行（DataParallel）训练模型。

    训练完成后返回训练好的 model（已移除 DataParallel wrapper，处于 eval 模式）。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"运行设备：{device}")

    dataloader = create_dataloader()
    model = SimpleModel(input_dim=10, output_dim=2).to(device)

    criterion: nn.Module = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scaler = GradScaler()

    # 数据并行：将模型复制到所有可用 GPU 上
    if torch.cuda.device_count() > 1:
        print(f"检测到 {torch.cuda.device_count()} 块 GPU，启用 DataParallel")
    model = nn.DataParallel(model)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f"第 {epoch + 1} 轮训练，平均损失：{avg_loss:.4f}")

    # 评估前剥离 DataParallel wrapper，恢复为原始模型
    if isinstance(model, nn.DataParallel):
        model = model.module
    model.eval()
    return model
