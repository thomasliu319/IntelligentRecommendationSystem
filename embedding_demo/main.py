"""【例1-6】嵌入生成与用户画像建模 —— 运行入口。

演示:
  1. 使用 PyTorch Embedding 层学习用户/物品的稠密向量
  2. 通过点积得分训练模型，使正向交互的得分趋近 1.0
  3. 导出训练后的嵌入向量作为用户/物品画像
"""
from __future__ import annotations

import torch

from train import train


def print_embedding(label: str, tensor: torch.Tensor) -> None:
    """友好打印嵌入向量。"""
    print(f"{label} 的嵌入向量: {tensor.detach().numpy()}")


def main() -> None:
    model = train()

    test_user = torch.tensor([0])
    test_item = torch.tensor([5])

    _, user_vector, item_vector = model(test_user, test_item)

    print("\n测试结果：")
    print_embedding(f"用户ID: {test_user.item()}", user_vector)
    print_embedding(f"物品ID: {test_item.item()}", item_vector)


if __name__ == "__main__":
    main()
