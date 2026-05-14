"""【例1-5】混合精度训练与数据并行 —— 运行入口。

演示:
  1. 使用 PyTorch AMP (autocast + GradScaler) 进行混合精度训练
  2. 使用 nn.DataParallel 实现数据并行
  3. 训练完成后进行推理评估
"""
from __future__ import annotations

import torch
from torch.cuda.amp import autocast

from train import train


def evaluate(model: torch.nn.Module, device: torch.device) -> None:
    """使用混合精度进行推理并打印结果。"""
    with torch.no_grad():
        sample_input = torch.rand(5, 10).to(device)
        with autocast():
            predictions = model(sample_input)
        print("示例输入：", sample_input.cpu().numpy())
        print("预测结果：", torch.argmax(predictions, dim=1).cpu().numpy())


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = train(epochs=5, lr=0.001)
    evaluate(model, device)


if __name__ == "__main__":
    main()
