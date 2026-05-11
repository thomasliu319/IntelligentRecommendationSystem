"""
Transformer 模型测试主程序
@author thomasliu319
"""
import torch
from model import Encoder


def main():
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # 模型超参数
    embed_size = 256
    num_layers = 2
    heads = 8
    dropout = 0.1
    forward_expansion = 4
    src_vocab_size = 10000
    max_length = 100

    print("\n模型配置:")
    print(f"  嵌入维度: {embed_size}")
    print(f"  编码器层数: {num_layers}")
    print(f"  注意力头数: {heads}")
    print(f"  Dropout率: {dropout}")
    print(f"  前馈网络扩展倍数: {forward_expansion}")
    print(f"  词汇表大小: {src_vocab_size}")
    print(f"  最大序列长度: {max_length}")

    # 创建编码器
    encoder = Encoder(
        src_vocab_size,
        embed_size,
        num_layers,
        heads,
        device,
        forward_expansion,
        dropout,
        max_length
    ).to(device)

    # 计算模型参数量
    total_params = sum(p.numel() for p in encoder.parameters())
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"\n模型参数:")
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数量: {trainable_params:,}")

    # 创建示例输入
    batch_size = 2
    seq_length = 10
    sample_input = torch.randint(0, src_vocab_size, (batch_size, seq_length)).to(device)
    mask = None

    print(f"\n输入数据:")
    print(f"  批次大小: {batch_size}")
    print(f"  序列长度: {seq_length}")
    print(f"  输入形状: {sample_input.shape}")
    print(f"  输入样例:\n{sample_input}")

    # 前向传播
    print("\n执行前向传播...")
    with torch.no_grad():
        output = encoder(sample_input, mask)

    print(f"\n输出结果:")
    print(f"  输出形状: {output.shape}")
    print(f"  预期形状: (batch_size={batch_size}, seq_length={seq_length}, embed_size={embed_size})")
    print(f"  输出统计:")
    print(f"    均值: {output.mean().item():.4f}")
    print(f"    标准差: {output.std().item():.4f}")
    print(f"    最小值: {output.min().item():.4f}")
    print(f"    最大值: {output.max().item():.4f}")

    print("\n✓ Transformer 编码器测试成功!")


if __name__ == "__main__":
    main()
