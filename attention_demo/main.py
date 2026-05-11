"""
Self-Attention 和 Multi-Head Attention 测试主程序
"""
import torch
from attention import SelfAttention, MultiHeadAttention


def main():
    # 设置随机种子保证结果可重现
    torch.manual_seed(42)

    # 示例数据参数
    batch_size = 2
    seq_len = 5
    embed_size = 8
    heads = 2

    print("=" * 60)
    print("Self-Attention 与 Multi-Head Attention 测试")
    print("=" * 60)
    print(f"\n输入参数:")
    print(f"  batch_size = {batch_size}")
    print(f"  seq_len    = {seq_len}")
    print(f"  embed_size = {embed_size}")
    print(f"  heads      = {heads}")

    # 随机输入数据
    x = torch.rand(batch_size, seq_len, embed_size)
    print(f"\n输入数据形状: {x.shape}")

    # Self-Attention 测试
    print("\n" + "-" * 60)
    print("1. Self-Attention 测试")
    print("-" * 60)
    self_attention = SelfAttention(embed_size)
    self_attention_output = self_attention(x)
    print(f"Self-Attention 输出形状: {self_attention_output.shape}")
    print(f"Self-Attention 输出均值: {self_attention_output.mean().item():.4f}")
    print(f"Self-Attention 输出标准差: {self_attention_output.std().item():.4f}")

    # Multi-Head Attention 测试
    print("\n" + "-" * 60)
    print("2. Multi-Head Attention 测试")
    print("-" * 60)
    multi_head_attention = MultiHeadAttention(embed_size, heads)
    multi_head_attention_output = multi_head_attention(x)
    print(f"Multi-Head Attention 输出形状: {multi_head_attention_output.shape}")
    print(f"Multi-Head Attention 输出均值: {multi_head_attention_output.mean().item():.4f}")
    print(f"Multi-Head Attention 输出标准差: {multi_head_attention_output.std().item():.4f}")

    # 参数量对比
    print("\n" + "-" * 60)
    print("3. 模型参数量对比")
    print("-" * 60)
    sa_params = sum(p.numel() for p in self_attention.parameters())
    mha_params = sum(p.numel() for p in multi_head_attention.parameters())
    print(f"Self-Attention 参数量:       {sa_params:}")
    print(f"Multi-Head Attention 参数量: {mha_params:}")

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
