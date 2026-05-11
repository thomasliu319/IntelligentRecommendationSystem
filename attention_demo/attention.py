"""
Self-Attention 和 Multi-Head Attention 的 PyTorch 实现
"""
import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    """Self-Attention 机制"""

    def __init__(self, embed_size):
        """
        初始化 Self-Attention 模块
        :param embed_size: 输入特征的维度
        """
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size

        # 定义查询(Query)、键(Key)、值(Value)矩阵的线性变换
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)
        self.scale = torch.sqrt(torch.FloatTensor([embed_size]))

    def forward(self, x):
        """
        前向传播
        :param x: 输入序列，形状为 (batch_size, seq_len, embed_size)
        :return: 输出序列，形状为 (batch_size, seq_len, embed_size)
        """
        # 计算查询、键和值
        Q = self.query(x)  # (batch_size, seq_len, embed_size)
        K = self.key(x)    # (batch_size, seq_len, embed_size)
        V = self.value(x)  # (batch_size, seq_len, embed_size)

        # 计算注意力分数矩阵 (batch_size, seq_len, seq_len)
        attention_scores = torch.matmul(Q, K.transpose(-1, -2)) / self.scale

        # 归一化得到注意力权重
        attention_weights = torch.softmax(attention_scores, dim=-1)

        # 使用注意力权重加权值向量 (batch_size, seq_len, embed_size)
        out = torch.matmul(attention_weights, V)
        return out


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention 机制"""

    def __init__(self, embed_size, heads):
        """
        初始化 Multi-Head Attention 模块
        :param embed_size: 输入特征的维度
        :param heads: 注意力头的数量
        """
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert self.head_dim * heads == embed_size, \
            "Embedding size needs to be divisible by heads"

        # 定义多个头的线性变换
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)

        # 输出的线性变换
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        """
        前向传播
        :param x: 输入序列，形状为 (batch_size, seq_len, embed_size)
        :return: 输出序列，形状为 (batch_size, seq_len, embed_size)
        """
        N, seq_len, embed_size = x.shape

        # 计算查询、键和值
        Q = self.query(x)  # (batch_size, seq_len, embed_size)
        K = self.key(x)    # (batch_size, seq_len, embed_size)
        V = self.value(x)  # (batch_size, seq_len, embed_size)

        # 拆分到多个头 (batch_size, heads, seq_len, head_dim)
        Q = Q.reshape(N, seq_len, self.heads, self.head_dim).transpose(1, 2)
        K = K.reshape(N, seq_len, self.heads, self.head_dim).transpose(1, 2)
        V = V.reshape(N, seq_len, self.heads, self.head_dim).transpose(1, 2)

        # 计算注意力
        attention_scores = torch.matmul(Q, K.transpose(-1, -2)) / torch.sqrt(
            torch.tensor(self.head_dim, dtype=torch.float32)
        )
        attention_weights = torch.softmax(attention_scores, dim=-1)

        # (batch_size, heads, seq_len, head_dim)
        out = torch.matmul(attention_weights, V)

        # 合并多头结果 (batch_size, seq_len, embed_size)
        out = out.transpose(1, 2).reshape(N, seq_len, self.embed_size)

        # 通过线性变换输出
        out = self.fc_out(out)
        return out
