# Self-Attention 与 Multi-Head Attention 实现

基于 PyTorch 框架实现的 Self-Attention 和 Multi-Head Attention 机制。

## 项目结构

```
attention_demo/
├── attention.py       # 注意力机制模块定义
├── main.py            # 测试主程序
├── requirements.txt   # 依赖包
└── README.md          # 项目说明
```

## 模块说明

### 1. SelfAttention (自注意力机制)
- 通过 Query、Key、Value 三个线性变换计算注意力
- 使用缩放点积注意力 (Scaled Dot-Product Attention)
- 输入输出形状: `(batch_size, seq_len, embed_size)`

### 2. MultiHeadAttention (多头注意力机制)
- 将输入拆分到多个注意力头并行计算
- 每个头独立学习不同的注意力模式
- 合并多个头的结果后通过线性层输出

## 安装依赖

```bash
pip install -r requirements.txt
```

## 运行示例

```bash
python main.py
```

## 测试参数

- **batch_size**: 2
- **seq_len**: 5
- **embed_size**: 8
- **heads**: 2

## 预期输出

程序会输出：
- Self-Attention 的输出形状和统计信息
- Multi-Head Attention 的输出形状和统计信息
- 两种模型的参数量对比
