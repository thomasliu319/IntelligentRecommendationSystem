# Transformer 模型示例

这是一个使用 PyTorch 从零实现的简化版 Transformer 编码器模型。

## 项目结构

```
transformer_demo/
├── model.py           # Transformer 模型定义
├── main.py            # 测试主程序
├── requirements.txt   # 依赖包
└── README.md          # 项目说明
```

## 模型组件

### 1. SelfAttention (自注意力机制)
- 实现多头自注意力机制
- 支持可选的 mask 操作
- 使用 einsum 进行高效的张量运算

### 2. TransformerBlock (Transformer 块)
- 包含自注意力层和前馈神经网络
- 使用残差连接和层归一化
- 支持 Dropout 正则化

### 3. Encoder (编码器)
- 包含词嵌入和位置嵌入
- 堆叠多个 Transformer 块
- 可配置层数和参数

## 安装依赖

```bash
pip install -r requirements.txt
```

## 运行示例

```bash
python main.py
```

## 模型参数

- **embed_size**: 256 (嵌入维度)
- **num_layers**: 2 (编码器层数)
- **heads**: 8 (注意力头数)
- **dropout**: 0.1 (Dropout 率)
- **forward_expansion**: 4 (前馈网络扩展倍数)
- **src_vocab_size**: 10000 (词汇表大小)
- **max_length**: 100 (最大序列长度)

## 输出说明

程序会输出：
- 使用的设备 (CPU/CUDA)
- 模型配置信息
- 模型参数统计
- 输入数据信息
- 编码器输出形状和统计信息

## 扩展建议

1. 添加解码器部分实现完整的 Transformer
2. 实现训练循环和损失函数
3. 添加数据加载和预处理模块
4. 实现模型保存和加载功能
5. 添加可视化注意力权重的功能
