# Faiss 向量检索示例

使用 Faiss 库实现向量数据库的完整检索流程，涵盖数据生成、索引创建和近邻查询。

## 项目结构

```
faiss_demo/
├── vector_search.py   # 向量检索核心模块
├── main.py            # 测试主程序
├── requirements.txt   # 依赖包
└── README.md          # 项目说明
```

## 核心模块说明

### vector_search.py
- **generate_data**: 生成指定维度和数量的随机向量
- **build_index**: 基于 `IndexFlatL2` 创建 L2 距离索引
- **search**: 在索引中查询 top-k 最近邻向量

## 安装依赖

```bash
pip install -r requirements.txt
```

如果使用 GPU 版本，请将 `faiss-cpu` 替换为 `faiss-gpu`。

## 运行示例

```bash
python main.py
```

## 参数配置

| 参数 | 值 | 说明 |
| --- | --- | --- |
| data_dimension | 128 | 向量维度 |
| num_data_points | 10000 | 数据集向量数量 |
| num_queries | 5 | 查询向量数量 |
| k | 10 | 返回最近邻数量 |

## 运行流程

1. 生成 10000 个 128 维的随机向量作为数据集
2. 生成 5 个 128 维的查询向量
3. 创建基于 L2 距离的 `IndexFlatL2` 索引
4. 将数据批量添加到索引中
5. 对每个查询向量搜索前 10 个最近邻
6. 打印索引和距离结果

## 索引类型说明

`IndexFlatL2` 是 Faiss 中最基础的索引类型：
- 基于 L2 距离（欧几里得距离）
- 不需要训练阶段（`is_trained = True`）
- 进行精确（穷举）搜索，适合中小规模数据
- 大规模数据场景可替换为 `IndexIVFFlat`、`IndexHNSWFlat` 等
