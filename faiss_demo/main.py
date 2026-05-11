"""
Faiss 向量检索测试主程序
@author thomasliu319
"""
from vector_search import generate_data, build_index, search


def main():
    # 配置参数
    data_dimension = 128      # 向量维度
    num_data_points = 10000   # 数据集向量数量
    num_queries = 5           # 查询向量数量
    k = 10                    # 返回前 k 个最近邻

    print("=" * 60)
    print("Faiss 向量检索示例")
    print("=" * 60)

    # 1. 创建数据
    print("\n[步骤 1] 生成数据")
    print("-" * 60)
    data = generate_data(num_data_points, data_dimension, seed=42)
    query_vectors = generate_data(num_queries, data_dimension, seed=123)
    print(f"Data Shape:          {data.shape}")
    print(f"Query Vectors Shape: {query_vectors.shape}")

    # 2. 创建 Faiss 索引
    print("\n[步骤 2] 创建 Faiss 索引")
    print("-" * 60)
    index = build_index(data, data_dimension)
    print(f"Is Index Trained?:        {index.is_trained}")
    print(f"Number of Vectors in Index: {index.ntotal}")

    # 3. 查询最近邻向量
    print("\n[步骤 3] 执行最近邻查询")
    print("-" * 60)
    distances, indices = search(index, query_vectors, k)

    # 输出结果
    print(f"\nQuery Results (top-{k}):")
    for i in range(num_queries):
        print(f"\nQuery {i + 1}:")
        print(f"  Nearest Neighbors' Indices: {indices[i]}")
        print(f"  Distances to Neighbors:     "
              f"{[f'{d:.4f}' for d in distances[i]]}")

    print("\n" + "=" * 60)
    print("检索完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
