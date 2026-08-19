"""
第2章 数据处理与特征工程
【例2-4】使用Pandas和Scikit-learn生成物品嵌入向量，
并通过余弦相似度计算商品之间的关系，以及特征交叉示例。
"""
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def item_embedding_demo():
    """物品嵌入向量与商品相似度计算"""
    print("=" * 60)
    print("例2-4：物品嵌入向量与余弦相似度")
    print("=" * 60)

    # 1. 创建示例数据（用户与商品交互行为记录）
    data = pd.DataFrame({
        '用户ID': [1, 1, 1, 2, 2, 3, 3, 4],
        '商品ID': [101, 102, 104, 101, 103, 102, 104, 103],
        '行为类型': ['点击', '购买', '点击', '购买', '点击', '点击', '购买', '购买']
    })
    print("原始交互数据：")
    print(data)

    # 2. 将行为数据转化为用户-物品交互矩阵
    weights = {'点击': 1, '购买': 5}
    data['行为权重'] = data['行为类型'].map(weights)

    interaction_matrix = data.pivot_table(
        index='用户ID',
        columns='商品ID',
        values='行为权重',
        aggfunc='sum',
        fill_value=0
    )
    print("\n用户-商品交互矩阵：")
    print(interaction_matrix)

    # 3. 使用嵌入向量表示商品特征
    item_embeddings = interaction_matrix.T.values  # 转置后行为商品，列为用户
    print("\n商品嵌入向量：")
    for i, vector in enumerate(item_embeddings):
        print(f"商品{interaction_matrix.columns[i]}的嵌入向量: {vector}")

    # 4. 计算商品之间的余弦相似度
    similarity_matrix = cosine_similarity(item_embeddings)
    similarity_df = pd.DataFrame(
        similarity_matrix,
        index=interaction_matrix.columns,
        columns=interaction_matrix.columns
    )
    print("\n商品之间的相似度矩阵：")
    print(similarity_df.round(4))

    # 5. 查找与某商品最相似的商品
    target_item = 101  # 指定目标商品
    most_similar_item = similarity_df[
        target_item].sort_values(ascending=False).index[1]
    print(f"\n与商品{target_item}最相似的商品是：商品{most_similar_item}")

    print("\n" + "=" * 60)
    print("物品嵌入向量与相似度计算完成!")
    print("=" * 60)


def feature_crossing_demo():
    """特征交叉示例"""
    print("\n\n" + "=" * 60)
    print("补充示例：特征交叉")
    print("=" * 60)

    data = pd.DataFrame({
        '用户ID': [1, 2, 3],
        '性别': ['男', '女', '男'],
        '商品类别': ['电子产品', '服装', '图书']
    })

    data['性别_商品类别'] = data['性别'] + "_" + data['商品类别']
    print("特征交叉后的数据：")
    print(data)

    print("\n" + "=" * 60)
    print("特征交叉完成!")
    print("=" * 60)


def main():
    item_embedding_demo()
    feature_crossing_demo()


if __name__ == "__main__":
    main()
