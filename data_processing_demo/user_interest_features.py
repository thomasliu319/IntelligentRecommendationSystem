"""
第2章 数据处理与特征工程
【例2-3】基于用户行为数据生成用户兴趣特征，通过Pandas进行统计分析、
加权评分和时间衰减处理，并使用余弦相似度计算商品相似度。
"""
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def user_interest_feature_demo():
    """基于用户行为数据生成用户兴趣特征"""
    print("=" * 60)
    print("例2-3：用户兴趣特征生成")
    print("=" * 60)

    # 1. 创建示例数据
    data = pd.DataFrame({
        '用户ID': [1, 1, 1, 2, 2, 3],
        '商品类别': ['电子产品', '电子产品', '服装', '电子产品', '服装', '图书'],
        '行为类型': ['点击', '购买', '点击', '收藏', '购买', '点击'],
        '行为次数': [10, 2, 5, 1, 3, 8],
        '行为时间': [1, 3, 7, 30, 5, 1]
    })
    print("原始数据：")
    print(data)

    # 2. 定义行为权重
    weights = {'点击': 1, '收藏': 2, '购买': 5}
    print(f"\n行为权重映射：{weights}")

    # 3. 计算加权兴趣值
    data['兴趣值'] = data['行为次数'] * data['行为类型'].map(weights)
    print("\n加权兴趣值计算后：")
    print(data)

    # 4. 时间衰减模型
    def time_decay(days):
        return np.exp(-0.1 * days)

    data['时间权重'] = time_decay(data['行为时间'])
    data['衰减兴趣值'] = data['兴趣值'] * data['时间权重']

    print("\n时间衰减兴趣值计算后：")
    display_data = data.copy()
    display_data['时间权重'] = display_data['时间权重'].round(4)
    display_data['衰减兴趣值'] = display_data['衰减兴趣值'].round(4)
    print(display_data)

    # 5. 按用户和商品类别汇总兴趣特征
    user_interest = data.groupby(
        ['用户ID', '商品类别'])['衰减兴趣值'].sum().reset_index()
    user_interest['衰减兴趣值'] = user_interest['衰减兴趣值'].round(4)
    print("\n用户兴趣特征汇总：")
    print(user_interest)

    print("\n" + "=" * 60)
    print("用户兴趣特征生成完成!")
    print("=" * 60)


def item_similarity_demo():
    """使用余弦相似度计算商品相似度"""
    print("\n\n" + "=" * 60)
    print("补充示例：商品嵌入向量余弦相似度")
    print("=" * 60)

    # 示例商品嵌入向量
    item_embeddings = np.array([
        [0.9, 0.1],   # 手机
        [0.8, 0.2],   # 手机壳
        [0.2, 0.9],   # 书籍
        [0.7, 0.3]    # 耳机
    ])
    item_names = ['手机', '手机壳', '书籍', '耳机']

    print("商品嵌入向量：")
    for name, vec in zip(item_names, item_embeddings):
        print(f"  {name}: {vec}")

    # 计算相似度矩阵
    similarity_matrix = cosine_similarity(item_embeddings)

    similarity_df = pd.DataFrame(
        similarity_matrix,
        index=item_names,
        columns=item_names
    ).round(4)
    print("\n商品相似度矩阵：")
    print(similarity_df)

    print("\n" + "=" * 60)
    print("商品相似度计算完成!")
    print("=" * 60)


def main():
    user_interest_feature_demo()
    item_similarity_demo()


if __name__ == "__main__":
    main()
