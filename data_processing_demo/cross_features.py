"""
第2章 数据处理与特征工程
【例2-5】使用Pandas生成交叉特征，从原始特征生成复合特征，
并分析其在推荐系统中的应用，以及时间特征提取示例。
"""
import pandas as pd


def cross_feature_demo():
    """交叉特征生成与独热编码"""
    print("=" * 60)
    print("例2-5：交叉特征生成")
    print("=" * 60)

    # 1. 创建示例数据
    data = pd.DataFrame({
        '用户ID': [1, 2, 3, 4],
        '性别': ['男', '女', '男', '女'],
        '商品类别': ['电子产品', '服装', '图书', '电子产品'],
        '购买时间': ['工作日', '周末', '工作日', '周末']
    })
    print("原始数据：")
    print(data)

    # 2. 特征交叉：性别与商品类别
    data['性别_商品类别'] = data['性别'] + "_" + data['商品类别']
    print("\n生成的交叉特征（性别_商品类别）：")
    print(data[['用户ID', '性别_商品类别']])

    # 3. 特征交叉：商品类别与购买时间
    data['商品类别_购买时间'] = data['商品类别'] + "_" + data['购买时间']
    print("\n生成的交叉特征（商品类别_购买时间）：")
    print(data[['用户ID', '商品类别_购买时间']])

    # 4. 统计特征组合分布
    feature_distribution = data['性别_商品类别'].value_counts()
    print("\n交叉特征分布统计：")
    print(feature_distribution)

    # 5. 独热编码实现交叉特征（可用于模型训练）
    encoded_features = pd.get_dummies(
        data[['性别_商品类别', '商品类别_购买时间']])
    print("\n独热编码后的特征矩阵：")
    print(encoded_features)

    print("\n" + "=" * 60)
    print("交叉特征生成完成!")
    print("=" * 60)


def time_feature_demo():
    """时间特征提取示例"""
    print("\n\n" + "=" * 60)
    print("补充示例：时间特征提取")
    print("=" * 60)

    data = pd.DataFrame({
        '用户ID': [1, 2, 3],
        '行为时间': ['2024-11-20 08:00:00',
                     '2024-11-20 15:30:00',
                     '2024-11-21 20:15:00']
    })

    # 转换为时间格式
    data['行为时间'] = pd.to_datetime(data['行为时间'])
    data['小时'] = data['行为时间'].dt.hour
    data['星期'] = data['行为时间'].dt.dayofweek
    print("时间特征提取后的数据：")
    print(data)

    print("\n" + "=" * 60)
    print("时间特征提取完成!")
    print("=" * 60)


def main():
    cross_feature_demo()
    time_feature_demo()


if __name__ == "__main__":
    main()
