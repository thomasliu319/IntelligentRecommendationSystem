"""
第2章 数据处理与特征工程
【例2-2】数据噪声过滤与异常检测，包括去重、缺失值填补、异常值过滤、
离群点检测，以及用户行为权重计算与时间衰减。
"""
import pandas as pd
import numpy as np


def noise_filtering_demo():
    """数据噪声过滤与异常检测"""
    print("=" * 60)
    print("例2-2：数据噪声过滤与异常检测")
    print("=" * 60)

    # 1. 模拟数据
    data = pd.DataFrame({
        '用户ID': [1, 1, 2, 2, 3, 4, 5],
        '商品ID': [101, 101, 102, 103, 104, 105, 106],
        '浏览次数': [3, 3, 2, np.nan, 10, 1, 200],
        '价格': [100, 100, 200, 0, 50, -20, 1000000]
    })
    print("原始数据：")
    print(data)

    # 2. 数据噪声过滤
    print("\n" + "-" * 40)
    print("2. 数据噪声过滤")
    print("-" * 40)

    # 2.1 删除重复数据
    data = data.drop_duplicates(subset=['用户ID', '商品ID'], keep='first')
    print("删除重复数据后：")
    print(data)

    # 2.2 填补缺失值
    data['浏览次数'] = data['浏览次数'].fillna(data['浏览次数'].median())
    print("\n填补缺失值后（浏览次数用中位数填补）：")
    print(data)

    # 3. 异常检测与处理
    print("\n" + "-" * 40)
    print("3. 异常检测与处理")
    print("-" * 40)

    # 3.1 规则过滤（去除价格异常值）
    data = data[(data['价格'] > 0) & (data['价格'] < 100000)]
    print("过滤价格异常值后（价格范围 0-100000）：")
    print(data)

    # 3.2 统计方法检测离群点
    mean_views = data['浏览次数'].mean()
    std_views = data['浏览次数'].std()
    outliers = data[np.abs(data['浏览次数'] - mean_views) > 3 * std_views]
    print("\n检测到的离群点（3-sigma规则）：")
    print(outliers)

    # 删除离群点
    data = data[np.abs(data['浏览次数'] - mean_views) <= 3 * std_views]
    print("\n删除离群点后：")
    print(data)

    # 4. 输出最终清洗后的数据
    print("\n" + "-" * 40)
    print("4. 清洗后的最终数据")
    print("-" * 40)
    print(data)

    print("\n" + "=" * 60)
    print("噪声过滤与异常检测完成!")
    print("=" * 60)


def user_interest_weight_demo():
    """用户行为权重计算"""
    print("\n\n" + "=" * 60)
    print("补充示例：用户行为权重计算")
    print("=" * 60)

    data = pd.DataFrame({
        '用户ID': [1, 1, 1, 2, 2],
        '行为类型': ['点击', '购买', '收藏', '点击', '购买'],
        '商品类别': ['电子产品', '电子产品', '电子产品', '服装', '服装'],
        '次数': [5, 2, 1, 2, 1]
    })
    print("原始行为数据：")
    print(data)

    weights = {'点击': 1, '收藏': 2, '购买': 5}
    data['兴趣值'] = data['次数'] * data['行为类型'].map(weights)
    print(f"\n行为权重映射：{weights}")
    print("加权后的数据：")
    print(data)

    user_interest = data.groupby(['用户ID', '商品类别'])['兴趣值'].sum()
    print("\n按用户和类别汇总的兴趣值：")
    print(user_interest)

    print("\n" + "=" * 60)
    print("用户行为权重计算完成!")
    print("=" * 60)


def time_decay_demo():
    """时间衰减特征工程"""
    print("\n\n" + "=" * 60)
    print("补充示例：时间衰减特征")
    print("=" * 60)

    def time_decay(days):
        return np.exp(-0.1 * days)

    data = pd.DataFrame({
        '行为时间': [1, 7, 30],
        '行为次数': [5, 2, 1]
    })
    print("原始行为数据：")
    print(data)

    data['权重'] = time_decay(data['行为时间'])
    data['衰减兴趣值'] = data['行为次数'] * data['权重']
    print(f"\n时间衰减公式：exp(-0.1 * days)")

    # 格式化显示衰减权重
    display_data = data.copy()
    display_data['权重'] = display_data['权重'].round(4)
    display_data['衰减兴趣值'] = display_data['衰减兴趣值'].round(4)
    print("应用时间衰减后：")
    print(display_data)

    print("\n" + "=" * 60)
    print("时间衰减特征计算完成!")
    print("=" * 60)


def main():
    noise_filtering_demo()
    user_interest_weight_demo()
    time_decay_demo()


if __name__ == "__main__":
    main()
