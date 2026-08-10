"""
第2章 数据处理与特征工程
【例2-1】对异构数据格式进行标准化处理，包括CSV、JSON和时间戳数据，
并最终将它们合并为统一的Pandas DataFrame格式。
"""
import pandas as pd
import numpy as np
import json

pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)
pd.set_option('display.width', 100)



def create_user_data():
    """模拟用户基本信息（CSV格式）"""
    user_data = pd.DataFrame({
        '用户ID': [1, 2, 3],
        '性别': ['男', '女', '男'],
        '年龄': ['25岁', '30岁', '35岁']
    })
    return user_data


def create_product_data():
    """模拟商品信息（JSON格式）"""
    product_json = '''
[
    {"商品ID": 101, "价格": "$100.0", "类别": "电子产品"},
    {"商品ID": 102, "价格": "$200.5", "类别": "服装"},
    {"商品ID": 103, "价格": "$50.99", "类别": "图书"}
]
'''
    product_data = pd.DataFrame(json.loads(product_json))
    return product_data


def create_behavior_data():
    """模拟用户行为数据（时间戳格式）"""
    behavior_data = pd.DataFrame({
        '用户ID': [1, 2, 1, 3],
        '商品ID': [101, 103, 102, 103],
        '行为时间': [1638352800, 1638449200, 1638545600, 1638642000]
    })
    return behavior_data


def normalize_user_data(user_data):
    """标准化用户信息：去除年龄单位并转为整数"""
    user_data = user_data.copy()
    user_data['年龄'] = user_data['年龄'].str.replace('岁', '').astype(int)
    return user_data


def normalize_product_data(product_data):
    """标准化商品信息：去除货币符号并转为浮点数"""
    product_data = product_data.copy()
    product_data['价格'] = product_data['价格'].str.replace('$', '').astype(float)
    return product_data


def normalize_behavior_data(behavior_data):
    """标准化用户行为数据：将UNIX时间戳转换为标准时间格式"""
    behavior_data = behavior_data.copy()
    behavior_data['行为时间'] = pd.to_datetime(behavior_data['行为时间'], unit='s')
    return behavior_data


def merge_data(behavior_data, user_data, product_data):
    """根据用户ID和商品ID合并所有数据"""
    merged_data = behavior_data.merge(user_data, on='用户ID').merge(
        product_data, on='商品ID')
    return merged_data


def main():
    print("=" * 60)
    print("第2章 数据处理与特征工程 — 例2-1")
    print("异构数据标准化处理与合并")
    print("=" * 60)

    # 1. 模拟不同数据来源
    print("\n1. 模拟不同数据来源")
    print("-" * 40)

    user_data = create_user_data()
    print("用户基本信息（CSV格式）：")
    print(user_data)

    product_data = create_product_data()
    print("\n商品信息（JSON格式）：")
    print(product_data)

    behavior_data = create_behavior_data()
    print("\n用户行为数据（UNIX时间戳）：")
    print(behavior_data)

    # 2. 数据标准化处理
    print("\n2. 数据标准化处理")
    print("-" * 40)

    user_data = normalize_user_data(user_data)
    print("标准化后的用户基本信息：")
    print(user_data)

    product_data = normalize_product_data(product_data)
    print("\n标准化后的商品信息：")
    print(product_data)

    behavior_data = normalize_behavior_data(behavior_data)
    print("\n标准化后的用户行为数据：")
    print(behavior_data)

    # 3. 合并数据
    print("\n3. 合并数据")
    print("-" * 40)

    merged_data = merge_data(behavior_data, user_data, product_data)
    print("合并后的完整数据：")
    print(merged_data)

    print("\n" + "=" * 60)
    print("处理完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
