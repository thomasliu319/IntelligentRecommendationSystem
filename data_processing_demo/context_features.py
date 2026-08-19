"""
第2章 数据处理与特征工程
【例2-6】领域知识的上下文特征增强，包括时间特征提取、
地理位置特征处理及动态行为特征生成的操作步骤。
"""
import pandas as pd


def time_segment(hour):
    """根据小时划分时间段"""
    if 6 <= hour < 12:
        return '早晨'
    elif 12 <= hour < 18:
        return '下午'
    elif 18 <= hour < 24:
        return '晚上'
    else:
        return '深夜'


def context_feature_demo():
    """领域知识的上下文特征增强"""
    print("=" * 60)
    print("例2-6：上下文特征增强")
    print("=" * 60)

    # 1. 创建示例数据
    data = pd.DataFrame({
        '用户ID': [1, 2, 3, 4],
        '行为时间': ['2024-11-25 08:00:00', '2024-11-25 12:30:00',
                     '2024-11-25 20:15:00', '2024-11-26 09:45:00'],
        '地理位置': ['北京', '上海', '北京', '广州'],
        '行为类型': ['浏览', '点击', '购买', '点击']
    })
    print("原始数据：")
    print(data)

    # 2. 提取时间特征
    data['行为时间'] = pd.to_datetime(data['行为时间'])
    data['小时'] = data['行为时间'].dt.hour
    data['星期'] = data['行为时间'].dt.dayofweek
    data['时间段'] = data['小时'].apply(time_segment)

    print("\n提取时间特征后的数据：")
    print(data[['用户ID', '行为时间', '小时', '星期', '时间段']])

    # 3. 处理地理位置特征
    data = pd.concat(
        [data, pd.get_dummies(data['地理位置'], prefix='位置')], axis=1)
    print("\n地理位置特征独热编码后：")
    print(data[['用户ID', '地理位置', '位置_北京', '位置_上海', '位置_广州']])

    # 4. 动态行为特征生成
    behavior_count = data.groupby('用户ID')[
        '行为类型'].count().reset_index(name='行为次数')
    data = pd.merge(data, behavior_count, on='用户ID')
    print("\n添加动态行为特征后的数据：")
    print(data[['用户ID', '行为类型', '行为次数']])

    print("\n" + "=" * 60)
    print("上下文特征增强完成!")
    print("=" * 60)


def main():
    context_feature_demo()


if __name__ == "__main__":
    main()
