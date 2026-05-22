"""
实时推荐与上下文处理核心模块
提供用户行为特征提取和基于上下文的推荐算法
"""
import pandas as pd
import numpy as np


def extract_context_features(user_behaviors, behavior):
    """
    根据用户行为提取实时上下文特征

    :param user_behaviors: 全量用户行为 DataFrame
    :param behavior: 当前行为记录（dict 或 Series）
    :return: 上下文特征字典
    """
    user_id = behavior["user_id"]
    device = behavior["device"]
    location = behavior["location"]

    recent_clicks = user_behaviors[
        (user_behaviors["user_id"] == user_id)
        & (user_behaviors["action_type"] == "click")
    ]
    num_recent_clicks = len(recent_clicks)

    return {
        "user_id": user_id,
        "recent_clicks": num_recent_clicks,
        "device": device,
        "location": location,
    }


def recommend_items(context_features):
    """
    基于上下文特征生成推荐结果

    :param context_features: 上下文特征字典
    :return: 推荐物品 ID 列表
    """
    location_items_map = {
        "杭州": [201, 202, 203],
        "上海": [204, 205, 206],
        "北京": [207, 208, 209],
    }
    device_items_map = {
        "手机": [301, 302],
        "电脑": [303, 304],
        "平板": [305, 306],
    }

    location_items = location_items_map.get(context_features["location"], [])
    device_items = device_items_map.get(context_features["device"], [])

    return list(set(location_items + device_items))


def simulate_behaviors():
    """
    模拟生成用户行为数据集

    :return: 用户行为 DataFrame
    """
    behaviors = pd.DataFrame({
        "user_id": [1, 2, 1, 3, 2, 1],
        "action_type": ["click", "search", "click", "click", "search", "click"],
        "item_id": [101, 102, 103, 101, 104, 105],
        "timestamp": [
            "2024-11-26 10:00",
            "2024-11-26 10:02",
            "2024-11-26 10:05",
            "2024-11-26 10:10",
            "2024-11-26 10:15",
            "2024-11-26 10:20",
        ],
        "device": ["手机", "电脑", "手机", "平板", "手机", "手机"],
        "location": ["杭州", "上海", "杭州", "北京", "上海", "杭州"],
    })
    return behaviors
