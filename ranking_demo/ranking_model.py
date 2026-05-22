"""
排序模块核心实现
使用随机森林回归器构建排序模型，对候选物品进行评分和排序
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def generate_data(num_samples=500, num_features_user=3, num_features_item=4, seed=42):
    """
    构造模拟数据：用户特征、物品特征和用户行为得分

    :param num_samples: 样本数量
    :param num_features_user: 用户特征维度
    :param num_features_item: 物品特征维度
    :param seed: 随机种子
    :return: (features, labels, feature_names) 特征矩阵、标签和特征列名
    """
    np.random.seed(seed)

    user_features = np.random.random((num_samples, num_features_user))
    item_features = np.random.random((num_samples, num_features_item))
    labels = np.random.uniform(0, 1, num_samples)

    features = np.hstack([user_features, item_features])

    feature_names = (
        [f"user_feat_{i+1}" for i in range(num_features_user)]
        + [f"item_feat_{i+1}" for i in range(num_features_item)]
    )

    return features, labels, feature_names


def train_model(X_train, y_train, n_estimators=100, seed=42):
    """
    训练随机森林回归排序模型

    :param X_train: 训练特征
    :param y_train: 训练标签
    :param n_estimators: 树的数量
    :param seed: 随机种子
    :return: 训练好的模型
    """
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=seed)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """
    评估模型并返回均方误差

    :param model: 训练好的模型
    :param X_test: 测试特征
    :param y_test: 测试标签
    :return: 均方误差值
    """
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse, y_pred


def rank_candidates(model, candidate_features):
    """
    对候选物品进行评分并返回排序结果

    :param model: 训练好的排序模型
    :param candidate_features: 候选物品特征矩阵
    :return: 排序后的 (索引, 得分) 列表，得分从高到低
    """
    scores = model.predict(candidate_features)
    sorted_indices = np.argsort(-scores)
    return [(idx, scores[idx]) for idx in sorted_indices]
