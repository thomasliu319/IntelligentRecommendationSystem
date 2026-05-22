"""
排序模型测试主程序
演示基于随机森林回归的推荐排序流程
"""
import numpy as np
import pandas as pd
from ranking_model import generate_data, train_model, evaluate_model, rank_candidates
from sklearn.model_selection import train_test_split


def main():
    # 参数配置
    num_samples = 500
    num_features_user = 3
    num_features_item = 4
    test_ratio = 0.2

    print("=" * 60)
    print("基于机器学习的排序模型示例")
    print("=" * 60)

    # 1. 构造模拟数据
    print("\n[步骤 1] 构造模拟数据")
    print("-" * 60)
    features, labels, feature_names = generate_data(
        num_samples, num_features_user, num_features_item
    )

    data = pd.DataFrame(features, columns=feature_names)
    data["score"] = labels
    print(data.head())

    # 2. 划分训练集和测试集
    print("\n[步骤 2] 划分训练集和测试集")
    print("-" * 60)
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_ratio, random_state=42
    )
    print(f"训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}")

    # 3. 训练排序模型
    print("\n[步骤 3] 训练随机森林排序模型")
    print("-" * 60)
    model = train_model(X_train, y_train, n_estimators=100)
    print(f"模型: RandomForestRegressor(n_estimators=100)")

    # 4. 模型评估
    print("\n[步骤 4] 模型评估")
    print("-" * 60)
    mse, _ = evaluate_model(model, X_test, y_test)
    print(f"均方误差 (MSE): {mse:.4f}")

    # 5. 对候选物品进行排序
    print("\n[步骤 5] 候选物品评分与排序")
    print("-" * 60)
    num_candidates = 10
    candidate_user_features = np.random.random(
        (num_candidates, num_features_user)
    )
    candidate_item_features = np.random.random(
        (num_candidates, num_features_item)
    )
    candidate_features = np.hstack(
        [candidate_user_features, candidate_item_features]
    )

    ranked = rank_candidates(model, candidate_features)
    print(f"\n候选物品排序结果 (共 {num_candidates} 个):")
    for rank, (idx, score) in enumerate(ranked):
        print(f"  排名 {rank+1}: 物品索引 {idx}, 预测得分 {score:.4f}")

    print("\n" + "=" * 60)
    print("排序完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
