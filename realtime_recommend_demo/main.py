"""
实时推荐与上下文处理测试主程序
演示实时捕捉用户行为、动态特征提取和快速推荐的全流程
"""
from context_recommender import simulate_behaviors, extract_context_features, recommend_items


def main():
    print("=" * 60)
    print("实时推荐与上下文处理模块示例")
    print("=" * 60)

    # 1. 模拟用户行为数据
    print("\n[步骤 1] 模拟用户行为数据")
    print("-" * 60)
    behaviors = simulate_behaviors()
    print(behaviors)

    # 2. 实时特征提取
    print("\n[步骤 2] 实时特征提取（取最新一条行为）")
    print("-" * 60)
    latest_behavior = behaviors.iloc[-1]
    context_features = extract_context_features(behaviors, latest_behavior)
    for key, value in context_features.items():
        print(f"  {key}: {value}")

    # 3. 基于上下文生成推荐结果
    print("\n[步骤 3] 基于上下文生成推荐结果")
    print("-" * 60)
    recommended_items = recommend_items(context_features)
    print(f"  推荐物品列表: {recommended_items}")

    # 4. 模拟依次处理每条行为
    print("\n[步骤 4] 逐条处理用户行为流")
    print("-" * 60)
    for idx, row in behaviors.iterrows():
        ctx = extract_context_features(behaviors, row)
        items = recommend_items(ctx)
        print(f"  用户{ctx['user_id']} [{row['action_type']}] -> 推荐: {items}")

    print("\n" + "=" * 60)
    print("实时推荐演示完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
