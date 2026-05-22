# 基于机器学习的排序模块

使用随机森林回归器构建推荐系统中的精排模块，对候选物品进行评分和排序。

## 项目结构

```
ranking_demo/
├── ranking_model.py   # 排序模型核心模块
├── main.py            # 测试主程序
├── requirements.txt   # 依赖包
└── README.md          # 项目说明
```

## 核心模块说明

### ranking_model.py
- **generate_data**: 生成模拟的用户特征、物品特征和行为得分
- **train_model**: 训练 RandomForestRegressor 排序模型
- **evaluate_model**: 计算均方误差评估模型性能
- **rank_candidates**: 对候选物品进行评分并按得分降序排列

## 安装依赖

```bash
pip install -r requirements.txt
```

## 运行示例

```bash
python main.py
```

## 参数配置

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| num_samples | 500 | 训练样本数量 |
| num_features_user | 3 | 用户特征维度 |
| num_features_item | 4 | 物品特征维度 |
| n_estimators | 100 | 随机森林树的数量 |
| test_size | 0.2 | 测试集比例 |

## 运行流程

1. 生成 500 条模拟样本（用户特征 + 物品特征 + 行为得分）
2. 按 8:2 划分训练集和测试集
3. 使用 RandomForestRegressor 训练排序模型
4. 计算 MSE 评估模型精度
5. 对 10 个候选物品进行预测评分和排序
