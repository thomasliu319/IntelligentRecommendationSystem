# 实时推荐与上下文处理模块

模拟实时推荐系统的核心功能：实时捕捉用户行为、提取动态上下文特征、快速生成推荐结果。

## 项目结构

```
realtime_recommend_demo/
├── context_recommender.py  # 核心模块：特征提取与推荐算法
├── main.py                 # 测试主程序
├── requirements.txt        # 依赖包
└── README.md               # 项目说明
```

## 核心模块说明

### context_recommender.py
- **simulate_behaviors**: 模拟生成用户行为数据集（点击、搜索等）
- **extract_context_features**: 从行为记录中提取实时上下文特征（设备、地点、最近点击次数）
- **recommend_items**: 基于地点和设备类型匹配推荐物品

## 安装依赖

```bash
pip install -r requirements.txt
```

## 运行示例

```bash
python main.py
```

## 运行流程

1. 模拟 6 条用户行为数据（包含用户ID、行为类型、物品ID、时间戳、设备、地点）
2. 提取最新行为的实时上下文特征
3. 基于地点和设备类型匹配候选物品
4. 逐条处理用户行为流，输出每条行为触发的推荐结果
