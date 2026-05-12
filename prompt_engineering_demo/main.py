"""【例1-4】Prompt工程与上下文学习 —— 运行入口。

演示三种由浅入深的 Prompt 设计:
  1. 简单任务 Prompt
  2. 利用上下文(用户观影历史)的推荐
  3. 复杂任务 + Few-Shot 示例的 In-Context Learning
"""
from __future__ import annotations

from dotenv import load_dotenv

from prompt_engine import PromptEngine


SIMPLE_PROMPT = "用户喜欢科幻电影,请推荐一部相关的电影。"

USER_CONTEXT_PROMPT = """\
用户最近观看了以下电影:
1. 星际穿越
2. 火星救援
3. 复仇者联盟
根据用户的观影历史,请推荐一部电影,并简要说明理由。
"""

FEW_SHOT_PROMPT = """\
任务:为用户生成个性化电影推荐。
用户行为数据:
- 最近观看:1. 盗梦空间 2. 黑客帝国
- 偏好类型:科幻、动作
- 用户喜欢剧情复杂的电影

示例:
输入:用户喜欢喜剧电影,推荐:《大话西游》
输入:用户喜欢历史剧,推荐:《敦刻尔克》

现在请根据用户行为数据,推荐一部电影,并给出推荐理由。
"""


def _print_block(title: str, prompt: str, response: str) -> None:
    print(f"\n===== {title} =====")
    print("提示语:")
    print(prompt)
    print("模型生成:")
    print(response)


def main() -> None:
    load_dotenv()
    engine = PromptEngine()

    _print_block("示例1:简单任务", SIMPLE_PROMPT, engine.generate(SIMPLE_PROMPT))
    _print_block("示例2:基于上下文", USER_CONTEXT_PROMPT, engine.generate(USER_CONTEXT_PROMPT))
    _print_block("示例3:Few-Shot In-Context Learning", FEW_SHOT_PROMPT, engine.generate(FEW_SHOT_PROMPT))


if __name__ == "__main__":
    main()
