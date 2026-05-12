"""【例1-4】Prompt工程与上下文学习技术。

使用新版 OpenAI SDK(>=1.0)封装 Chat Completions 调用。
原书中的 `openai.Completion.create` + `text-davinci-003` 已被官方下线,
这里迁移到 `chat.completions` 接口并用 gpt-4o-mini 作为默认模型,
可通过环境变量 OPENAI_MODEL 覆盖。
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from openai import OpenAI


@dataclass
class PromptEngine:
    model: str = os.getenv("OPENAI_MODEL", "deepseek-v4-pro")
    temperature: float = 0.7
    max_tokens: int = 200

    def __post_init__(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "未检测到 OPENAI_API_KEY,请在 .env 中配置或导出环境变量。"
            )
        base_url = os.getenv("OPENAI_BASE_URL")  # 允许指向兼容代理
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, prompt: str) -> str:
        """用给定 Prompt 生成一次响应。"""
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()
