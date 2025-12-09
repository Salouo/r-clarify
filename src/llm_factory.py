"""Factory for selecting which LLM implementation to use in this run."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from .hf_llm import HuggingFaceChatModel
from .gpt_llm import OpenAIChatModel


# 简单的配置开关：手动指定本次要使用的后端
DEFAULT_BACKEND = "gpt"  # 取值 "gpt" 或 "hf"

# OpenAI (GPT) 配置
GPT_MODEL_NAME = "gpt-4o-mini"
GPT_TEMPERATURE = 0.7

# Hugging Face (本地/自建) 模型配置
HF_MODEL_ID = "Qwen/Qwen2-7B-Instruct"
HF_MAX_NEW_TOKENS = 512
HF_TEMPERATURE = 0.7


@lru_cache(maxsize=1)
def get_primary_llm(backend: str | None = None) -> Any:
    """Return the LLM instance to use for router/clarify/reflection."""

    target = (backend or DEFAULT_BACKEND).lower()
    if target == "hf":
        return HuggingFaceChatModel(
            model_id=HF_MODEL_ID,
            max_new_tokens=HF_MAX_NEW_TOKENS,
            temperature=HF_TEMPERATURE,
        )
    if target == "gpt":
        return OpenAIChatModel(
            model=GPT_MODEL_NAME,
            temperature=GPT_TEMPERATURE,
        )
    raise ValueError("backend 只能是 'gpt' 或 'hf'")
