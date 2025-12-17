"""Factory for selecting which LLM implementation to use in this run."""

from __future__ import annotations

from functools import lru_cache
from threading import Lock
from typing import Any, Iterable

import torch

from .hf_llm import HuggingFaceChatModel
from .gpt_llm import OpenAIChatModel


# 简单的配置开关：手动指定本次要使用的后端
DEFAULT_BACKEND = "gpt"  # 取值 "gpt" 或 "hf"

# OpenAI (GPT) 配置
GPT_MODEL_NAME = "gpt-5-mini"
GPT_TEMPERATURE: float | None = None

# Hugging Face (本地/自建) 模型配置
# meta-llama/Llama-3.1-8B-Instruct
# Qwen/Qwen3-4B-Instruct-2507
HF_MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507"
HF_MAX_NEW_TOKENS = 2048
HF_TEMPERATURE = 0.7
HF_DEVICE_MAP = "auto"  # e.g. "auto", "balanced"; use auto to shard one model across visible GPUs


def get_primary_model_name() -> str:
    """Return the model name/id for the selected backend."""
    target = (DEFAULT_BACKEND).lower()
    if target == "hf":
        return HF_MODEL_ID
    if target == "gpt":
        return GPT_MODEL_NAME
    raise ValueError("backend 只能是 'gpt' 或 'hf'")


def get_primary_model_dirname() -> str:
    """Filesystem-safe dirname for logging outputs."""
    name = get_primary_model_name()
    return name.replace("/", "__")


class RoundRobinLLM:
    """Dispatch invoke() calls across multiple LLM instances for parallel use."""

    def __init__(self, llms: Iterable[Any]) -> None:
        self._llms = list(llms)
        if not self._llms:
            raise ValueError("RoundRobinLLM requires at least one LLM instance")
        self._lock = Lock()
        self._index = 0

    def invoke(self, messages):
        with self._lock:
            llm = self._llms[self._index]
            self._index = (self._index + 1) % len(self._llms)
        return llm.invoke(messages)


@lru_cache(maxsize=1)
def get_primary_llm() -> Any:
    """Return the LLM instance to use for agent/clarify/reflection."""

    target = (DEFAULT_BACKEND).lower()
    if target == "hf":
        return HuggingFaceChatModel(
            model_id=HF_MODEL_ID,
            device_map=HF_DEVICE_MAP,
            max_new_tokens=HF_MAX_NEW_TOKENS,
            temperature=HF_TEMPERATURE,
            dtype=torch.bfloat16, # 使用 BF16 降低显存占用
        )
    if target == "gpt":
        return OpenAIChatModel(
            model=GPT_MODEL_NAME,
            temperature=GPT_TEMPERATURE,
        )
    raise ValueError("backend 只能是 'gpt' 或 'hf'")
