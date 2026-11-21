"""Model selection and configuration for experiment runs."""

from __future__ import annotations

from functools import lru_cache
from threading import Lock
from typing import Any, Iterable

import torch

from .hf_llm import HuggingFaceChatModel
from .gpt_llm import OpenAIChatModel


DEFAULT_BACKEND = "gpt"  # Select "gpt" or "hf".

GPT_MODEL_NAME = "gpt-5"
GPT_TEMPERATURE: float | None = None

HF_MODEL_ID = "Qwen/Qwen3-Next-80B-A3B-Instruct"
HF_MAX_NEW_TOKENS = 2048
HF_TEMPERATURE = 0.7
HF_DEVICE_MAP = "auto"


def get_primary_model_name() -> str:
    """Return the model identifier for the selected backend."""
    target = (DEFAULT_BACKEND).lower()
    if target == "hf":
        return HF_MODEL_ID
    if target == "gpt":
        return GPT_MODEL_NAME
    raise ValueError("backend 只能是 'gpt' 或 'hf'")


def get_primary_model_dirname() -> str:
    """Return the final component of the model identifier for output paths."""
    name = get_primary_model_name()
    return name.split("/")[-1]


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
    """Return the shared model used for agent decisions and reflection."""

    target = (DEFAULT_BACKEND).lower()
    if target == "hf":
        return HuggingFaceChatModel(
            model_id=HF_MODEL_ID,
            device_map=HF_DEVICE_MAP,
            max_new_tokens=HF_MAX_NEW_TOKENS,
            temperature=HF_TEMPERATURE,
            dtype=torch.bfloat16,  # Use bfloat16 to reduce GPU memory use.
        )
    if target == "gpt":
        return OpenAIChatModel(
            model=GPT_MODEL_NAME,
            temperature=GPT_TEMPERATURE,
        )
    raise ValueError("backend 只能是 'gpt' 或 'hf'")
