from __future__ import annotations
from typing import Any, Iterable, List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from .token_usage import get_usage_collector

def _message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return str(content)


def _convert_messages(messages: Iterable[BaseMessage]) -> list[dict[str, str]]:
    payload: list[dict[str, str]] = []
    for message in messages:
        if isinstance(message, SystemMessage):
            role = "system"
        elif isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage):
            role = "assistant"
        else:
            role = "user"
        payload.append({"role": role, "content": _message_content_to_text(message.content)})
    return payload


class HuggingFaceChatModel:
    """Minimal chat wrapper using transformers for local HF models."""

    def __init__(
        self,
        *,
        model_id: str,
        device_map: str | None = "auto",
        max_new_tokens: int = 4096,
        temperature: float = 0.7,
        dtype: Any | None = torch.bfloat16,
    ) -> None:
        """Initialize parameters."""
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        self._tokenizer = AutoTokenizer.from_pretrained(model_id)

        model_kwargs: dict[str, Any] = {}
        if device_map:
            model_kwargs["device_map"] = device_map
        if dtype is not None:
            model_kwargs["dtype"] = dtype

        self._model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        # Ensure pad token is set to avoid generation warnings
        pad_id = self._tokenizer.pad_token_id or self._tokenizer.eos_token_id
        if pad_id is None and self._tokenizer.eos_token_id is not None:
            pad_id = self._tokenizer.eos_token_id
        if pad_id is not None:
            self._tokenizer.pad_token_id = pad_id
            self._tokenizer.pad_token = self._tokenizer.eos_token
            self._model.config.pad_token_id = pad_id
            # Some HF versions read from generation_config instead of config
            if hasattr(self._model, "generation_config"):
                self._model.generation_config.pad_token_id = pad_id

    def invoke(self, messages: List[BaseMessage]):
        tokenizer = self._tokenizer
        model = self._model

        messages = _convert_messages(messages)

        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        if hasattr(inputs, "to"):
            inputs = inputs.to(model.device)
        else:
            for key, value in inputs.items():
                if hasattr(value, "to"):
                    inputs[key] = value.to(model.device)

        generation_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "do_sample": self.temperature > 0,
        }
        outputs = model.generate(**inputs, **generation_kwargs)

        input_len = inputs["input_ids"].shape[-1]
        generated = outputs[0][input_len:]

        text = self._tokenizer.decode(generated, skip_special_tokens=True)
        prompt_tokens = int(input_len)
        completion_tokens = int(generated.shape[-1])

        collector = get_usage_collector()
        if collector:
            collector.add(prompt=prompt_tokens, completion=completion_tokens)

        return AIMessage(
            content=text,
            additional_kwargs={
                "token_usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                }
            },
        )
