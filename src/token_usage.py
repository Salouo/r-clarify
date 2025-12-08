from dataclasses import dataclass
from typing import Dict, Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult


@dataclass
class TokenStats:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    def add_usage(self, usage: Dict[str, Any]) -> None:
        self.input_tokens += int(usage.get("input_tokens", 0) or 0)
        self.output_tokens += int(usage.get("output_tokens", 0) or 0)
        self.total_tokens += int(usage.get("total_tokens", 0) or 0)


# 全局的 token 统计（一个 Python 进程内累计）
global_token_stats = TokenStats()


class TokenUsageCallback(BaseCallbackHandler):
    """收集所有 LLM 调用的 token 使用量。"""

    def on_llm_end(self, output: LLMResult, **kwargs) -> None:
        # 对于 ChatOpenAI，output.generations 是 ChatGeneration 的二维列表
        for gen_list in output.generations:
            for gen in gen_list:
                msg = getattr(gen, "message", None)
                if msg is None:
                    continue
                usage = getattr(msg, "usage_metadata", None)
                if not usage:
                    continue
                global_token_stats.add_usage(usage)


# 给外部模块直接使用的单例 callback
token_usage_callback = TokenUsageCallback()

