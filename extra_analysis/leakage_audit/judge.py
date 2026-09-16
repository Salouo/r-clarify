from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import BadRequestError, OpenAI, RateLimitError
from tqdm import tqdm

from .lexical_audit import audit_record
from .models import LABELS, ClarificationRecord
from .prompts import build_judge_messages
from .utils import append_jsonl, read_jsonl, stable_hash


_FENCE_RE = re.compile(r"```[a-zA-Z0-9_-]*\s*([\s\S]*?)\s*```", re.MULTILINE)


def parse_judge_json(text: str) -> dict[str, Any]:
    cleaned = _strip_to_json(text)
    data = json.loads(cleaned)
    if not isinstance(data, dict):
        raise ValueError("Judge response must be a JSON object.")
    label = data.get("label")
    if label not in LABELS:
        raise ValueError(f"Unknown judge label: {label!r}")
    parsed = {"label": label}
    for key in (
        "question_precisely_elicited_the_information",
        "answer_truthfully_responds_to_question",
        "target_object_explicitly_mentioned",
        "gold_action_explicitly_requested",
        "volunteered_gold_specific_information",
    ):
        parsed[key] = _as_bool(data.get(key))
    parsed["brief_reason"] = str(data.get("brief_reason", "")).strip().splitlines()[0][:240]
    return parsed


class JudgeCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._rows_by_key = {row.get("cache_key"): row for row in read_jsonl(path) if row.get("cache_key")}

    def get(self, key: str) -> dict[str, Any] | None:
        return self._rows_by_key.get(key)

    def put(self, key: str, row: dict[str, Any]) -> None:
        payload = {"cache_key": key, **row}
        self._rows_by_key[key] = payload
        append_jsonl(self.path, [payload])


class OpenAIJudge:
    def __init__(self, *, model: str, temperature: float = 0.0) -> None:
        load_dotenv(".env", override=False)
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key) if api_key else OpenAI()
        self.model = model
        self.temperature = temperature

    def judge_once(self, record: ClarificationRecord) -> tuple[dict[str, Any], dict[str, Any]]:
        messages = build_judge_messages(record)
        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
        }
        try:
            response = self.client.chat.completions.create(**request_kwargs)
        except BadRequestError as err:
            err_msg = str(err).lower()
            if "response_format" in err_msg:
                request_kwargs.pop("response_format", None)
            if "temperature" in err_msg:
                request_kwargs.pop("temperature", None)
            response = self.client.chat.completions.create(**request_kwargs)
        content = response.choices[0].message.content or ""
        usage = getattr(response, "usage", None)
        usage_dict = {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
            "completion_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
            "total_tokens": getattr(usage, "total_tokens", 0) if usage else 0,
        }
        return parse_judge_json(content), usage_dict


def judge_records(
    records: list[ClarificationRecord],
    *,
    model: str,
    temperature: float,
    num_runs: int,
    cache_path: Path,
    max_retries: int = 4,
    sleep_seconds: float = 2.0,
    show_progress: bool = True,
) -> list[dict[str, Any]]:
    cache = JudgeCache(cache_path)
    judge = OpenAIJudge(model=model, temperature=temperature)
    rows: list[dict[str, Any]] = []
    total = len(records) * num_runs
    progress = tqdm(total=total, desc=f"Judging {model}", unit="call", dynamic_ncols=True) if show_progress and total else None
    try:
        for record_idx, record in enumerate(records, start=1):
            sample_label = record.instance_id or record.record_id
            for run_idx in range(1, num_runs + 1):
                if progress is not None:
                    progress.set_postfix_str(
                        f"record={record_idx}/{len(records)}, sample={sample_label}, run={run_idx}/{num_runs}"
                    )
                messages = build_judge_messages(record)
                cache_key = stable_hash([model, temperature, run_idx, messages])
                cached = cache.get(cache_key)
                if cached:
                    rows.append(_cache_row_to_output(record, run_idx, cached))
                    if progress is not None:
                        progress.set_postfix_str(
                            f"record={record_idx}/{len(records)}, sample={sample_label}, run={run_idx}/{num_runs}, status=cache"
                        )
                        progress.update(1)
                    continue
                parsed, usage = _judge_with_retries(judge, record, max_retries=max_retries, sleep_seconds=sleep_seconds)
                row = {
                    "record_id": record.record_id,
                    "judge_run": run_idx,
                    "judge_model": model,
                    "temperature": temperature,
                    **parsed,
                    "usage": usage,
                }
                cache.put(cache_key, row)
                rows.append(row)
                if progress is not None:
                    progress.set_postfix_str(
                        f"record={record_idx}/{len(records)}, sample={sample_label}, run={run_idx}/{num_runs}, status=api"
                    )
                    progress.update(1)
    finally:
        if progress is not None:
            progress.close()
    return rows


def heuristic_label_for_tests(record: ClarificationRecord) -> str:
    lexical = audit_record(record)
    question = (record.clarification_question or "").lower()
    answer = (record.response or "").lower()
    eliciting_question = any(
        phrase in question
        for phrase in (
            "which",
            "what would",
            "what do you want",
            "what would you like",
            "where would",
            "where should",
            "do you want",
            "would you prefer",
        )
    )
    if lexical["obvious_robot_command"]:
        return "DIRECT_TARGET_LEAKAGE"
    if eliciting_question and answer.strip():
        return "LEGITIMATE_INTENT_SIGNAL"
    if lexical["target_object_explicitly_mentioned"]:
        return "NEAR_DIRECT_TARGET_EXPOSURE"
    return "LEGITIMATE_INTENT_SIGNAL" if answer.strip() and answer.strip().lower() not in {"cannot answer", "unknown"} else "NO_TARGET_SIGNAL"


def _judge_with_retries(
    judge: OpenAIJudge,
    record: ClarificationRecord,
    *,
    max_retries: int,
    sleep_seconds: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            return judge.judge_once(record)
        except (json.JSONDecodeError, ValueError, RateLimitError) as exc:
            last_error = exc
            time.sleep(sleep_seconds * (2**attempt))
    assert last_error is not None
    raise last_error


def _cache_row_to_output(record: ClarificationRecord, run_idx: int, cached: dict[str, Any]) -> dict[str, Any]:
    row = {k: v for k, v in cached.items() if k != "cache_key"}
    row["record_id"] = record.record_id
    row["judge_run"] = run_idx
    row["cache_hit"] = True
    return row


def _strip_to_json(text: str) -> str:
    text = (text or "").strip()
    fenced = _FENCE_RE.findall(text)
    if fenced:
        text = fenced[0].strip()
    if text.startswith("{") and text.endswith("}"):
        return text
    start = text.find("{")
    if start < 0:
        raise ValueError("No JSON object found in judge response.")
    depth = 0
    in_string = False
    escaped = False
    for idx, char in enumerate(text[start:], start=start):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    raise ValueError("Unbalanced JSON object in judge response.")


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)
