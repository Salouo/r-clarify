from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Sequence


_TRAILING_SHARD_RE = re.compile(r"_(\d+)_(\d+)$")


def stable_hash(parts: Sequence[Any]) -> str:
    payload = json.dumps(list(parts), ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        seen: list[str] = []
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.append(key)
        fieldnames = seen
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def normalize_text(text: str | None) -> str:
    text = (text or "").lower()
    text = re.sub(r"\[[0-9]+\]", " ", text)
    text = re.sub(r"[^a-z0-9\u3040-\u30ff\u3400-\u9fff]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def strip_action_id(text: str | None) -> str:
    return re.sub(r"^\s*\[[0-9]+\]\s*", "", text or "").strip()


def infer_condition_from_path(path: Path) -> str:
    stem = _TRAILING_SHARD_RE.sub("", path.stem)
    return stem


def infer_model_dir_from_path(path: Path) -> str | None:
    parts = path.parts
    if "results" in parts:
        idx = parts.index("results")
        if idx > 0:
            return parts[idx - 1]
    for i, part in enumerate(parts):
        if part == "outputs" and i + 1 < len(parts) and parts[i + 1] != "outputs":
            return parts[i + 1]
    return None


def load_dataset_by_index(dataset_path: Path) -> dict[int, dict[str, Any]]:
    if not dataset_path.exists():
        return {}
    data = load_json(dataset_path)
    samples = data.get("samples", []) if isinstance(data, dict) else []
    by_index: dict[int, dict[str, Any]] = {}
    for sample in samples:
        idx = sample.get("index")
        if isinstance(idx, int):
            by_index[idx] = sample
    return by_index


def first_present(row: dict[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in row and row[name] not in (None, ""):
            return row[name]
    return None


def to_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
