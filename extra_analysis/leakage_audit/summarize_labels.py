from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .models import LABELS


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Print four-class leakage-label counts from audit_final_labels.jsonl."
    )
    parser.add_argument(
        "path",
        help="Path to audit_final_labels.jsonl, or a result directory containing it.",
    )
    parser.add_argument(
        "--by-source",
        action="store_true",
        help="Also print separate counts for simulator/human sources.",
    )
    parser.add_argument(
        "--by-model",
        action="store_true",
        help="Also print separate counts by model_dir.",
    )
    parser.add_argument(
        "--by-condition",
        action="store_true",
        help="Also print separate counts by original_condition.",
    )
    args = parser.parse_args()

    path = resolve_input_path(Path(args.path))
    rows = read_jsonl(path)
    if not rows:
        print(f"No rows found in {path}")
        return 2

    print(f"File: {path}")
    print_summary(rows)

    if args.by_source:
        print_grouped(rows, "source")
    if args.by_model:
        print_grouped(rows, "model_dir")
    if args.by_condition:
        print_grouped(rows, "original_condition")
    return 0


def resolve_input_path(path: Path) -> Path:
    if path.is_dir():
        candidate = path / "audit_final_labels.jsonl"
        if candidate.exists():
            return candidate
        raise SystemExit(f"Directory does not contain audit_final_labels.jsonl: {path}")
    if path.exists():
        return path
    raise SystemExit(f"File not found: {path}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON on line {line_no} in {path}: {exc}") from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def print_summary(rows: list[dict[str, Any]]) -> None:
    total = len(rows)
    counts = Counter(row.get("label") for row in rows)
    print(f"Total: {total}")
    for label in LABELS:
        count = counts.get(label, 0)
        pct = 100 * count / total if total else 0.0
        print(f"{label}: {count} ({pct:.2f}%)")
    leak_count = counts.get("NEAR_DIRECT_TARGET_EXPOSURE", 0) + counts.get("DIRECT_TARGET_LEAKAGE", 0)
    leak_pct = 100 * leak_count / total if total else 0.0
    print(f"Potential leakage: {leak_count} ({leak_pct:.2f}%)")


def print_grouped(rows: list[dict[str, Any]], key: str) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key) or "unknown")].append(row)

    print("")
    print(f"By {key}:")
    for group_name in sorted(grouped):
        group_rows = grouped[group_name]
        total = len(group_rows)
        counts = Counter(row.get("label") for row in group_rows)
        leak_count = counts.get("NEAR_DIRECT_TARGET_EXPOSURE", 0) + counts.get("DIRECT_TARGET_LEAKAGE", 0)
        leak_pct = 100 * leak_count / total if total else 0.0
        label_text = ", ".join(f"{label}={counts.get(label, 0)}" for label in LABELS)
        print(f"- {group_name}: n={total}, {label_text}, potential_leakage={leak_count} ({leak_pct:.2f}%)")


if __name__ == "__main__":
    raise SystemExit(main())

