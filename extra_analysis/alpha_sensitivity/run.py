from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Sequence

from src.eval import calculate_dpass_at_k


DEFAULT_ALPHAS = (0.5, 0.6, 0.7, 0.8, 0.9)
DEFAULT_TRIALS = (3, 5)
_SHARD_SUFFIX_RE = re.compile(r"_(\d+)_(\d+)$")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Offline alpha sensitivity analysis for DPass@k and TDPass@k."
    )
    parser.add_argument(
        "--input",
        action="append",
        type=Path,
        required=True,
        help="Existing result JSON path. Can be passed multiple times.",
    )
    parser.add_argument(
        "--alphas",
        nargs="+",
        default=[str(alpha) for alpha in DEFAULT_ALPHAS],
        help="Alpha values to evaluate. Default: 0.5 0.6 0.7 0.8 0.9.",
    )
    parser.add_argument(
        "--trials",
        nargs="+",
        type=int,
        default=list(DEFAULT_TRIALS),
        help="Trial budgets for @k metrics. Default: 3 5.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional CSV output path. If omitted, no file is written.",
    )
    parser.add_argument(
        "--percent",
        action="store_true",
        help="Print metric values as percentages instead of proportions.",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=4,
        help="Decimal places for printed metrics. Default: 4.",
    )
    args = parser.parse_args()

    alphas = parse_alphas(args.alphas)
    trials = sorted(set(args.trials))
    if not trials or any(trial <= 0 for trial in trials):
        raise ValueError("trials must be positive integers.")

    rows: list[dict[str, Any]] = []
    for path in args.input:
        data = load_result_json(path)
        rows.extend(
            analyze_file(
                data=data,
                path=path,
                alphas=alphas,
                trials=trials,
                percent=args.percent,
                precision=args.precision,
            )
        )

    print_readable_table(rows, trials=trials)
    print()
    print("CSV:")
    write_csv(sys.stdout, rows)

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", encoding="utf-8", newline="") as f:
            write_csv(f, rows)
        print(f"\nWrote CSV to {args.csv}")

    return 0


def analyze_file(
    *,
    data: dict[str, Any],
    path: Path,
    alphas: Sequence[float],
    trials: Sequence[int],
    percent: bool,
    precision: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    scale = 100.0 if percent else 1.0
    model = infer_model_from_path(path)
    variant = infer_variant_from_path(path)

    for alpha in alphas:
        row: dict[str, Any] = {
            "model": model,
            "variant": variant,
            "alpha": fmt(alpha, precision),
        }
        for trial in trials:
            dpass = calculate_dpass_at_k(data=data, trial=trial, alpha=alpha)
            tdpass = calculate_tdpass_at_k(data=data, trial=trial, alpha=alpha)
            row[f"DPass@{trial}"] = fmt(dpass * scale, precision)
            row[f"TDPass@{trial}"] = fmt(tdpass * scale, precision)
        rows.append(row)
    return rows


def calculate_tdpass_at_k(data: dict[str, Any], trial: int, alpha: float) -> float:
    """Compute TDPass@k over all evaluated samples.

    A success within the trial limit contributes alpha ** (steps - 1), using
    cumulative steps through success. All other samples contribute zero.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1).")

    results = data["results"]
    if not results:
        return 0.0

    total_score = 0.0
    for res in results:
        if not res.get("correct"):
            continue
        num_trials = res.get("num_trials", 0)
        if num_trials <= 0 or num_trials > trial:
            continue
        steps_per_trial = res.get("steps_per_trial", [])
        if len(steps_per_trial) < num_trials:
            continue
        cumulative_steps = sum(steps_per_trial[:num_trials])
        total_score += alpha ** (cumulative_steps - 1)

    return total_score / len(results)


def parse_alphas(raw_values: Sequence[str]) -> list[float]:
    alphas: list[float] = []
    for raw_value in raw_values:
        for part in raw_value.split(","):
            part = part.strip()
            if not part:
                continue
            alpha = float(part)
            if not (0.0 < alpha < 1.0):
                raise ValueError(f"alpha must be in (0, 1), got {alpha}")
            if alpha not in alphas:
                alphas.append(alpha)
    return sorted(alphas)


def load_result_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or not isinstance(data.get("results"), list):
        raise ValueError(f"Expected a result JSON with a 'results' list: {path}")
    return data


def infer_model_from_path(path: Path) -> str:
    parts = path.parts
    if "outputs" in parts:
        idx = parts.index("outputs")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    if "results" in parts:
        idx = parts.index("results")
        if idx > 0:
            return parts[idx - 1]
    return "unknown"


def infer_variant_from_path(path: Path) -> str:
    return _SHARD_SUFFIX_RE.sub("", path.stem)


def print_readable_table(rows: list[dict[str, Any]], *, trials: Sequence[int]) -> None:
    columns = ["model", "variant", "alpha"]
    for trial in trials:
        columns.extend([f"DPass@{trial}", f"TDPass@{trial}"])

    widths = {
        column: max(len(column), *(len(str(row.get(column, ""))) for row in rows))
        for column in columns
    }
    header = "  ".join(column.ljust(widths[column]) for column in columns)
    rule = "  ".join("-" * widths[column] for column in columns)
    print(header)
    print(rule)
    for row in rows:
        print("  ".join(str(row.get(column, "")).ljust(widths[column]) for column in columns))


def write_csv(file_obj: Any, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    writer = csv.DictWriter(file_obj, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)


def fmt(value: float, precision: int) -> str:
    return f"{value:.{precision}f}"


if __name__ == "__main__":
    raise SystemExit(main())
