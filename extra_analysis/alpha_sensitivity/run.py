from __future__ import annotations

import argparse
import csv
import glob
import json
import re
from pathlib import Path
from typing import Any, Iterable

from src.eval import (
    calculate_accuracy_by_trial,
    calculate_average_steps_of_successful_trial,
    calculate_dpass_at_k,
)


DEFAULT_RESULTS_GLOB = "outputs/**/results/*.json"
DEFAULT_OUTPUT_DIR = Path("extra_analysis/alpha_sensitivity/results")
DEFAULT_ALPHAS = "0.50,0.60,0.70,0.75,0.80,0.85,0.90,0.95"
_SHARD_SUFFIX_RE = re.compile(r"_(\d+)_(\d+)$")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run post-hoc sensitivity analysis for DPass@k(alpha)."
    )
    parser.add_argument(
        "--results-glob",
        action="append",
        default=None,
        help=(
            "Result JSON glob to analyze. Can be passed multiple times. "
            f"Default: {DEFAULT_RESULTS_GLOB}"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for analysis outputs. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--trial",
        type=int,
        default=3,
        help="Allowed trial count used in Pass@trial and DPass@trial. Default: 3.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Evaluate only the first k samples in each result file. Default: full file.",
    )
    parser.add_argument(
        "--alphas",
        default=DEFAULT_ALPHAS,
        help=f"Comma-separated alpha values in (0, 1). Default: {DEFAULT_ALPHAS}",
    )
    parser.add_argument(
        "--baseline-alpha",
        type=float,
        default=0.8,
        help="Alpha used as the baseline for delta columns. Default: 0.8.",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=6,
        help="Decimal places written for floating-point metrics. Default: 6.",
    )
    args = parser.parse_args()

    if not (0.0 < args.baseline_alpha < 1.0):
        raise ValueError(f"baseline alpha must be in (0, 1), got {args.baseline_alpha}")

    alphas = parse_alphas(args.alphas)
    if args.baseline_alpha not in alphas:
        alphas = sorted([*alphas, args.baseline_alpha])

    result_paths = discover_result_paths(args.results_glob or [DEFAULT_RESULTS_GLOB])
    if not result_paths:
        print("No result JSON files found.")
        return 2

    rows = analyze_paths(
        result_paths,
        trial=args.trial,
        k=args.k,
        alphas=alphas,
        baseline_alpha=args.baseline_alpha,
        precision=args.precision,
    )
    summary_rows = summarize_by_file(rows, precision=args.precision)
    ranking_rows = rank_by_alpha(rows, precision=args.precision)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "alpha_sensitivity.csv", rows)
    write_csv(args.output_dir / "summary.csv", summary_rows)
    write_csv(args.output_dir / "rankings.csv", ranking_rows)
    write_json(
        args.output_dir / "config.json",
        {
            "results_glob": args.results_glob or [DEFAULT_RESULTS_GLOB],
            "num_result_files": len(result_paths),
            "trial": args.trial,
            "k": args.k,
            "alphas": alphas,
            "baseline_alpha": args.baseline_alpha,
            "outputs": [
                "alpha_sensitivity.csv",
                "summary.csv",
                "rankings.csv",
                "summary.md",
            ],
        },
    )
    write_markdown_summary(
        args.output_dir / "summary.md",
        rows=rows,
        summary_rows=summary_rows,
        ranking_rows=ranking_rows,
        trial=args.trial,
        k=args.k,
        baseline_alpha=args.baseline_alpha,
    )

    print(f"Analyzed {len(result_paths)} result files.")
    print(f"Wrote alpha sensitivity outputs under {args.output_dir}")
    return 0


def parse_alphas(raw: str) -> list[float]:
    alphas: list[float] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        alpha = float(part)
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if alpha not in alphas:
            alphas.append(alpha)
    if not alphas:
        raise ValueError("At least one alpha value is required.")
    return sorted(alphas)


def discover_result_paths(patterns: Iterable[str]) -> list[Path]:
    paths: set[Path] = set()
    for pattern in patterns:
        for match in glob.glob(pattern, recursive=True):
            path = Path(match)
            if path.is_file():
                paths.add(path)
    return sorted(paths)


def analyze_paths(
    paths: list[Path],
    *,
    trial: int,
    k: int | None,
    alphas: list[float],
    baseline_alpha: float,
    precision: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        data = load_json(path)
        results = data.get("results", [])
        evaluated_results = results[:k] if k is not None else results
        num_samples = len(evaluated_results)
        num_success = count_successes(data, trial=trial, k=k)
        pass_at_trial = calculate_accuracy_by_trial(data=data, trial=trial, k=k)
        avg_steps_success = calculate_average_steps_of_successful_trial(
            data=data,
            trial=trial,
            k=k,
        )
        baseline_dpass = calculate_dpass_at_k(
            data=data,
            trial=trial,
            alpha=baseline_alpha,
            k=k,
        )
        scores = {
            alpha: calculate_dpass_at_k(data=data, trial=trial, alpha=alpha, k=k)
            for alpha in alphas
        }
        for alpha in alphas:
            dpass = scores[alpha]
            rows.append(
                {
                    "model": infer_model_from_path(path),
                    "condition": infer_condition_from_path(path),
                    "result_path": str(path),
                    "trial": trial,
                    "k": k if k is not None else "full",
                    "num_samples": num_samples,
                    "num_success": num_success,
                    "pass_at_trial": fmt(pass_at_trial, precision),
                    "avg_steps_success": fmt(avg_steps_success, precision),
                    "alpha": fmt(alpha, precision),
                    "dpass": fmt(dpass, precision),
                    "baseline_alpha": fmt(baseline_alpha, precision),
                    "baseline_dpass": fmt(baseline_dpass, precision),
                    "delta_from_baseline": fmt(dpass - baseline_dpass, precision),
                    "relative_delta_from_baseline_pct": fmt(
                        percent_change(dpass, baseline_dpass),
                        precision,
                    ),
                }
            )
    return rows


def summarize_by_file(rows: list[dict[str, Any]], *, precision: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["result_path"], []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for result_path, file_rows in grouped.items():
        ordered = sorted(file_rows, key=lambda row: float(row["alpha"]))
        dpass_values = [float(row["dpass"]) for row in ordered]
        first = ordered[0]
        summary_rows.append(
            {
                "model": first["model"],
                "condition": first["condition"],
                "result_path": result_path,
                "trial": first["trial"],
                "k": first["k"],
                "num_samples": first["num_samples"],
                "num_success": first["num_success"],
                "pass_at_trial": first["pass_at_trial"],
                "avg_steps_success": first["avg_steps_success"],
                "min_alpha": ordered[0]["alpha"],
                "max_alpha": ordered[-1]["alpha"],
                "min_dpass": fmt(min(dpass_values), precision),
                "max_dpass": fmt(max(dpass_values), precision),
                "dpass_range": fmt(max(dpass_values) - min(dpass_values), precision),
                "baseline_alpha": first["baseline_alpha"],
                "baseline_dpass": first["baseline_dpass"],
            }
        )
    return sorted(summary_rows, key=lambda row: (row["model"], row["condition"], row["result_path"]))


def rank_by_alpha(rows: list[dict[str, Any]], *, precision: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["alpha"], []).append(row)

    ranking_rows: list[dict[str, Any]] = []
    for alpha, alpha_rows in sorted(grouped.items(), key=lambda item: float(item[0])):
        ranked = sorted(
            alpha_rows,
            key=lambda row: (float(row["dpass"]), float(row["pass_at_trial"])),
            reverse=True,
        )
        for rank, row in enumerate(ranked, start=1):
            ranking_rows.append(
                {
                    "alpha": alpha,
                    "rank": rank,
                    "model": row["model"],
                    "condition": row["condition"],
                    "dpass": fmt(float(row["dpass"]), precision),
                    "pass_at_trial": row["pass_at_trial"],
                    "avg_steps_success": row["avg_steps_success"],
                    "num_samples": row["num_samples"],
                    "result_path": row["result_path"],
                }
            )
    return ranking_rows


def write_markdown_summary(
    path: Path,
    *,
    rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    ranking_rows: list[dict[str, Any]],
    trial: int,
    k: int | None,
    baseline_alpha: float,
) -> None:
    lines = [
        "# Alpha Sensitivity Analysis",
        "",
        f"- Metric: `DPass@{trial}(alpha)`",
        f"- k: `{k if k is not None else 'full'}`",
        f"- Baseline alpha: `{baseline_alpha}`",
        f"- Result files: `{len(summary_rows)}`",
        f"- Alpha values: `{', '.join(sorted({row['alpha'] for row in rows}, key=float))}`",
        "",
        "## Per-File Stability",
        "",
        "| model | condition | Pass | baseline DPass | DPass range |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            "| {model} | {condition} | {pass_at_trial} | {baseline_dpass} | {dpass_range} |".format(
                **row
            )
        )

    lines.extend(
        [
            "",
            "## Top Rank By Alpha",
            "",
            "| alpha | model | condition | DPass | Pass |",
            "| ---: | --- | --- | ---: | ---: |",
        ]
    )
    top_rows = [row for row in ranking_rows if row["rank"] == 1]
    for row in top_rows:
        lines.append(
            "| {alpha} | {model} | {condition} | {dpass} | {pass_at_trial} |".format(**row)
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def count_successes(data: dict[str, Any], *, trial: int, k: int | None) -> int:
    results = data.get("results", [])
    if k is not None:
        results = results[:k]
    count = 0
    for res in results:
        if res.get("correct") and 0 < res.get("num_trials", 0) <= trial:
            count += 1
    return count


def load_json(path: Path) -> dict[str, Any]:
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


def infer_condition_from_path(path: Path) -> str:
    return _SHARD_SUFFIX_RE.sub("", path.stem)


def fmt(value: float, precision: int) -> str:
    return f"{value:.{precision}f}"


def percent_change(value: float, baseline: float) -> float:
    if baseline == 0:
        return 0.0
    return (value - baseline) / baseline * 100.0


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    raise SystemExit(main())
