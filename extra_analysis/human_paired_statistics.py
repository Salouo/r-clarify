from __future__ import annotations

import argparse
import csv
import itertools
import json
import random
from pathlib import Path
from typing import Any, Iterable

from src.eval import calculate_accuracy_by_trial, calculate_dpass_at_k


DEFAULT_R_CLARIFY = Path("outputs/human_runs_gpt-5/results/r-clarify.json")
DEFAULT_WO_TUR = Path("outputs/human_runs_gpt-5/results/clarify.json")
DEFAULT_OUTPUT_DIR = Path("extra_analysis/human_paired_statistics/results")
METRICS = ("Pass@3", "Pass@5", "DPass@3", "DPass@5", "TDPass@3", "TDPass@5")
EXPECTED_TABLE6 = {
    "r_clarify": {
        "Pass@3": 75.50,
        "Pass@5": 83.00,
        "DPass@3": 67.20,
        "DPass@5": 73.16,
        "TDPass@3": 60.84,
        "TDPass@5": 62.30,
    },
    "wo_tur": {
        "Pass@3": 62.50,
        "Pass@5": 66.50,
        "DPass@3": 57.58,
        "DPass@5": 60.64,
        "TDPass@3": 55.68,
        "TDPass@5": 56.71,
    },
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Paired participant-level statistics for the human experiment."
    )
    parser.add_argument("--r-clarify", type=Path, default=DEFAULT_R_CLARIFY)
    parser.add_argument("--wo-tur", type=Path, default=DEFAULT_WO_TUR)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    r_records, r_paths = load_condition_records(args.r_clarify)
    w_records, w_paths = load_condition_records(args.wo_tur)

    r_by_pair, r_dupes = records_by_pair(r_records)
    w_by_pair, w_dupes = records_by_pair(w_records)
    matched_keys = sorted(set(r_by_pair) & set(w_by_pair))
    unmatched_r = sorted(set(r_by_pair) - set(w_by_pair))
    unmatched_w = sorted(set(w_by_pair) - set(r_by_pair))

    matched_r = [r_by_pair[key] for key in matched_keys]
    matched_w = [w_by_pair[key] for key in matched_keys]
    participants = sorted({participant for participant, _ in matched_keys})

    aggregate_r_raw = compute_metrics(r_records, alpha=args.alpha)
    aggregate_w_raw = compute_metrics(w_records, alpha=args.alpha)
    aggregate_r_deduped = compute_metrics(matched_r, alpha=args.alpha)
    aggregate_w_deduped = compute_metrics(matched_w, alpha=args.alpha)
    participant_rows = build_participant_rows(
        matched_keys=matched_keys,
        r_by_pair=r_by_pair,
        w_by_pair=w_by_pair,
        participants=participants,
        alpha=args.alpha,
    )
    stats_rows = build_statistics_rows(
        aggregate_r=aggregate_r_raw,
        aggregate_w=aggregate_w_raw,
        participant_rows=participant_rows,
        bootstrap_samples=args.bootstrap,
        seed=args.seed,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stats_path = args.output_dir / "human_paired_statistics.csv"
    participant_path = args.output_dir / "human_participant_level_metrics.csv"
    write_csv(stats_path, stats_rows)
    write_csv(participant_path, participant_rows)

    print_pairing_summary(
        r_paths=r_paths,
        w_paths=w_paths,
        r_records=r_records,
        w_records=w_records,
        matched_keys=matched_keys,
        unmatched_r=unmatched_r,
        unmatched_w=unmatched_w,
        r_dupes=r_dupes,
        w_dupes=w_dupes,
        participants=participants,
    )
    print()
    print_table6_check(
        aggregate_r_raw,
        aggregate_w_raw,
        aggregate_r_deduped,
        aggregate_w_deduped,
    )
    print()
    print_markdown_results(stats_rows)
    print()
    print_participant_differences(participant_rows)
    print()
    print_sanity_checks(
        participants=participants,
        unmatched_r=unmatched_r,
        unmatched_w=unmatched_w,
        r_dupes=r_dupes,
        w_dupes=w_dupes,
        bootstrap_samples=args.bootstrap,
    )
    print()
    print_interpretation(stats_rows, participant_rows)
    print()
    print(f"Saved CSV: {stats_path}")
    print(f"Saved CSV: {participant_path}")
    return 0


def load_condition_records(path: Path) -> tuple[list[dict[str, Any]], list[Path]]:
    paths = discover_files(path)
    records: list[dict[str, Any]] = []
    for file_path in paths:
        records.extend(load_records_file(file_path))
    return records, paths


def discover_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.exists():
        raise FileNotFoundError(path)
    suffixes = {".json", ".jsonl", ".csv"}
    return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in suffixes)


def load_records_file(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict) and isinstance(payload.get("results"), list):
            return [normalize_record(row, path) for row in payload["results"] if isinstance(row, dict)]
        if isinstance(payload, list):
            return [normalize_record(row, path) for row in payload if isinstance(row, dict)]
        return []
    if suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    if isinstance(row, dict):
                        rows.append(normalize_record(row, path))
        return rows
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            return [normalize_record(row, path) for row in csv.DictReader(f)]
    return []


def normalize_record(row: dict[str, Any], path: Path) -> dict[str, Any]:
    record = dict(row)
    participant_id = first_present(record, ("participant_id", "worker_id", "subject_id", "run_id"))
    instance_id = first_present(record, ("instance_id", "sample_id", "item_id", "sample_index", "index"))
    if participant_id in (None, ""):
        participant_id = participant_from_path(path)
    if instance_id in (None, ""):
        raise ValueError(f"Missing instance id in {path}")
    if participant_id in (None, ""):
        raise ValueError(f"Missing participant id in {path}")
    record["participant_id"] = str(participant_id)
    record["instance_id"] = str(instance_id)
    record["sample_index"] = to_int(first_present(record, ("sample_index", "index", "instance_id")))
    record["correct"] = to_bool(record.get("correct"))
    record["num_trials"] = to_int(record.get("num_trials")) or 0
    record["steps_per_trial"] = parse_list(record.get("steps_per_trial"))
    return record


def records_by_pair(records: list[dict[str, Any]]) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[tuple[str, str], int]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in records:
        key = (record["participant_id"], record["instance_id"])
        grouped.setdefault(key, []).append(record)
    deduped = {key: rows[-1] for key, rows in grouped.items()}
    dupes = {key: len(rows) for key, rows in grouped.items() if len(rows) > 1}
    return deduped, dupes


def build_participant_rows(
    *,
    matched_keys: list[tuple[str, str]],
    r_by_pair: dict[tuple[str, str], dict[str, Any]],
    w_by_pair: dict[tuple[str, str], dict[str, Any]],
    participants: list[str],
    alpha: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    keys_by_participant = {
        participant: [key for key in matched_keys if key[0] == participant]
        for participant in participants
    }
    for participant in participants:
        r_records = [r_by_pair[key] for key in keys_by_participant[participant]]
        w_records = [w_by_pair[key] for key in keys_by_participant[participant]]
        r_metrics = compute_metrics(r_records, alpha=alpha)
        w_metrics = compute_metrics(w_records, alpha=alpha)
        for metric in METRICS:
            rows.append(
                {
                    "participant_id": participant,
                    "metric": metric,
                    "r_clarify": r_metrics[metric],
                    "wo_tur": w_metrics[metric],
                    "difference": r_metrics[metric] - w_metrics[metric],
                    "n_pairs": len(keys_by_participant[participant]),
                }
            )
    return rows


def build_statistics_rows(
    *,
    aggregate_r: dict[str, float],
    aggregate_w: dict[str, float],
    participant_rows: list[dict[str, Any]],
    bootstrap_samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    by_metric = {
        metric: [row for row in participant_rows if row["metric"] == metric]
        for metric in METRICS
    }
    raw_ps: dict[str, float] = {}
    rows: list[dict[str, Any]] = []
    for metric in METRICS:
        diffs = [float(row["difference"]) for row in by_metric[metric]]
        raw_p = exact_sign_flip_p_value(diffs)
        raw_ps[metric] = raw_p
        ci_low, ci_high = bootstrap_ci(diffs, samples=bootstrap_samples, seed=seed)
        rows.append(
            {
                "metric": metric,
                "r_clarify": aggregate_r[metric] * 100.0,
                "wo_tur": aggregate_w[metric] * 100.0,
                "paired_difference": mean(diffs) * 100.0,
                "ci_lower": ci_low * 100.0,
                "ci_upper": ci_high * 100.0,
                "raw_p": raw_p,
                "holm_p": None,
                "holm_significant": None,
                "n_participants": len(diffs),
            }
        )

    holm = holm_adjust(raw_ps)
    for row in rows:
        metric = row["metric"]
        row["holm_p"] = holm[metric]
        row["holm_significant"] = holm[metric] < 0.05
    return rows


def compute_metrics(records: list[dict[str, Any]], *, alpha: float) -> dict[str, float]:
    data = {"results": records}
    return {
        "Pass@3": calculate_accuracy_by_trial(data, trial=3),
        "Pass@5": calculate_accuracy_by_trial(data, trial=5),
        "DPass@3": calculate_dpass_at_k(data, trial=3, alpha=alpha),
        "DPass@5": calculate_dpass_at_k(data, trial=5, alpha=alpha),
        "TDPass@3": calculate_tdpass_at_k(data, trial=3, alpha=alpha),
        "TDPass@5": calculate_tdpass_at_k(data, trial=5, alpha=alpha),
    }


def calculate_tdpass_at_k(data: dict[str, Any], trial: int, alpha: float) -> float:
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
        total_score += alpha ** (sum(steps_per_trial[:num_trials]) - 1)
    return total_score / len(results)


def bootstrap_ci(diffs: list[float], *, samples: int, seed: int) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(diffs)
    values = []
    for _ in range(samples):
        values.append(mean(rng.choice(diffs) for _ in range(n)))
    values.sort()
    return percentile(values, 2.5), percentile(values, 97.5)


def exact_sign_flip_p_value(diffs: list[float]) -> float:
    observed = abs(mean(diffs))
    if not diffs:
        return 1.0
    extreme = 0
    total = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(diffs)):
        permuted = mean(sign * diff for sign, diff in zip(signs, diffs))
        total += 1
        if abs(permuted) >= observed - 1e-15:
            extreme += 1
    return extreme / total


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running_max = 0.0
    m = len(ordered)
    for rank, (metric, p_value) in enumerate(ordered, start=1):
        candidate = min((m - rank + 1) * p_value, 1.0)
        running_max = max(running_max, candidate)
        adjusted[metric] = running_max
    return adjusted


def print_pairing_summary(
    *,
    r_paths: list[Path],
    w_paths: list[Path],
    r_records: list[dict[str, Any]],
    w_records: list[dict[str, Any]],
    matched_keys: list[tuple[str, str]],
    unmatched_r: list[tuple[str, str]],
    unmatched_w: list[tuple[str, str]],
    r_dupes: dict[tuple[str, str], int],
    w_dupes: dict[tuple[str, str], int],
    participants: list[str],
) -> None:
    counts = participant_counts(matched_keys)
    print("Pairing/data-integrity summary")
    print(f"- R-Clarify files: {', '.join(str(p) for p in r_paths)}")
    print(f"- w/o TUR files: {', '.join(str(p) for p in w_paths)}")
    print(f"- participants: {len(participants)} ({', '.join(participants)})")
    print(f"- R-Clarify participant-instance pairs: {len(records_by_pair(r_records)[0])}")
    print(f"- w/o TUR participant-instance pairs: {len(records_by_pair(w_records)[0])}")
    print(f"- matched participant-instance pairs: {len(matched_keys)}")
    print(f"- unmatched R-Clarify pairs: {len(unmatched_r)}")
    print(f"- unmatched w/o TUR pairs: {len(unmatched_w)}")
    print(f"- duplicate R-Clarify pairs: {len(r_dupes)}")
    print(f"- duplicate w/o TUR pairs: {len(w_dupes)}")
    if r_dupes or w_dupes:
        dup_participants = sorted({key[0] for key in [*r_dupes.keys(), *w_dupes.keys()]})
        print(f"- duplicate participant ids: {', '.join(dup_participants)}")
        print("- duplicate handling for inference: collapse to one record per participant_id + sample_index")
    print("- matched instances per participant: " + ", ".join(f"{p}={counts[p]}" for p in participants))


def print_table6_check(
    r_metrics_raw: dict[str, float],
    w_metrics_raw: dict[str, float],
    r_metrics_deduped: dict[str, float],
    w_metrics_deduped: dict[str, float],
) -> None:
    print("Table 6 reproduction check (percentage points)")
    for label, metrics, expected in (
        ("R-Clarify raw logs", r_metrics_raw, EXPECTED_TABLE6["r_clarify"]),
        ("w/o TUR raw logs", w_metrics_raw, EXPECTED_TABLE6["wo_tur"]),
    ):
        parts = []
        for metric in METRICS:
            observed = metrics[metric] * 100.0
            diff = observed - expected[metric]
            parts.append(f"{metric}: {observed:.2f} (expected {expected[metric]:.2f}, diff {diff:+.2f})")
        print(f"- {label}: " + "; ".join(parts))
    print("- Deduped participant-instance logs used for inference:")
    print(
        "  R-Clarify: "
        + "; ".join(f"{metric}: {r_metrics_deduped[metric] * 100.0:.2f}" for metric in METRICS)
    )
    print(
        "  w/o TUR: "
        + "; ".join(f"{metric}: {w_metrics_deduped[metric] * 100.0:.2f}" for metric in METRICS)
    )


def print_markdown_results(rows: list[dict[str, Any]]) -> None:
    print("| Metric | R-Clarify | w/o TUR | Paired Δ | 95% CI of Δ | Raw p | Holm p |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            "| {metric} | {r:.2f} | {w:.2f} | {d:+.2f} | [{lo:+.2f}, {hi:+.2f}] | {raw} | {holm} |".format(
                metric=row["metric"],
                r=row["r_clarify"],
                w=row["wo_tur"],
                d=row["paired_difference"],
                lo=row["ci_lower"],
                hi=row["ci_upper"],
                raw=format_p(row["raw_p"]),
                holm=format_p(row["holm_p"]),
            )
        )


def print_participant_differences(rows: list[dict[str, Any]]) -> None:
    print("Participant-level paired differences, percentage points")
    participants = sorted({row["participant_id"] for row in rows})
    by_key = {(row["participant_id"], row["metric"]): row for row in rows}
    print("| participant_id | " + " | ".join(METRICS) + " |")
    print("|---|" + "|".join("---:" for _ in METRICS) + "|")
    for participant in participants:
        values = [
            f"{by_key[(participant, metric)]['difference'] * 100.0:+.2f}"
            for metric in METRICS
        ]
        print(f"| {participant} | " + " | ".join(values) + " |")


def print_sanity_checks(
    *,
    participants: list[str],
    unmatched_r: list[tuple[str, str]],
    unmatched_w: list[tuple[str, str]],
    r_dupes: dict[tuple[str, str], int],
    w_dupes: dict[tuple[str, str], int],
    bootstrap_samples: int,
) -> None:
    print("Sanity checks")
    print("- Aggregate metric values are compared against Table 6 above.")
    print(f"- Every included participant has both conditions: {'yes' if participants and not unmatched_r and not unmatched_w else 'check unmatched pairs'}")
    if not r_dupes and not w_dupes:
        print("- No duplicate participant-instance pairs: yes")
    else:
        print("- No participant is counted twice because of duplicate logs: yes, duplicates are collapsed before participant-level inference")
    print("- Participant-level aggregation is used for inference: yes")
    print(f"- Bootstrap resampling is participant-level: yes ({bootstrap_samples} samples)")
    print(f"- Exact sign-flip testing is participant-level: yes (2^{len(participants)} assignments per metric)")
    print("- No LLM/API/model call occurs during this analysis: yes")


def print_interpretation(rows: list[dict[str, Any]], participant_rows: list[dict[str, Any]]) -> None:
    ci_excluding_zero = [row["metric"] for row in rows if row["ci_lower"] > 0 or row["ci_upper"] < 0]
    raw_sig = [row["metric"] for row in rows if row["raw_p"] < 0.05]
    holm_sig = [row["metric"] for row in rows if row["holm_p"] < 0.05]
    consistency = {}
    for metric in METRICS:
        diffs = [row["difference"] for row in participant_rows if row["metric"] == metric]
        consistency[metric] = sum(1 for diff in diffs if diff > 0)

    print("ACL-rebuttal-ready interpretation")
    print(
        "- Paired participant-level improvements were: "
        + "; ".join(f"{row['metric']} {row['paired_difference']:+.2f} pp" for row in rows)
        + "."
    )
    print(
        "- 95% bootstrap CIs excluding zero: "
        + (", ".join(ci_excluding_zero) if ci_excluding_zero else "none")
        + "."
    )
    print(
        "- Exact paired sign-flip p<0.05 before Holm correction: "
        + (", ".join(raw_sig) if raw_sig else "none")
        + "."
    )
    print(
        "- Metrics remaining significant after Holm correction: "
        + (", ".join(holm_sig) if holm_sig else "none")
        + "."
    )
    print(
        "- Positive participant-level differences by metric: "
        + "; ".join(f"{metric} {count}/10" for metric, count in consistency.items())
        + "."
    )
    if len(holm_sig) < len(METRICS):
        print("- Some metrics are not significant after correction, so the evidence should be stated metric-by-metric rather than overgeneralized.")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def participant_counts(keys: Iterable[tuple[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for participant, _ in keys:
        counts[participant] = counts.get(participant, 0) + 1
    return counts


def first_present(row: dict[str, Any], names: tuple[str, ...]) -> Any:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return value
    return None


def participant_from_path(path: Path) -> str | None:
    stem = path.stem
    if "_sample" in stem:
        return stem.split("_sample", 1)[0]
    return None


def to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def to_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            return []
    return []


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * q / 100.0
    lower = int(pos)
    upper = min(lower + 1, len(values) - 1)
    weight = pos - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def format_p(value: float) -> str:
    return f"{value:.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
