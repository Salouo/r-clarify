from __future__ import annotations

import argparse
import random
import re
from pathlib import Path
from typing import Any

from .lexical_audit import audit_records
from .metrics import examples_by_label, final_label_rows, summarize, summary_csv_rows
from .models import LABELS, ClarificationRecord
from .parsers import parse_all_records, schema_mapping
from .prompts import judge_prompt_preview
from .utils import write_csv, write_json, write_jsonl


DEFAULT_OUTPUT_DIR = Path("extra_analysis/leakage_audit/results")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit clarification responses for post-hoc label leakage.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="Inspect logs and print the planned judge prompt without LLM calls.")
    _add_common_args(inspect_parser)
    _add_judge_args(inspect_parser)

    pilot_parser = subparsers.add_parser("pilot", help="Run a small pilot only.")
    _add_common_args(pilot_parser)
    _add_judge_args(pilot_parser)
    pilot_parser.add_argument("--pilot-size", type=int, default=10)
    pilot_parser.add_argument("--dry-run", action="store_true", help="Print sampled inputs without calling the judge.")

    run_parser = subparsers.add_parser("run", help="Run the full audit after explicit approval.")
    _add_common_args(run_parser)
    _add_judge_args(run_parser)
    run_parser.add_argument("--approve-full-run", action="store_true", help="Required guard for full LLM judging.")

    args = parser.parse_args()
    if args.command == "inspect":
        return cmd_inspect(args)
    if args.command == "pilot":
        return cmd_pilot(args)
    if args.command == "run":
        return cmd_run(args)
    raise AssertionError(args.command)


def cmd_inspect(args: argparse.Namespace) -> int:
    simulator_logs, human_logs, records = _load_records(args)
    _print_inspection(args, simulator_logs, human_logs, records)
    return 0


def cmd_pilot(args: argparse.Namespace) -> int:
    simulator_logs, human_logs, records = _load_records(args)
    _print_inspection(args, simulator_logs, human_logs, records)
    if not records:
        print("\nNo clarification responses were found; pilot skipped.")
        return 2
    sampled = _sample_records(records, size=args.pilot_size, seed=args.seed)
    print("\nSampled pilot inputs:")
    for record in sampled:
        print(_format_record_for_console(record))
    if args.dry_run:
        print("\nDry run requested; no judge calls made.")
        return 0
    output_dir = _base_output_dir(args, simulator_logs, human_logs, records) / "pilot"
    return _run_judging(sampled, args=args, output_dir=output_dir)


def cmd_run(args: argparse.Namespace) -> int:
    simulator_logs, human_logs, records = _load_records(args)
    _print_inspection(args, simulator_logs, human_logs, records)
    if not args.approve_full_run:
        print("\nFull run blocked: pass --approve-full-run only after reviewing and approving the inspection output.")
        return 2
    if not records:
        print("\nNo clarification responses were found; full audit skipped.")
        return 2
    return _run_judging(
        records,
        args=args,
        output_dir=_base_output_dir(args, simulator_logs, human_logs, records),
    )


def _run_judging(records: list[ClarificationRecord], *, args: argparse.Namespace, output_dir: Path) -> int:
    from .judge import judge_records

    output_dir.mkdir(parents=True, exist_ok=True)
    lexical_rows = audit_records(records)
    write_jsonl(output_dir / "lexical_audit.jsonl", lexical_rows)

    judge_rows = judge_records(
        records,
        model=args.judge_model,
        temperature=args.temperature,
        num_runs=args.num_judge_runs,
        cache_path=Path(args.cache_path),
        max_retries=args.max_retries,
        show_progress=not args.no_progress,
    )
    write_jsonl(output_dir / "audit_labels.jsonl", judge_rows)

    final_rows = final_label_rows(records, judge_rows, lexical_rows)
    write_jsonl(output_dir / "audit_final_labels.jsonl", final_rows)
    summary = summarize(final_rows, bootstrap_samples=args.bootstrap_samples, seed=args.seed)
    write_json(output_dir / "summary.json", summary)
    write_csv(output_dir / "summary.csv", summary_csv_rows(summary))

    example_rows = examples_by_label(final_rows, per_label=args.examples_per_label)
    example_fields = [
        "record_id",
        "source",
        "instance_id",
        "original_condition",
        "trial",
        "clarification_step",
        "original_utterance",
        "clarification_question",
        "response",
        "gold_action",
        "label",
        "brief_reason",
        "raw_agreement",
    ]
    label_to_name = {
        "NO_TARGET_SIGNAL": "examples_no_target_signal.csv",
        "LEGITIMATE_INTENT_SIGNAL": "examples_legitimate_intent.csv",
        "NEAR_DIRECT_TARGET_EXPOSURE": "examples_near_direct_exposure.csv",
        "DIRECT_TARGET_LEAKAGE": "examples_direct_leakage.csv",
    }
    for label in LABELS:
        write_csv(output_dir / label_to_name[label], example_rows[label], fieldnames=example_fields)

    print(f"\nWrote audit outputs under {output_dir}")
    return 0


def _print_inspection(
    args: argparse.Namespace,
    simulator_logs: list[Path],
    human_logs: list[Path],
    records: list[ClarificationRecord],
) -> None:
    source_counts: dict[str, int] = {}
    for record in records:
        source_counts[record.source] = source_counts.get(record.source, 0) + 1

    print("Found simulator log files:")
    if simulator_logs:
        for path in simulator_logs:
            print(f"- {path}")
    else:
        print("- none")

    print("\nFound human response files:")
    if human_logs:
        for path in human_logs:
            print(f"- {path}")
    else:
        print("- none")

    print("\nClarification response counts:")
    print(f"- total: {len(records)}")
    for source in sorted(source_counts):
        print(f"- {source}: {source_counts[source]}")
    print(f"- human role-playing responses available: {'yes' if source_counts.get('human') else 'no'}")

    print("\nSchema mapping:")
    for row in schema_mapping():
        print(f"- {row['required_field']}: {row['simulator_mapping']}")

    estimated_calls = len(records) * args.num_judge_runs
    print("\nJudge-call estimate:")
    print(f"- responses: {len(records)}")
    print(f"- judge runs per response: {args.num_judge_runs}")
    print(f"- estimated judge API calls: {estimated_calls}")
    print("- approximate cost: not computed by default; use current model pricing with the call count above")
    print(f"- planned output directory: {_base_output_dir(args, simulator_logs, human_logs, records)}")

    print("\nExact judge prompt planned:")
    print(judge_prompt_preview())


def _load_records(args: argparse.Namespace) -> tuple[list[Path], list[Path], list[ClarificationRecord]]:
    repo_root = Path(args.repo_root).resolve()
    dataset_path = Path(args.dataset_path)
    if not dataset_path.is_absolute():
        dataset_path = repo_root / dataset_path
    return parse_all_records(
        repo_root,
        results_globs=args.results_glob,
        human_globs=[] if args.no_human else args.human_log_glob,
        dataset_path=dataset_path,
    )


def _sample_records(records: list[ClarificationRecord], *, size: int, seed: int) -> list[ClarificationRecord]:
    if len(records) <= size:
        return list(records)
    rng = random.Random(seed)
    return rng.sample(records, size)


def _format_record_for_console(record: ClarificationRecord) -> str:
    return (
        f"\n[{record.record_id}] source={record.source} instance={record.instance_id} "
        f"trial={record.trial} step={record.clarification_step}\n"
        f"Gold: {record.gold_action}\n"
        f"Utterance: {record.original_utterance}\n"
        f"Q: {record.clarification_question}\n"
        f"A: {record.response}"
    )


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo-root", default=".", help="Repository root.")
    parser.add_argument("--dataset-path", default="data/processed_data_expanded.json")
    parser.add_argument("--results-glob", action="append", default=None, help="Simulator result glob; repeatable.")
    parser.add_argument("--human-log-glob", action="append", default=None, help="Optional human response glob; repeatable.")
    parser.add_argument("--no-human", action="store_true", help="Do not include auto-discovered logs in folders starting with human_runs.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--output-name",
        default=None,
        help="Optional result subdirectory name, e.g. gpt-5/r-clarify. If omitted, a single result file is written under model/mode.",
    )
    parser.add_argument("--no-progress", action="store_true", help="Disable the tqdm progress bar during judge calls.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--examples-per-label", type=int, default=25)


def _add_judge_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--judge-model", default="gpt-5.2")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num-judge-runs", type=int, default=1)
    parser.add_argument("--cache-path", default=str(DEFAULT_OUTPUT_DIR / "judge_cache.jsonl"))
    parser.add_argument("--max-retries", type=int, default=4)


def _base_output_dir(
    args: argparse.Namespace,
    simulator_logs: list[Path],
    human_logs: list[Path],
    records: list[ClarificationRecord],
) -> Path:
    base = Path(args.output_dir)
    if args.output_name:
        return base / _safe_subpath(args.output_name)

    inferred = _infer_output_subpath(simulator_logs, human_logs, records)
    if inferred is not None:
        return base / inferred
    return base


def _infer_output_subpath(
    simulator_logs: list[Path],
    human_logs: list[Path],
    records: list[ClarificationRecord],
) -> Path | None:
    all_logs = simulator_logs + human_logs
    model_dirs = {record.model_dir for record in records if record.model_dir}
    conditions = {record.original_condition for record in records if record.original_condition}

    if len(model_dirs) == 1 and len(conditions) == 1:
        model = next(iter(model_dirs))
        condition = next(iter(conditions))
        source_dirs = {record.source for record in records if record.source}
        if source_dirs == {"human"}:
            return _safe_subpath(f"{model}/human/{condition}")
        return _safe_subpath(f"{model}/{condition}")

    if len(all_logs) == 1 and records:
        log = all_logs[0]
        condition = next(iter(conditions)) if len(conditions) == 1 else log.stem
        parts = log.parts
        for idx, part in enumerate(parts):
            if part.startswith("human_runs"):
                model = next(iter(model_dirs)) if len(model_dirs) == 1 else part
                return _safe_subpath(f"{model}/human/{condition}")
        if "results" in parts:
            idx = parts.index("results")
            model = parts[idx - 1] if idx > 0 else "unknown_model"
            return _safe_subpath(f"{model}/{condition}")

    return None


def _safe_subpath(text: str) -> Path:
    parts = re.split(r"[\\/]+", text.strip())
    safe_parts = [_slugify(part) for part in parts if part.strip()]
    if not safe_parts:
        return Path("run")
    return Path(*safe_parts)


def _slugify(text: str) -> str:
    text = text.strip()
    text = re.sub(r"[^\w.\-]+", "_", text, flags=re.UNICODE)
    text = re.sub(r"_+", "_", text).strip("_.")
    return text or "run"


if __name__ == "__main__":
    raise SystemExit(main())
