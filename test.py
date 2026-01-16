#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _sum_token_usage(usage: Dict[str, Any], bucket: Dict[str, int]) -> None:
    prompt = int(usage.get("prompt_tokens", 0))
    completion = int(usage.get("completion_tokens", 0))
    total = int(usage.get("total_tokens", prompt + completion))
    bucket["prompt_tokens"] += prompt
    bucket["completion_tokens"] += completion
    bucket["total_tokens"] += total


def _count_clarify_turns(res: Dict[str, Any], trial_idx: int) -> int:
    turns = res.get("clarify_turns_per_trial")
    if isinstance(turns, list) and trial_idx < len(turns):
        trial_turns = turns[trial_idx] or []
        return len(trial_turns)
    return 0


def compute_success_trial_metrics(path: Path) -> Dict[str, Any]:
    data = _load_json(path)
    results = data.get("results", [])
    total_samples = len(results)
    success_results: List[Dict[str, Any]] = [r for r in results if r.get("correct")]

    success_count = len(success_results)
    token_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    interaction_steps_total = 0
    clarify_turns_total = 0

    for res in success_results:
        num_trials = int(res.get("num_trials", 0))
        if num_trials <= 0:
            continue
        trial_idx = num_trials - 1
        steps_detail = res.get("steps_detail_per_trial", [])
        if trial_idx >= len(steps_detail):
            continue
        trial_steps = steps_detail[trial_idx] or []
        interaction_steps_total += len(trial_steps)
        clarify_turns_total += _count_clarify_turns(res, trial_idx)

        for step in trial_steps:
            usage_agent = step.get("token_usage_agent")
            if usage_agent:
                _sum_token_usage(usage_agent, token_totals)
            usage_reflect = step.get("token_usage_reflect")
            if usage_reflect:
                _sum_token_usage(usage_reflect, token_totals)

    if success_count == 0:
        return {
            "num_samples": total_samples,
            "num_success": 0,
            "success_rate": 0.0,
            "avg_success_trial_tokens": {"prompt_tokens": 0.0, "completion_tokens": 0.0, "total_tokens": 0.0},
            "avg_success_trial_steps": 0.0,
            "avg_success_trial_clarify_turns": 0.0,
        }

    return {
        "num_samples": total_samples,
        "num_success": success_count,
        "success_rate": success_count / total_samples if total_samples else 0.0,
        "avg_success_trial_tokens": {
            "prompt_tokens": token_totals["prompt_tokens"] / success_count,
            "completion_tokens": token_totals["completion_tokens"] / success_count,
            "total_tokens": token_totals["total_tokens"] / success_count,
        },
        "avg_success_trial_steps": interaction_steps_total / success_count,
        "avg_success_trial_clarify_turns": clarify_turns_total / success_count,
    }


def _format_metrics(label: str, metrics: Dict[str, Any]) -> str:
    tokens = metrics["avg_success_trial_tokens"]
    return (
        f"{label}\n"
        f"  samples: {metrics['num_samples']}\n"
        f"  success: {metrics['num_success']} ({metrics['success_rate']:.4f})\n"
        f"  avg_success_trial_tokens: "
        f"prompt={tokens['prompt_tokens']:.2f}, "
        f"completion={tokens['completion_tokens']:.2f}, "
        f"total={tokens['total_tokens']:.2f}\n"
        f"  avg_success_trial_steps: {metrics['avg_success_trial_steps']:.2f}\n"
        f"  avg_success_trial_clarify_turns: {metrics['avg_success_trial_clarify_turns']:.2f}\n"
    )


def _default_paths() -> Tuple[Path, Path]:
    base = Path("outputs/gpt-5-nano/results")
    reflect = base / "reflect.json"
    clarify = base / "clarify.json"
    return reflect, clarify


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare reflect vs clarify on success-trial tokens, success rate, and interactions."
    )
    parser.add_argument("--reflect", type=Path, help="Path to reflect results.json")
    parser.add_argument("--clarify", type=Path, help="Path to clarify results.json")
    args = parser.parse_args()

    if args.reflect is None or args.clarify is None:
        default_reflect, default_clarify = _default_paths()
        if args.reflect is None:
            args.reflect = default_reflect
        if args.clarify is None:
            args.clarify = default_clarify

    if not args.reflect.exists():
        raise SystemExit(f"reflect file not found: {args.reflect}")
    if not args.clarify.exists():
        raise SystemExit(f"clarify file not found: {args.clarify}")

    reflect_metrics = compute_success_trial_metrics(args.reflect)
    clarify_metrics = compute_success_trial_metrics(args.clarify)

    print(_format_metrics("REFLECT", reflect_metrics), end="")
    print(_format_metrics("CLARIFY", clarify_metrics), end="")


if __name__ == "__main__":
    main()
