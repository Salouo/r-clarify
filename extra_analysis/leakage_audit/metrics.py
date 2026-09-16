from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from .bootstrap import bootstrap_rate_ci_by_instance
from .models import LABELS, ClarificationRecord


def majority_vote(labels: list[str]) -> str:
    if not labels:
        return "NO_TARGET_SIGNAL"
    counts = Counter(labels)
    return sorted(counts.items(), key=lambda item: (-item[1], LABELS.index(item[0]) if item[0] in LABELS else 999))[0][0]


def final_label_rows(
    records: list[ClarificationRecord],
    judge_rows: list[dict[str, Any]],
    lexical_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    record_by_id = {record.record_id: record for record in records}
    labels_by_id: dict[str, list[str]] = defaultdict(list)
    reasons_by_id: dict[str, list[str]] = defaultdict(list)
    for row in judge_rows:
        rid = row.get("record_id")
        label = row.get("label")
        if rid and label in LABELS:
            labels_by_id[rid].append(label)
            reasons_by_id[rid].append(str(row.get("brief_reason", "")))
    lexical_by_id = {row.get("record_id"): row for row in lexical_rows}

    final_rows: list[dict[str, Any]] = []
    for rid, labels in labels_by_id.items():
        record = record_by_id.get(rid)
        if record is None:
            continue
        counts = Counter(labels)
        final_label = majority_vote(labels)
        raw_agreement = counts[final_label] / len(labels) if labels else 0.0
        lexical = lexical_by_id.get(rid, {})
        final_rows.append(
            {
                **record.metadata(),
                "label": final_label,
                "judge_run_labels": labels,
                "raw_agreement": raw_agreement,
                "brief_reason": reasons_by_id[rid][0] if reasons_by_id[rid] else "",
                "exact_gold_action_phrase_overlap": bool(lexical.get("exact_gold_action_phrase_overlap")),
                "target_object_explicitly_mentioned": bool(lexical.get("target_object_explicitly_mentioned")),
                "action_verb_overlap": bool(lexical.get("action_verb_overlap")),
                "obvious_robot_command": bool(lexical.get("obvious_robot_command")),
            }
        )
    return final_rows


def summarize(
    final_rows: list[dict[str, Any]],
    *,
    bootstrap_samples: int = 10000,
    seed: int = 42,
) -> dict[str, Any]:
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in final_rows:
        by_source[str(row.get("source", "unknown"))].append(row)

    summary: dict[str, Any] = {
        "total_clarification_responses": len(final_rows),
        "sources": {},
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
    }
    for source, rows in sorted(by_source.items()):
        total = len(rows)
        label_counts = Counter(row.get("label") for row in rows)
        label_metrics = {}
        for label in LABELS:
            count = label_counts.get(label, 0)
            pct = count / total if total else 0.0
            ci_low, ci_high = bootstrap_rate_ci_by_instance(
                rows,
                predicate=lambda row, expected=label: row.get("label") == expected,
                samples=bootstrap_samples,
                seed=seed,
            )
            label_metrics[label] = {
                "count": count,
                "percentage": pct,
                "bootstrap_ci_95": [ci_low, ci_high],
            }

        leakage_count = label_counts.get("NEAR_DIRECT_TARGET_EXPOSURE", 0) + label_counts.get("DIRECT_TARGET_LEAKAGE", 0)
        leak_low, leak_high = bootstrap_rate_ci_by_instance(
            rows,
            predicate=lambda row: row.get("label") in {"NEAR_DIRECT_TARGET_EXPOSURE", "DIRECT_TARGET_LEAKAGE"},
            samples=bootstrap_samples,
            seed=seed,
        )

        target_mentions = sum(1 for row in rows if row.get("target_object_explicitly_mentioned"))
        command_like = sum(1 for row in rows if row.get("obvious_robot_command"))
        summary["sources"][source] = {
            "n": total,
            "labels": label_metrics,
            "potential_leakage_rate": {
                "count": leakage_count,
                "percentage": leakage_count / total if total else 0.0,
                "bootstrap_ci_95": [leak_low, leak_high],
            },
            "target_object_mention": {
                "count": target_mentions,
                "percentage": target_mentions / total if total else 0.0,
            },
            "explicit_robot_command": {
                "count": command_like,
                "percentage": command_like / total if total else 0.0,
            },
            "mean_raw_agreement": sum(float(row.get("raw_agreement", 0.0)) for row in rows) / total if total else 0.0,
        }
    return summary


def summary_csv_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source, source_data in summary.get("sources", {}).items():
        for label in LABELS:
            metric = source_data["labels"][label]
            rows.append(
                {
                    "source": source,
                    "metric": label,
                    "count": metric["count"],
                    "percentage": metric["percentage"],
                    "ci_low": metric["bootstrap_ci_95"][0],
                    "ci_high": metric["bootstrap_ci_95"][1],
                }
            )
        for metric_name in ("potential_leakage_rate", "target_object_mention", "explicit_robot_command"):
            metric = source_data[metric_name]
            rows.append(
                {
                    "source": source,
                    "metric": metric_name,
                    "count": metric["count"],
                    "percentage": metric["percentage"],
                    "ci_low": (metric.get("bootstrap_ci_95") or ["", ""])[0],
                    "ci_high": (metric.get("bootstrap_ci_95") or ["", ""])[1],
                }
            )
    return rows


def examples_by_label(final_rows: list[dict[str, Any]], *, per_label: int = 25) -> dict[str, list[dict[str, Any]]]:
    examples: dict[str, list[dict[str, Any]]] = {label: [] for label in LABELS}
    ordered = sorted(final_rows, key=lambda row: (-float(row.get("raw_agreement", 0.0)), str(row.get("record_id", ""))))
    for row in ordered:
        label = row.get("label")
        if label in examples and len(examples[label]) < per_label:
            examples[label].append(row)
    return examples

