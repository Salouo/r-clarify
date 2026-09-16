from __future__ import annotations

import csv
import glob
import json
from pathlib import Path
from typing import Any, Iterable

from .models import ClarificationRecord
from .utils import (
    first_present,
    infer_condition_from_path,
    infer_model_dir_from_path,
    load_dataset_by_index,
    load_json,
    stable_hash,
    to_int,
)


DEFAULT_RESULTS_GLOBS = ("outputs/**/results/*.json",)
DEFAULT_HUMAN_GLOBS = (
    "outputs/**/results/*.json",
    "outputs/human_runs*/results/*.json",
    "outputs/human_runs*/results/*.jsonl",
    "outputs/human_runs*/results/*.csv",
    "outputs/human_runs*.json",
)


def discover_simulator_logs(repo_root: Path, patterns: Iterable[str] | None = None) -> list[Path]:
    found: set[Path] = set()
    pattern_list = list(patterns) if patterns is not None else None
    for pattern in pattern_list or DEFAULT_RESULTS_GLOBS:
        full_pattern = str((repo_root / pattern) if not Path(pattern).is_absolute() else Path(pattern))
        for match in glob.glob(full_pattern, recursive=True):
            path = Path(match)
            if path.is_file() and path.suffix.lower() == ".json" and not _is_human_result_path(path):
                found.add(path.resolve())
    return sorted(found)


def discover_human_logs(repo_root: Path, patterns: Iterable[str] | None = None) -> list[Path]:
    pattern_list = list(patterns) if patterns is not None else None
    if pattern_list == []:
        return []
    found: set[Path] = set()
    for pattern in pattern_list or DEFAULT_HUMAN_GLOBS:
        full_pattern = str((repo_root / pattern) if not Path(pattern).is_absolute() else Path(pattern))
        for match in glob.glob(full_pattern, recursive=True):
            path = Path(match)
            if path.is_file() and path.suffix.lower() in {".json", ".jsonl", ".csv"} and (
                pattern_list is not None or _is_human_result_path(path)
            ):
                found.add(path.resolve())
    return sorted(found)


def parse_simulator_results_file(
    path: Path,
    *,
    dataset_by_index: dict[int, dict[str, Any]] | None = None,
    repo_root: Path | None = None,
) -> list[ClarificationRecord]:
    data = load_json(path)
    results = data.get("results", []) if isinstance(data, dict) else []
    records: list[ClarificationRecord] = []
    condition = str(data.get("mode") or infer_condition_from_path(path)) if isinstance(data, dict) else infer_condition_from_path(path)
    top_user_source = data.get("user_source") if isinstance(data, dict) else None
    model_dir = _infer_model_dir(path, data)
    rel_path = str(path.relative_to(repo_root)) if repo_root and path.is_relative_to(repo_root) else str(path)
    dataset_by_index = dataset_by_index or {}
    top_participant_id = data.get("participant_id") if isinstance(data, dict) else None

    for sample_record in results:
        if not isinstance(sample_record, dict):
            continue
        sample_index = to_int(sample_record.get("sample_index"))
        sample = dataset_by_index.get(sample_index) if sample_index is not None else None
        original_utterance = first_present(sample_record, ("original_utterance", "utterance"))
        if original_utterance is None and sample:
            original_utterance = sample.get("utterance")
        gold_action = first_present(sample_record, ("gold_action", "reflective_action"))
        if gold_action is None and sample:
            gold_action = sample.get("reflective_action")

        emitted_any_from_steps = False
        steps_by_trial = sample_record.get("steps_detail_per_trial") or []
        if isinstance(steps_by_trial, list):
            for trial_idx, trial_steps in enumerate(steps_by_trial, start=1):
                if not isinstance(trial_steps, list):
                    continue
                clarification_step = 0
                for step in trial_steps:
                    if not isinstance(step, dict) or step.get("action") != "clarify":
                        continue
                    clarification_step += 1
                    emitted_any_from_steps = True
                    records.append(
                        _build_record(
                            source=_infer_source(path, sample_record, step, top_user_source),
                            path=rel_path,
                            sample_index=sample_index,
                            instance_id=str(sample_index) if sample_index is not None else None,
                            original_utterance=original_utterance,
                            gold_action=gold_action,
                            question=_question_from_turn(step),
                            response=_response_from_turn(step),
                            condition=sample_record.get("mode") or condition,
                            trial=trial_idx,
                            clarification_step=clarification_step,
                            step_t=to_int(step.get("t")),
                            model_dir=model_dir,
                            participant_id=sample_record.get("participant_id") or top_participant_id or _participant_from_path(path),
                        )
                    )

        if emitted_any_from_steps:
            continue

        turns_by_trial = sample_record.get("clarify_turns_per_trial") or []
        if isinstance(turns_by_trial, list):
            for trial_idx, trial_turns in enumerate(turns_by_trial, start=1):
                if not isinstance(trial_turns, list):
                    continue
                for clarification_step, turn in enumerate(trial_turns, start=1):
                    if not isinstance(turn, dict):
                        continue
                    records.append(
                        _build_record(
                            source=_infer_source(path, sample_record, turn, top_user_source),
                            path=rel_path,
                            sample_index=sample_index,
                            instance_id=str(sample_index) if sample_index is not None else None,
                            original_utterance=original_utterance,
                            gold_action=gold_action,
                            question=_question_from_turn(turn),
                            response=_response_from_turn(turn),
                            condition=sample_record.get("mode") or condition,
                            trial=trial_idx,
                            clarification_step=clarification_step,
                            step_t=to_int(turn.get("t")),
                            model_dir=model_dir,
                            participant_id=sample_record.get("participant_id") or top_participant_id or _participant_from_path(path),
                        )
                    )

    return records


def parse_human_file(path: Path, *, repo_root: Path | None = None) -> list[ClarificationRecord]:
    rows = _load_human_rows(path)
    rel_path = str(path.relative_to(repo_root)) if repo_root and path.is_relative_to(repo_root) else str(path)
    records: list[ClarificationRecord] = []
    for row_idx, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            continue
        question = first_present(row, ("clarification_question", "question", "agent_question"))
        response = first_present(row, ("response", "answer", "user_reply", "human_response", "participant_response"))
        gold_action = first_present(row, ("gold_action", "reflective_action", "target_action"))
        if question is None and response is None:
            continue
        sample_index = to_int(first_present(row, ("sample_index", "index", "instance_index")))
        instance_id = first_present(row, ("instance_id", "sample_id", "item_id"))
        if instance_id is None and sample_index is not None:
            instance_id = str(sample_index)
        records.append(
            _build_record(
                source="human",
                path=rel_path,
                sample_index=sample_index,
                instance_id=str(instance_id) if instance_id is not None else f"{path.stem}:{row_idx}",
                original_utterance=first_present(row, ("original_utterance", "utterance", "user_utterance")),
                gold_action=gold_action,
                question=question,
                response=response,
                condition=first_present(row, ("condition", "mode", "original_condition")),
                trial=to_int(first_present(row, ("trial", "trial_number"))),
                clarification_step=to_int(first_present(row, ("clarification_step", "turn", "turn_index"))),
                step_t=to_int(first_present(row, ("t", "step_t", "step"))),
                participant_id=first_present(row, ("participant_id", "worker_id", "subject_id")),
                model_dir=_infer_human_model_dir_from_path(path),
            )
        )
    return records


def parse_all_records(
    repo_root: Path,
    *,
    results_globs: Iterable[str] | None = None,
    human_globs: Iterable[str] | None = None,
    dataset_path: Path | None = None,
) -> tuple[list[Path], list[Path], list[ClarificationRecord]]:
    dataset_by_index = load_dataset_by_index(dataset_path or repo_root / "data/processed_data_expanded.json")
    simulator_logs = discover_simulator_logs(repo_root, results_globs)
    human_logs = discover_human_logs(repo_root, human_globs)
    records: list[ClarificationRecord] = []
    for path in simulator_logs:
        records.extend(parse_simulator_results_file(path, dataset_by_index=dataset_by_index, repo_root=repo_root))
    for path in human_logs:
        parsed = parse_simulator_results_file(path, dataset_by_index=dataset_by_index, repo_root=repo_root)
        records.extend(parsed if parsed else parse_human_file(path, repo_root=repo_root))
    return simulator_logs, human_logs, records


def schema_mapping() -> list[dict[str, str]]:
    return [
        {"required_field": "instance_id", "simulator_mapping": "results[*].sample_index as a string"},
        {"required_field": "gold_action", "simulator_mapping": "results[*].gold_action, fallback data/processed_data_expanded.json samples[*].reflective_action"},
        {"required_field": "original_utterance", "simulator_mapping": "data/processed_data_expanded.json samples[*].utterance joined by sample_index"},
        {"required_field": "clarification_question", "simulator_mapping": "results[*].steps_detail_per_trial[trial][step].clarification_question or question where action == clarify"},
        {"required_field": "response", "simulator_mapping": "results[*].steps_detail_per_trial[trial][step].user_reply or response where action == clarify"},
        {"required_field": "trial", "simulator_mapping": "1-based index in steps_detail_per_trial"},
        {"required_field": "clarification_step", "simulator_mapping": "1-based clarify step count within the trial"},
        {"required_field": "source", "simulator_mapping": "step/record/top-level user_source; folders starting with human_runs map to human; otherwise simulator"},
        {"required_field": "participant_id", "simulator_mapping": "record/top-level participant_id for human rows, fallback participant directory name when available"},
        {"required_field": "original_condition", "simulator_mapping": "top-level mode if present, otherwise result filename with trailing shard suffix removed"},
    ]


def _build_record(
    *,
    source: str,
    path: str,
    sample_index: int | None,
    instance_id: str | None,
    original_utterance: Any,
    gold_action: Any,
    question: Any,
    response: Any,
    condition: Any,
    trial: int | None,
    clarification_step: int | None,
    step_t: int | None,
    model_dir: str | None,
    participant_id: Any = None,
) -> ClarificationRecord:
    rid = stable_hash(
        [
            source,
            path,
            instance_id,
            sample_index,
            participant_id,
            condition,
            trial,
            clarification_step,
            step_t,
            question,
            response,
            gold_action,
        ]
    )
    return ClarificationRecord(
        record_id=rid,
        instance_id=str(instance_id) if instance_id is not None else rid,
        source=source,
        participant_id=str(participant_id) if participant_id not in (None, "") else None,
        original_condition=str(condition) if condition not in (None, "") else None,
        trial=trial,
        clarification_step=clarification_step,
        step_t=step_t,
        sample_index=sample_index,
        model_dir=model_dir,
        log_path=path,
        original_utterance=str(original_utterance or ""),
        gold_action=str(gold_action or ""),
        clarification_question=str(question or ""),
        response=str(response or ""),
    )


def _load_human_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    if suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    data = load_json(path)
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if isinstance(data, dict):
        for key in ("responses", "records", "rows", "data", "results"):
            value = data.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
    return []


def _is_human_result_path(path: Path) -> bool:
    return any(part.startswith("human_runs") for part in path.parts)


def _participant_from_path(path: Path) -> str | None:
    parts = path.parts
    for idx, part in enumerate(parts):
        if not part.startswith("human_runs"):
            continue
        if idx + 1 < len(parts):
            participant = parts[idx + 1]
            if participant not in {"merged_results", "results"}:
                return participant
    return None


def _infer_source(path: Path, sample_record: dict[str, Any], turn: dict[str, Any], top_user_source: Any) -> str:
    raw = first_present(turn, ("user_source",)) or first_present(sample_record, ("user_source",)) or top_user_source
    raw_text = str(raw or "").lower()
    if raw_text in {"human", "participant", "roleplay", "role_play"} or _is_human_result_path(path):
        return "human"
    return "simulator"


def _infer_model_dir(path: Path, data: Any) -> str | None:
    if isinstance(data, dict):
        user_source = str(data.get("user_source") or "").lower()
        model = data.get("model")
        if user_source == "human" and model:
            return str(model)
    return infer_model_dir_from_path(path)


def _infer_human_model_dir_from_path(path: Path) -> str | None:
    for part in path.parts:
        if part.startswith("human_runs_"):
            model = part.removeprefix("human_runs_")
            return model or None
    return infer_model_dir_from_path(path)


def _question_from_turn(turn: dict[str, Any]) -> Any:
    return first_present(turn, ("clarification_question", "question", "agent_question"))


def _response_from_turn(turn: dict[str, Any]) -> Any:
    return first_present(turn, ("user_reply", "response", "answer", "human_response", "participant_response"))
