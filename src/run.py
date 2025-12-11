import json
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List

from tqdm import tqdm

from .reflection import run_trials_for_one_sample
from .state import action_to_id, id_to_action
from .token_usage import global_token_stats
from .utils import extract_trial_steps
from .llm_factory import get_primary_model_dirname


def _run_for_one_sample(
    sample: dict,
    use_reflection: bool,
    mode: str,
    run_id: str | None,
) -> List[dict]:
    """
    在一个 sample 上跑指定模式（baseline / reflect）。
    返回：该 sample 的单条 trial 的 final_state 列表。
    """
    return run_trials_for_one_sample(
        sample,
        use_reflection=use_reflection,
        mode=mode,
        run_id=run_id,
    )


def main(mode: str, start: int | None = None, end: int | None = None) -> None:
    """
    按照不同 mode 跑数据集
    - mode = "baseline"  -> ReAct，无 reflection
    - mode = "reflect"   -> ReAct + reflection
    - mode = "cot"       -> 纯 CoT，一步到位（不走 clarify）
    - mode = "direct"    -> 完全ダイレクト：thinking もほぼ書かず，即座に行動を選ぶ
    """
    assert mode in {
        "baseline",
        "reflect",
        "cot",
        "direct",
    }, "mode 必须是 'baseline' / 'reflect' / 'cot' / 'direct'"

    # 按模式选择是否启用澄清后的反思整理
    if mode in {"cot", "direct"}:
        use_reflection = False
    elif mode == "baseline":
        use_reflection = False
    elif mode == "reflect":
        use_reflection = True

    # 最大的 trial 轮数
    max_trials = 1

    # Load sample
    data_path = Path("data/processed_data_expanded.json")
    with data_path.open("r", encoding="utf-8") as f:
        # 默认跑全量数据；可通过 start/end 灵活截取子集
        samples = json.load(f)["samples"]
    if start is not None or end is not None:
        samples = samples[start:end]

    # 结果日志文件（按模型 + mode 分开保存，便于断点续跑和单独做评估）
    model_dir = get_primary_model_dirname()
    results_dir = Path("outputs") / model_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = results_dir / f"{mode}.json"

    # 读取已经完成的样本结果（如果有）
    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            records: List[dict] = data.get("results", [])
    else:
        records = []

    done_indices = {
        rec.get("sample_index")
        for rec in records
        if rec.get("sample_index") is not None
    }

    print(
        f"[Progress] 模式: {mode}, 已完成样本数: {len(done_indices)}"
    )

    # 过滤掉已经完成的样本，只对剩余样本继续跑，而且还能补齐漏了的样本。
    remaining_samples = [s for s in samples if s.get("index") not in done_indices]

    if not remaining_samples:
        print(f"所有样本在模式 {mode} 下都已经完成。")
        return

    print(f"[Progress] 本次运行将处理样本数: {len(remaining_samples)}")

    # 不使用时间戳/运行 ID，histories 直接落在 histories/<mode>
    run_id = None

    # 使用线程池并行处理不同的 sample（简单并行版）
    with ThreadPoolExecutor(max_workers=1) as executor:
        future_to_sample = {
            executor.submit(
                _run_for_one_sample,
                sample,
                use_reflection,
                mode,
                run_id,
            ): sample
            for sample in remaining_samples
        }

        processed = 0

        for future in tqdm(
            as_completed(future_to_sample),
            total=len(remaining_samples),
            desc="Samples",
            unit="sample",
        ):
            states = future.result()
            sample = future_to_sample[future]
            sample_index = sample.get("index")
            label_action_id = sample.get("label_action_id")
            if label_action_id is None:
                reflective_action = sample.get("reflective_action")
                if reflective_action:
                    label_action_id = action_to_id.get(reflective_action)
            gold_action_desc = sample.get("reflective_action")

            if not states:
                num_trials = 0
                correct = False
                steps_total = 0
                steps_per_trial: List[int] = []
                steps_detail_per_trial: List[List[dict]] = []
            else:
                num_trials = len(states)
                steps_per_trial = [st.get("step", 0) for st in states]
                steps_total = sum(steps_per_trial)
                steps_detail_per_trial = [extract_trial_steps(st) for st in states]

                last_state = states[-1]
                pred = last_state.get("action_id")
                label_action_id = last_state.get("label_action_id") or label_action_id
                correct = (pred == label_action_id)

            if not gold_action_desc and label_action_id is not None:
                gold_action_desc = id_to_action.get(label_action_id)
            if not states:
                steps_detail_per_trial = []

            records.append(
                {
                    "sample_index": sample_index,
                    "max_trials": max_trials,
                    "num_trials": num_trials,
                    "correct": correct,
                    "steps_total": steps_total,
                    "steps_per_trial": steps_per_trial,
                    "steps_detail_per_trial": steps_detail_per_trial,
                    "gold_action": gold_action_desc,
                }
            )

            processed += 1

            # 每完成若干个样本（例如 20 个）就排序并写回一次，避免频繁写盘
            if processed % 10 == 0:
                records.sort(
                    key=lambda r: (
                        r.get("sample_index") is None,
                        r.get("sample_index", -1),
                    )
                )
                payload = {
                    "n_processed": len(records),
                    "results": records,
                }
                with metrics_path.open("w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)

    # 循环结束后再做一次最终排序和落盘，把剩余的的样本也保存
    records.sort(
        key=lambda r: (r.get("sample_index") is None, r.get("sample_index", -1))
    )
    payload = {
        "n_processed": len(records),
        "results": records,
    }
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # 额外写一份本次运行的 token 使用统计
    tokens_path = results_dir / f"tokens_{mode}.json"
    tokens_payload = {
        "run_id": run_id,
        "mode": mode,
        "input_tokens": global_token_stats.input_tokens,
        "output_tokens": global_token_stats.output_tokens,
        "total_tokens": global_token_stats.total_tokens,
    }

    # 追加写：历史记录存在则加载为列表，没有则创建新的列表
    tokens_records = []
    if tokens_path.exists():
        try:
            existing = json.loads(tokens_path.read_text(encoding="utf-8"))
            if isinstance(existing, list):
                tokens_records = existing
            elif isinstance(existing, dict):
                # 兼容旧格式的单条记录
                tokens_records = [existing]
        except Exception:
            # 如果旧文件损坏或格式异常，保留原文件同时重新开始记录
            tokens_records = []

    tokens_records.append(tokens_payload)
    tokens_path.write_text(
        json.dumps(tokens_records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run dataset in specified mode.")
    parser.add_argument(
        "--mode",
        choices=["baseline", "reflect", "cot", "direct"],
        help="Select running mode.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help="样本起始 index（含）。默认从头开始。",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="样本结束 index（不含）。默认到最后一个样本。",
    )
    args = parser.parse_args()
    main(mode=args.mode, start=args.start, end=args.end)
