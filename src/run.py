import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List

from tqdm import tqdm

from .reflection import run_trials_for_one_sample


def _run_for_one_sample(
    sample: dict,
    max_trials: int,
    lang: str,
    use_reflection: bool,
) -> List[dict]:
    """
    在一个 sample 上跑指定模式（baseline / reflect）。
    返回：该 sample 的多条 trial 的 final_state 列表。
    """
    return run_trials_for_one_sample(
        sample,
        max_trials=max_trials,
        use_reflection=use_reflection,
        lang=lang,
    )


def main(lang: str = "ja", mode: str = "baseline") -> None:
    """
    按照不同 mode 跑数据集
    - mode = "baseline"  -> use_reflection=False，结果写入 sample_metrics_baseline.json
    - mode = "reflect"   -> use_reflection=True，结果写入 sample_metrics_reflect.json
    """
    assert mode in {"baseline", "reflect"}, "mode 必须是 'baseline' 或 'reflect'"
    use_reflection = mode == "reflect"

    # 读取样本
    data_path = Path("data/processed_data_expanded.json")
    with data_path.open("r", encoding="utf-8") as f:
        samples = json.load(f)["samples"]

    # 每个样本最多跑 3 个 trials。
    max_trials = 3

    # 结果日志文件（按 mode 分开保存，便于断点续跑和单独做评估）
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = results_dir / f"sample_metrics_{mode}.json"

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

    # 过滤掉已经完成的样本，只对剩余样本继续跑
    remaining_samples = [s for s in samples if s.get("index") not in done_indices]

    if not remaining_samples:
        print(f"所有样本在模式 {mode} 下都已经完成。")
        return

    print(f"[Progress] 本次运行将处理样本数: {len(remaining_samples)}")

    # 使用线程池并行处理不同的 sample（简单并行版）
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_sample = {
            executor.submit(
                _run_for_one_sample,
                sample,
                max_trials,
                lang,
                use_reflection,
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

            if not states:
                num_trials = 0
                correct = False
                steps_total = 0
                steps_per_trial: List[int] = []
            else:
                num_trials = len(states)
                steps_per_trial = [st.get("step", 0) for st in states]
                steps_total = sum(steps_per_trial)

                last_state = states[-1]
                pred = last_state.get("action_id")
                label = last_state.get("label_action_id")
                correct = (pred == label)

            records.append(
                {
                    "sample_index": sample_index,
                    "lang": lang,
                    "max_trials": max_trials,
                    "num_trials": num_trials,
                    "correct": correct,
                    "steps_total": steps_total,
                    "steps_per_trial": steps_per_trial,
                }
            )

            processed += 1

            # 每完成若干个样本（例如 20 个）就排序并写回一次，避免频繁写盘
            if processed % 20 == 0:
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

    # 循环结束后再做一次最终排序和落盘，确保把不足 20 个的尾巴也写进去
    records.sort(
        key=lambda r: (r.get("sample_index") is None, r.get("sample_index", -1))
    )
    payload = {
        "n_processed": len(records),
        "results": records,
    }
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # 这里可以切换 "ja" / "en"，以及 "baseline" / "reflect"
    for mode in ["baseline", "reflect"]:
        main(lang="ja", mode=mode)
