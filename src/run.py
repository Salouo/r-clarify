import json
from pathlib import Path
from tqdm import tqdm
from typing import List

from .reflection import run_trials_for_one_sample
from .metrics import (
    calculate_accuracy,
    calculate_average_steps_per_sample,
)


def main(lang: str = "zh") -> None:
    data_path = Path("data/processed_data_expanded.json")
    with data_path.open("r", encoding="utf-8") as f:
        samples = json.load(f)["samples"][:5]

    # 每个样本最多跑 3 个 trials。
    max_trials = 3

    # 每个元素：一个 sample 上的多条 trial（final state 列表，dict 形式）
    trials_by_sample: List[List[dict]] = []

    for sample in tqdm(samples, desc="Samples", unit="sample"):
        sample_states = run_trials_for_one_sample(
            sample,
            max_trials=max_trials,
            use_reflection=False,
            lang=lang,
        )
        trials_by_sample.append(sample_states)

    # 计算并打印整体 metrics（按 sample 计）
    acc = calculate_accuracy(trials_by_sample)
    avg_steps_per_sample = calculate_average_steps_per_sample(trials_by_sample)

    print(f"Sample-level accuracy: {acc:.3f}")
    print(f"Average steps per sample: {avg_steps_per_sample:.3f}")

if __name__ == "__main__":
    # 这里可以切换 "zh" / "ja"
    main(lang="zh")
