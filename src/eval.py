import json
from pathlib import Path


def calculate_accuracy_by_trial(data: dict, trial: int) -> float:
    total = data.get("n_processed", len(data.get("results", [])))
    if total == 0:
        return 0.0

    num_correct = 0
    for res in data.get("results", []):
        if res.get("correct") and res.get("num_trials", 0) <= trial:
            num_correct += 1

    return num_correct / total



def calculate_average_steps_per_sample_by_trial(data: dict, trial: int) -> float:
    """
    计算：如果最多只允许跑到第 `trial` 次 trial，
    每个样本平均会消耗多少 step（不管对不对）。
    """
    total = data.get("n_processed", len(data.get("results", [])))
    if total == 0:
        return 0.0

    total_steps = 0

    for res in data.get("results", []):
        steps_per_trial = res.get("steps_per_trial", [])
        # 这一样本实际参与的 trial 数（有些模式下 max_trials 会大于 num_trials）
        used_trials = min(trial, len(steps_per_trial))
        total_steps += sum(steps_per_trial[:used_trials])

    return total_steps / total


def main():
    metrics_path = Path("outputs/Qwen__Qwen3-30B-A3B-Instruct-2507/results/reflect_k.json")
    with metrics_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    acc = calculate_accuracy_by_trial(data=data, trial=3)
    avg_steps = calculate_average_steps_per_sample_by_trial(data=data, trial=3)

    print(f"Accuracy: {acc:.4f}")
    print(f"Steps per sample: {avg_steps:.4f}")


if __name__ == "__main__":
    main()
