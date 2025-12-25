import json
from pathlib import Path


def calculate_accuracy_by_trial(data: dict, trial: int, k: int | None = None) -> float:
    results = data["results"]
    if k is not None:
        results = results[:k]
    total = len(results)
    if total == 0:
        return 0.0

    num_correct = 0
    for res in results:
        if res.get("correct") and res.get("num_trials", 0) <= trial:
            num_correct += 1

    return num_correct / total



def calculate_average_steps_of_successful_trial(
    data: dict, trial: int, k: int | None = None
) -> float:
    """
    计算：如果最多只允许跑到第 `trial` 次 trial，
    只统计成功样本在成功那一次的 step 平均值（失败样本不计入）。
    """
    results = data["results"]
    if k is not None:
        results = results[:k]

    total_steps = 0
    total_success = 0

    for res in results:
        if not res.get("correct"):
            continue
        num_trials = res.get("num_trials", 0)
        if num_trials <= 0 or num_trials > trial:
            continue
        steps_per_trial = res.get("steps_per_trial", [])
        if len(steps_per_trial) < num_trials:
            continue
        total_steps += steps_per_trial[num_trials - 1]
        total_success += 1

    if total_success == 0:
        return 0.0
    return total_steps / total_success


def calculate_dpass_at_k(
    data: dict, trial: int, alpha: float, k: int | None = None
) -> float:
    """
    DPass@k(alpha):
    - 成功样本按成功那次 trial 的 step 数折扣计分
    - 失败样本记 0
    - 按全样本数归一化
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1).")

    results = data["results"]
    if k is not None:
        results = results[:k]
    total = len(results)
    if total == 0:
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
        steps = steps_per_trial[num_trials - 1]
        total_score += alpha ** (steps - 1)

    return total_score / total


def main():
    metrics_path = Path("outputs/Qwen__Qwen3-Next-80B-A3B-Instruct/results/reflect.json")
    with metrics_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    trial = 3   # 最大允许多少个 trial
    k = None     # 测试多少个样本
    alpha = 0.6
    acc = calculate_accuracy_by_trial(data=data, trial=trial, k=k)
    avg_steps = calculate_average_steps_of_successful_trial(data=data, trial=trial, k=k)
    dpass = calculate_dpass_at_k(data=data, trial=trial, alpha=alpha, k=k)

    if k:  
        print(f"Pass@{trial} (k={k}): {acc:.4f}")
        print(f"DPass@{trial} (k={k}, alpha={alpha}): {dpass:.4f}")
    else:
        print(f"Pass@{trial} (k=full): {acc:.4f}")
        print(f"DPass@{trial} (k=full, alpha={alpha}): {dpass:.4f}")

    print(f"Steps per successful trial: {avg_steps:.4f}")


if __name__ == "__main__":
    main()
