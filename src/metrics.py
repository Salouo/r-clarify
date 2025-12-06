from typing import List


def calculate_accuracy(states: List[dict]) -> float:
    """
    计算给定一组 trial 的 accuracy。
    - 输入：同一个 sample 上多条 trial 的 final_state 列表（run_trials_for_one_sample 的返回值）
    - 输出：在这些 trial 里的平均准确率（命中 label 的 trial 数 / trial 总数）
    """
    total = len(states)
    if total == 0:
        return -1
    
    num_correct = 0

    for s in states:
        pred = s["action_id"]
        label = s["label_action_id"]

        if pred == label:
            num_correct += 1

    return num_correct / total


def calculate_average_steps_per_sample(states: List[dict]) -> int:
    """
    计算一个 sample 在多条 trial 中总共用了多少步（router step 数）。
    - 输入：同一个 sample 上多条 trial 的 final_state 列表
    - 输出：所有 trial 的 step 数之和
    """
    total = len(states)
    if total == 0:
        return -1
    
    total_steps = 0

    for s in states:
        trace = s.get("trace", [])
        # trace 通常是 RouterStep 的列表，这里只关心长度
        if isinstance(trace, list):
            total_steps += len(trace)

    return total_steps / total
