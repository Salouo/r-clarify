import json
import copy
from pathlib import Path


def expand_reflective_actions(data):
    """Expand each sample into one independent copy per reflective action."""

    if not isinstance(data, dict):
        raise ValueError("顶层 JSON 结构应该是一个 dict（包含 'samples' 字段）。")

    samples = data.get("samples")
    if not isinstance(samples, list):
        raise ValueError("data['samples'] 必须是一个列表。")

    new_samples = []

    for sample in samples:
        reflective_actions = sample.get("reflective_actions")

        if not reflective_actions:
            continue

        for action in reflective_actions:
            # Copy nested annotations so expanded samples can be edited independently.
            new_sample = copy.deepcopy(sample)

            new_sample["index"] = len(new_samples)

            new_sample["reflective_action"] = action
            del new_sample["reflective_actions"]

            new_samples.append(new_sample)

    # Preserve the count field spelling used by the source dataset.
    data["samples"] = new_samples
    data["n_sampels"] = len(new_samples)

    return data


def main():
    input_path_str = "data/processed_data2.json"
    output_path_str = "data/processed_data_expanded.json"

    input_path = Path(input_path_str)
    output_path = Path(output_path_str)

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    transformed = expand_reflective_actions(data)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(transformed, f, ensure_ascii=False, indent=2)

    print(f"处理完成，结果已保存到: {output_path}")
    print(f"新的样本数 n_sampels = {transformed.get('n_sampels')}")


if __name__ == "__main__":
    main()
