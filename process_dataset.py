import json
import copy
from pathlib import Path


def expand_reflective_actions(data):
    """
    把每个 sample 中的 reflective_actions（列表）
    展开为多个 sample，每个只保留一个 reflective_action（单个字符串）。

    顶层结构假定为：
    {
      "n_sampels": ...,
      "samples": [ {...}, {...}, ... ]
    }
    """

    if not isinstance(data, dict):
        raise ValueError("顶层 JSON 结构应该是一个 dict（包含 'samples' 字段）。")

    samples = data.get("samples")
    if not isinstance(samples, list):
        raise ValueError("data['samples'] 必须是一个列表。")

    new_samples = []
    new_index = 0

    for sample in samples:
        # 深拷贝防止互相影响
        reflective_actions = sample.get("reflective_actions")

        # 如果没有 reflective_actions 或者为空，就跳过（你也可以改成保留原样）
        if not reflective_actions:
            continue

        for action in reflective_actions:
            # 拷贝一份原 sample
            new_sample = copy.deepcopy(sample)

            # 改 index
            new_sample["index"] = new_index
            new_index += 1

            # 替换 reflective_actions -> reflective_action
            new_sample["reflective_action"] = action
            if "reflective_actions" in new_sample:
                del new_sample["reflective_actions"]

            new_samples.append(new_sample)

    # 更新顶层信息
    data["samples"] = new_samples
    data["n_sampels"] = len(new_samples)

    return data


def main():
    # 你可以根据自己的路径改这里
    input_path_str = "data/processed_data2.json"
    output_path_str = "data/processed_data_expanded.json"

    input_path = Path(input_path_str)
    output_path = Path(output_path_str)

    # 读取 JSON
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # 处理数据
    transformed = expand_reflective_actions(data)

    # 写回 JSON
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(transformed, f, ensure_ascii=False, indent=2)

    print(f"处理完成，结果已保存到: {output_path}")
    print(f"新的样本数 n_sampels = {transformed.get('n_sampels')}")


if __name__ == "__main__":
    main()