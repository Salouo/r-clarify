import json


def calculate_accuracy(data: dict) -> float:
    total = data["n_processed"]
    
    num_correct = 0 # 累计正确样本数

    for res in data["results"]:
        if res["correct"] == True:
            num_correct += 1

    return num_correct / total


def calculate_average_steps_per_sample(data: dict) -> float:
    total = data["n_processed"]
    
    total_steps = 0

    for res in data["results"]:
        total_steps += res["steps_total"]

    return total_steps / total


def main():
    with open("results/cot.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    acc = calculate_accuracy(data=data)
    avg_steps = calculate_average_steps_per_sample(data=data)

    print(f"Accuracy: {acc:.5f}")
    print(f"Steps per sample: {avg_steps:.5f}")


if __name__ == "__main__":
    main()