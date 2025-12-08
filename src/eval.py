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
    with open("results/reflect.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    acc = calculate_accuracy(data=data)
    avg_steps = calculate_average_steps_per_sample(data=data)

    print(f"Accuracy: {acc:.4f}")
    print(f"Steps per sample: {avg_steps:.4f}")


if __name__ == "__main__":
    main()
