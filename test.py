import json
import numpy as np


with open("outputs/gpt-5-nano/results/reflect.json", "r", encoding="utf-8") as f:
    data = json.load(f)


samples = data["results"]
questions = []
for sample in samples:
    for clarify_turn in sample["clarify_turns_per_trial"]:
        if clarify_turn is None:
            continue
        for item in clarify_turn:
            questions.append((item["question_words"], item["question"]))

# 按 question_words 倒序打印长度和文本
for qlen, qtext in sorted(questions, key=lambda x: x[0], reverse=True):
    print(f"{qlen}\t{qtext}")
