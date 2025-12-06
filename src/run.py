import json
from pathlib import Path

from tqdm.auto import tqdm

from .reflection import run_trials_for_one_sample


def main(max_trials: int = 3) -> None:
    data_path = Path("data/processed_data_expanded.json")
    with data_path.open("r", encoding="utf-8") as f:
        samples = json.load(f)["samples"][112:118]

    for sample in tqdm(samples, desc="Samples", unit="sample"):
        run_trials_for_one_sample(sample, max_trials=max_trials, use_reflection=True)

        # 对照组（无反思）
        run_trials_for_one_sample(sample, max_trials=max_trials, use_reflection=False)

if __name__ == "__main__":
    main()
