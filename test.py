import argparse
import json
from pathlib import Path


def _sample_has_clarify(res: dict, trial: int) -> bool:
    steps_detail = res.get("steps_detail_per_trial") or []
    if steps_detail:
        for trial_steps in steps_detail[:trial]:
            for step in trial_steps or []:
                if step.get("action") == "clarify":
                    return True
        return False

    clarify_turns = res.get("clarify_turns_per_trial") or []
    if clarify_turns:
        for trial_clarify in clarify_turns[:trial]:
            if trial_clarify:
                return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Count samples that chose clarify within a trial limit."
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=Path(
            "outputs/Qwen3-Next-80B-A3B-Instruct/results/reflect_action_only.json"
        ),
        help="Path to results JSON.",
    )
    parser.add_argument("--trial", type=int, default=3, help="Trial limit.")
    args = parser.parse_args()

    with args.path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", [])
    count = sum(1 for res in results if _sample_has_clarify(res, args.trial))
    print(f"clarify_samples (trial<={args.trial}): {count}")
    print(f"num_samples: {len(results)}")


if __name__ == "__main__":
    main()
