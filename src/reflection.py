from pathlib import Path
from typing import List, Optional

from langchain_core.messages import SystemMessage, HumanMessage

from .state import AgentState, action_to_id, id_to_action
from .utils import (
    make_system_prompt_from_one_sample,
    format_single_trial_episode,
)
from .llm_factory import get_primary_llm, get_primary_model_dirname
from .prompts import trial_reflection_system_prompt
from .token_usage import TokenUsageCollector, set_usage_collector, clear_usage_collector


# Reflection uses the same model instance as the agent.
reflection_llm = get_primary_llm()
model_dirname = get_primary_model_dirname()

ACTION_ELIMINATION_MODES = {"action_elimination"}


def _format_episode_for_reflection(
    *,
    trial_index: int,
    final_state: dict,
) -> str:
    """Format the user context and trial log for the reflection model."""
    user_ctx = final_state.get("user_context") or ""
    trial_body = format_single_trial_episode(final_state)
    return (
        "================ 📖 User Context ================\n\n"
        f"{user_ctx}\n\n"
        f"---------------- ⚙️ Episode (trial {trial_index}) ----------------\n\n"
        f"{trial_body}"
    )


def generate_trial_reflection_memo(
    *,
    trial_index: int,
    final_state: dict,
    success: bool,
    recent_memos: list[str],
    mode: str,
    usage_collector: TokenUsageCollector | None = None,
) -> Optional[str]:
    """Generate a memo for the next trial using the outcome and prior memos."""
    # Action elimination carries forward only the IDs of failed actions.
    if mode in ACTION_ELIMINATION_MODES:
        return None

    episode_text = _format_episode_for_reflection(trial_index=trial_index, final_state=final_state)
    prior_memos_text = "\n".join(f"- {m}" for m in recent_memos) if recent_memos else "なし"

    if mode == "r-clarify_reflexion":
        memo_instruction = (
            "次のトライアルのために、このトライアルの失敗や学びを3〜6行の箇条書きで新規メモとしてまとめてください。"
            "既存メモを上書きせず、今回の学びを簡潔に追加するイメージで。"
        )
        
    elif mode in {"r-clarify", "r-clarify_non_cot", "cot_reflect", "reflect"}:
        memo_instruction = (
            "次のトライアルのために、直近メモを踏まえて必要なら修正しつつ、3〜6行の箇条書きでまとめてください。"
        )

    human_prompt = (
        f"トライアル {trial_index} の完全ログ:\n"
        f"{episode_text}\n"
        "\n"
        f"- 成否: {'成功' if success else '失敗'}\n"
        f"- 直近の反省メモ（新しい順）:\n{prior_memos_text}\n"
        "\n"
        f"{memo_instruction}"
    )

    system_msg = SystemMessage(content=trial_reflection_system_prompt(mode))
    token = None

    if usage_collector is not None:
        token = set_usage_collector(usage_collector)
    try:
        resp = reflection_llm.invoke([system_msg, HumanMessage(content=human_prompt)])
    finally:
        if token is not None:
            clear_usage_collector(token)

    return resp.content if resp and resp.content else None


def _format_failed_action_list_message(failed_action_ids: list[int]) -> str:
    if failed_action_ids:
        failed_actions_text = "\n".join(
            f"- {action_id}: {id_to_action.get(action_id, f'Unknown action_id={action_id}')}"
            for action_id in failed_action_ids
        )
    else:
        failed_actions_text = "None"
    return (
        "[Failed Action List]\n"
        "The following actions were executed in previous trials and were confirmed to be incorrect:\n"
        f"{failed_actions_text}\n\n"
        "- Do not execute any action in this list again.\n"
        "- The failed action list is the only information carried over from previous trials.\n"
        "- Do not assume or infer any previous clarification answers, reasoning, reflection, or user preferences beyond what is explicitly shown in this list.\n"
        "- If the list is empty, no previous action has been ruled out."
    )


def run_one_trial(
    sample: dict,
    mode: str = "clarify",
    use_reflection: bool = True,
    clarify_quota: int | None = None,
    extra_system_messages: Optional[list] = None,
    failed_action_ids: Optional[list[int]] = None,
) -> dict:
    """Run one trial with the sample context and any evidence from earlier trials."""
    system_prompt = make_system_prompt_from_one_sample(sample)

    new_system_messages = [SystemMessage(content=system_prompt)] + (extra_system_messages or [])

    initial_state = AgentState(
        messages=new_system_messages,
        next_decision=None,
        action_id=None,
        env=sample["annotations"],
        user_feedback=None,
        label_action_id=action_to_id[sample["reflective_action"]],
        original_utterance=sample["utterance"],
        mode=mode,
        enable_reflection=use_reflection,
        user_context=system_prompt,
        clarify_quota_total=clarify_quota,
        clarify_quota_left=clarify_quota,
        failed_action_ids=list(failed_action_ids or []),
    )

    # Import the graph here to defer its initialization until a trial starts.
    from .graph import graph

    return graph.invoke(
        initial_state,
        config={"recursion_limit": 1000},
    )


def run_trials_for_one_sample(
    sample: dict,
    use_reflection: bool = True,
    mode: str = "clarify",
    run_id: str | None = None,
    clarify_quota: int | None = None,
    max_trials: int = 1,
    memo_window: int | None = None,
) -> List[dict]:
    """Run trials until success or the trial limit and write the episode log."""
    reflective_action = sample.get("reflective_action")
    label_action_id = action_to_id.get(reflective_action) if reflective_action else None

    all_states: List[dict] = []
    reflection_memos: List[str] = []
    failed_action_ids: List[int] = []
    trial_memos: List[dict] = []
    trial_contexts: List[dict] = []

    reflection_usage = TokenUsageCollector()

    if memo_window is None:
        memo_window = 0 if mode in ACTION_ELIMINATION_MODES else (3 if mode == "r-clarify_reflexion" else 1)

    for trial_idx in range(max_trials):
        extra_messages: List[SystemMessage] = []
        failed_action_ids_before_trial = list(failed_action_ids)

        prior_memos: List[str] = []
        if mode in ACTION_ELIMINATION_MODES:
            extra_messages.append(SystemMessage(content=_format_failed_action_list_message(failed_action_ids_before_trial)))
        elif use_reflection and reflection_memos:
            prior_memos = reflection_memos[-memo_window:]
            for idx, memo in enumerate(prior_memos, start=1):
                if mode in {"r-clarify", "r-clarify_non_cot", "cot_reflect", "reflect"}:
                    extra_messages.append(SystemMessage(content=f"【Reflection memo】\n{memo}"))
                # Label each memo with its distance from the current trial.
                else:
                    distance = len(prior_memos) - idx + 1
                    label = f"{distance} trial前"
                    extra_messages.append(SystemMessage(content=f"【Reflection memo（{label}）】\n{memo}"))

        final_state = run_one_trial(
            sample,
            mode=mode,
            use_reflection=use_reflection,
            clarify_quota=clarify_quota,
            extra_system_messages=extra_messages,
            failed_action_ids=failed_action_ids_before_trial,
        )
        final_state["trial_index"] = trial_idx + 1
        final_state["failed_action_ids_before_trial"] = failed_action_ids_before_trial
        final_state["failed_actions_before_trial"] = [
            id_to_action.get(action_id, f"Unknown action_id={action_id}")
            for action_id in failed_action_ids_before_trial
        ]
        all_states.append(final_state)

        # Keep the memos each trial actually received for the episode log.
        trial_contexts.append(
            {
                "prior_memos": prior_memos if use_reflection else [],
            }
        )

        pred = final_state.get("action_id")
        success = bool(label_action_id is not None and pred == label_action_id)
        final_state["success"] = success

        if (
            mode in ACTION_ELIMINATION_MODES
            and not success
            and isinstance(pred, int)
            and pred in id_to_action
            and pred != -1
            and pred not in failed_action_ids
        ):
            failed_action_ids.append(pred)

        is_last_trial = (trial_idx + 1) >= max_trials

        if use_reflection and (not success) and (not is_last_trial):
            usage_before = reflection_usage.snapshot()

            memo = generate_trial_reflection_memo(
                trial_index=trial_idx + 1,
                final_state=final_state,
                success=success,
                recent_memos=reflection_memos[-memo_window:],
                mode=mode,
                usage_collector=reflection_usage,
            )
            usage_after = reflection_usage.snapshot()
            delta = TokenUsageCollector.diff(usage_after, usage_before)
            if memo:
                reflection_memos.append(memo)
                trial_memos.append(
                    {
                        "trial": trial_idx + 1,
                        "new_memo": memo,
                    }
                )
                if len(reflection_memos) > memo_window:
                    reflection_memos = reflection_memos[-memo_window:]
                try:
                    all_states[-1]["reflection_token_usage"] = delta
                except Exception:
                    pass

        if success:
            break

    episodes_base = Path("outputs") / model_dirname
    if run_id:
        episodes_dir = episodes_base / "episodes" / f"{mode}_{run_id}"
    else:
        episodes_dir = episodes_base / "episodes" / mode
    episodes_dir.mkdir(parents=True, exist_ok=True)

    episode_name = f"sample{sample['index']}.txt"
    episode_lines = [
        "================ 📖 User Context ================\n",
        make_system_prompt_from_one_sample(sample=sample),
        "",
    ]

    for idx, state in enumerate(all_states, start=1):
        context = trial_contexts[idx - 1] if idx - 1 < len(trial_contexts) else {}
        prior_memos = context.get("prior_memos") or []
        memo_for_trial = None
        for rec in trial_memos:
            if rec.get("trial") == idx:
                memo_for_trial = rec.get("new_memo")
                break
        episode_lines.extend(
            [
                f"---------------- ⚙️ Episode (trial {idx}/{max_trials}) ----------------\n",
                format_single_trial_episode(
                    state,
                    prior_memos=prior_memos,
                ),
            ]
        )
        if memo_for_trial:
            episode_lines.extend(
                [
                    f"\n【Reflection memo after trial {idx}】",
                    memo_for_trial,
                ]
            )
        episode_lines.append("")

    # Append the gold action after all trial logs have been assembled.
    episode_lines.append(f"Gold action: {sample.get('reflective_action')}")

    episode_str = "\n".join(episode_lines)
    (episodes_dir / episode_name).write_text(episode_str, encoding="utf-8")

    return all_states
