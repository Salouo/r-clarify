from pathlib import Path
from typing import List, Optional

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from .state import AgentState, action_to_id, id_to_action
from .retry_utils import invoke_with_retry
from .utils import (
    make_system_prompt_from_one_sample,
    format_single_trial_episode,
    extract_trial_steps,
)
from .llm_factory import get_primary_llm, get_primary_model_dirname

# 专门给 reflection 用的 LLM（和 agent/clarify 用的可以是同一个模型）
reflection_llm = get_primary_llm()
model_dirname = get_primary_model_dirname()


def _trial_reflection_system_prompt() -> str:
    return (
        "You are a reflection helper that writes a short memo for the next trial of an agent.\n"
        "Inputs you receive:\n"
        "- User context (scene)\n"
        "- The latest trial transcript (clarify Q/A and final chosen action)\n"
        "- Whether that trial succeeded (true/false); no gold label is provided.\n"
        "- Actions already attempted in previous trials.\n"
        "- Recent reflection memos (if any).\n"
        "\n"
        "Write a concise bullet list (3-6 bullets) to help the next trial improve:\n"
        "- Key evidence learned from clarify answers or observations.\n"
        "- Likely mistakes or missing info from the last trial.\n"
        "- Concrete next steps: what to ask next / what action to try, and which tried actions to avoid.\n"
        "- Do not mention any hidden gold label or external knowledge; reason only from provided info.\n"
        "- Keep it terse; no preface or closing sentences.\n"
    )


def _format_steps_for_reflection(final_state: dict) -> str:
    steps = extract_trial_steps(final_state)
    lines = []
    for st in steps:
        if st.get("action") == "clarify":
            q = st.get("question") or ""
            a = st.get("user_reply") or ""
            lines.append(f"- Clarify Q: {q}")
            lines.append(f"  User A: {a}")
        else:
            aid = st.get("chosen_action")
            lines.append(f"- Execute action_id={aid} ({id_to_action.get(aid, 'Unknown')})")
    return "\n".join(lines)


def generate_trial_reflection_memo(
    *,
    trial_index: int,
    final_state: dict,
    success: bool,
    tried_actions: list[int],
    recent_memos: list[str],
) -> Optional[str]:
    """
    生成一段供下一轮 trial 使用的反思 memo，不暴露金标签，只说明成功/失败。
    """
    user_ctx = final_state.get("user_context") or ""
    chosen_action = final_state.get("action_id")
    chosen_desc = id_to_action.get(chosen_action, f"Unknown action_id={chosen_action}")

    tried_action_descs = [
        f"{aid}: {id_to_action.get(aid, 'Unknown')}" for aid in tried_actions
    ]
    steps_text = _format_steps_for_reflection(final_state)
    prior_memos_text = "\n".join(f"- {m}" for m in recent_memos) if recent_memos else "None"

    human_prompt = (
        f"Trial #{trial_index} summary\n"
        f"- User context:\n{user_ctx}\n"
        f"- Trial success: {'true' if success else 'false'}\n"
        f"- Chosen action: {chosen_desc} (id={chosen_action})\n"
        f"- Steps:\n{steps_text}\n"
        f"- Tried actions so far: {', '.join(tried_action_descs) if tried_action_descs else 'None'}\n"
        f"- Recent reflection memos (latest first):\n{prior_memos_text}\n"
        "\n"
        "Write a memo for the next trial."
    )

    system_msg = SystemMessage(content=_trial_reflection_system_prompt())
    resp = invoke_with_retry(reflection_llm, [system_msg, HumanMessage(content=human_prompt)])
    return resp.content if resp and resp.content else None


def run_one_trial(
    sample: dict,
    mode: str = "baseline",
    use_reflection: bool = True,
    clarify_quota: int | None = None,
    extra_system_messages: Optional[list] = None,
) -> dict:

    # 构建 system prompt：用户场景
    system_prompt = make_system_prompt_from_one_sample(sample)

    # initial_state：第一条 message 是 SystemMessage
    system_messages = [SystemMessage(content=system_prompt)]
    if extra_system_messages:
        system_messages.extend(extra_system_messages)

    initial_state = AgentState(
        messages=system_messages,
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
    )

    # 这里导入 graph，避免循环依赖
    from .graph import graph

    # 跑一条轨迹
    final_state = graph.invoke(
        initial_state,
        config={"recursion_limit": 1000},   # 最大在graph中跑1000次
    )
    return final_state


def run_trials_for_one_sample(
    sample: dict,
    use_reflection: bool = True,
    mode: str = "baseline",
    run_id: str | None = None,
    clarify_quota: int | None = None,
    max_trials: int = 1,
    memo_window: int | None = None,
) -> List[dict]:

    reflective_action = sample.get("reflective_action")
    label_action_id = action_to_id.get(reflective_action) if reflective_action else None

    all_states: List[dict] = []
    reflection_memos: List[str] = []
    trial_memos: List[dict] = []
    tried_actions: List[int] = []

    if memo_window is None:
        memo_window = 3 if mode == "reflect_k" else 1

    for trial_idx in range(max_trials):
        extra_messages: List[SystemMessage] = []

        prior_memos: List[str] = []
        if use_reflection and reflection_memos:
            prior_memos = reflection_memos[-memo_window:]
            for idx, memo in enumerate(prior_memos, start=1):
                extra_messages.append(SystemMessage(content=f"【Reflection memo #{idx}】\n{memo}"))

        if tried_actions:
            tried_lines = [f"- {id_to_action.get(aid, 'Unknown')} (id={aid})" for aid in tried_actions]
            extra_messages.append(
                SystemMessage(
                    content="【Tried actions so far（avoid repeating）】\n" + "\n".join(tried_lines)
                )
            )

        final_state = run_one_trial(
            sample,
            mode=mode,
            use_reflection=use_reflection,
            clarify_quota=clarify_quota,
            extra_system_messages=extra_messages,
        )
        all_states.append(final_state)

        pred = final_state.get("action_id")
        if isinstance(pred, int):
            tried_actions.append(pred)

        success = bool(label_action_id is not None and pred == label_action_id)

        if use_reflection:
            memo = generate_trial_reflection_memo(
                trial_index=trial_idx + 1,
                final_state=final_state,
                success=success,
                tried_actions=tried_actions,
                recent_memos=reflection_memos[-memo_window:],
            )
            if memo:
                reflection_memos.append(memo)
                trial_memos.append(
                    {
                        "trial": trial_idx + 1,
                        "prior_memos": prior_memos,
                        "new_memo": memo,
                    }
                )
                if len(reflection_memos) > memo_window:
                    reflection_memos = reflection_memos[-memo_window:]

        if success:
            break

    # episodes 目录：<model>/histories/<mode> 或 <model>/histories/<mode>_<run_id>
    base_dir = Path("outputs") / model_dirname / "episodes"
    if run_id:
        episodes_dir = base_dir / f"{mode}_{run_id}"
    else:
        episodes_dir = base_dir / mode
    episodes_dir.mkdir(parents=True, exist_ok=True)

    # 生成 episode 文件
    episode_name = f"sample{sample['index']}.txt"
    episode_lines = [
        "================ 📖 User Context ================\n",
        make_system_prompt_from_one_sample(sample=sample),
        "",
    ]

    for idx, state in enumerate(all_states, start=1):
        episode_lines.extend(
            [
                f"---------------- ⚙️ Episode (trial {idx}/{max_trials}) ----------------\n",
                format_single_trial_episode(state),
                "",
            ]
        )
        # 在 episode 中写入本轮 trial 使用的旧 memo 和生成的新 memo（如果有）
        for rec in trial_memos:
            if rec.get("trial") == idx:
                prior = rec.get("prior_memos") or []
                new_memo = rec.get("new_memo")
                if prior:
                    episode_lines.append("【Prior reflection memos given to this trial】")
                    for memo_idx, memo_text in enumerate(prior, start=1):
                        episode_lines.append(f"[{memo_idx}] {memo_text}")
                    episode_lines.append("")
                if new_memo:
                    episode_lines.extend(
                        [
                            f"【Reflection memo after trial {idx}】",
                            new_memo,
                            "",
                        ]
                    )

    episode_lines.append(f"Gold action: {sample.get('reflective_action')}")

    episode_str = "\n".join(episode_lines)
    (episodes_dir / episode_name).write_text(episode_str, encoding="utf-8")

    return all_states
