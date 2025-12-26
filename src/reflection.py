from pathlib import Path
from typing import List, Optional

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from .state import AgentState, action_to_id
from .utils import (
    make_system_prompt_from_one_sample,
    format_single_trial_episode,
)
from .llm_factory import get_primary_llm, get_primary_model_dirname
from .prompts import trial_reflection_system_prompt
from .token_usage import TokenUsageCollector, set_usage_collector, clear_usage_collector


# 专门给 reflection 用的 LLM，规定了 agent 和 relfection 用的都是同一个 LLM
reflection_llm = get_primary_llm()
model_dirname = get_primary_model_dirname()


def _format_episode_for_reflection(
    *,
    trial_index: int,
    final_state: dict,
) -> str:
    """
    将最新トライアルを episode と同じフォーマットでまとめる。
    （ユーザコンテキスト + ステップログ）
    """
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
    """
    生成一段供下一轮 trial 使用的反思 memo，不暴露金标签，只说明成功/失败。
    """
    episode_text = _format_episode_for_reflection(trial_index=trial_index, final_state=final_state)
    prior_memos_text = "\n".join(f"- {m}" for m in recent_memos) if recent_memos else "なし"

    if mode == "reflect_k":
        memo_instruction = (
            "次のトライアルのために、このトライアルの失敗や学びを3〜6行の箇条書きで新規メモとしてまとめてください。"
            "既存メモを上書きせず、今回の学びを簡潔に追加するイメージで。"
        )
    else:
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
        token = set_usage_collector(usage_collector)    # 在此处记录下 token usage
    try:
        resp = reflection_llm.invoke([system_msg, HumanMessage(content=human_prompt)])  # Call the LLM
    finally:
        if token is not None:
            clear_usage_collector(token)

    return resp.content if resp and resp.content else None


def run_one_trial(
    sample: dict,
    mode: str = "clarify",
    use_reflection: bool = True,
    clarify_quota: int | None = None,
    extra_system_messages: Optional[list] = None,
) -> dict:

    # 构建 system prompt：用户场景
    system_prompt = make_system_prompt_from_one_sample(sample)

    # initial_state：第一条 message 是 SystemMessage
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
    )

    # Import `graph`, avoiding 循环依赖
    from .graph import graph

    # Run a single trial.
    final_state = graph.invoke(
        initial_state,
        config={"recursion_limit": 1000},   # 最大在graph中跑1000次
    )
    return final_state


# 重要【Entrance of the Main Process】
def run_trials_for_one_sample(
    sample: dict,
    use_reflection: bool = True,
    mode: str = "clarify",
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
    trial_contexts: List[dict] = []

    reflection_usage = TokenUsageCollector()

    # 只有当mode是reflect_k时，memo的长度才是3
    if memo_window is None:
        memo_window = 3 if mode == "reflect_k" else 1

    # 开始遍历所有 trial
    for trial_idx in range(max_trials):
        extra_messages: List[SystemMessage] = []

        prior_memos: List[str] = []
        if use_reflection and reflection_memos:
            prior_memos = reflection_memos[-memo_window:]   # 取前 k 条反思memo
            for idx, memo in enumerate(prior_memos, start=1):
                # 当mode是 reflect 或者是 cot_reflect，那么此时 reflection memo 的长度固定为1
                if mode in {"reflect", "cot_reflect"}:
                    extra_messages.append(SystemMessage(content=f"【Reflection memo】\n{memo}"))
                # 当 mode 是 reflect_k ，则会告诉 agent，给的memo分别是距离当前 trial第几条的反思memo
                else:
                    distance = len(prior_memos) - idx + 1
                    label = f"{distance} trial前"
                    extra_messages.append(SystemMessage(content=f"【Reflection memo（{label}）】\n{memo}"))

        # 得到一条 trial 运行后的结果（final_state）
        final_state = run_one_trial(
            sample,
            mode=mode,
            use_reflection=use_reflection,
            clarify_quota=clarify_quota,
            extra_system_messages=extra_messages,
        )
        # 保存当前 trial 的运行结果
        all_states.append(final_state)

        # 保存当前 reflection memos，用于 reflect_k 模式中，提示前 k 条 reflection memo
        trial_contexts.append(
            {
                "prior_memos": prior_memos if use_reflection else [],
            }
        )

        # 得到当前 trial 最后执行 Execute 得到的 reflective aciton id
        pred = final_state.get("action_id")
        # 得到是否最终预测成功
        success = bool(label_action_id is not None and pred == label_action_id)

        is_last_trial = (trial_idx + 1) >= max_trials

        if use_reflection and (not success) and (not is_last_trial):
            usage_before = reflection_usage.snapshot()

            # 根据当前 trial 的运行结果，生成反思memo
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
                # 把本轮反思的 token 消耗挂到当前 trial 的最终 state 上
                try:
                    all_states[-1]["reflection_token_usage"] = delta
                except Exception:
                    pass

        # 如果成功了则提前退出所有 trial，不然则继续直到最大的trial。
        if success:
            break

    # 记录 episodes
    # episodes 目录：<model>/histories/<mode> 或 <model>/histories/<mode>_<run_id>
    episodes_base = Path("outputs") / model_dirname
    if run_id:
        episodes_dir = episodes_base / "episodes" / f"{mode}_{run_id}"
    else:
        episodes_dir = episodes_base / "episodes" / mode
    episodes_dir.mkdir(parents=True, exist_ok=True)

    # 生成 episode 文件
    episode_name = f"sample{sample['index']}.txt"
    episode_lines = [
        "================ 📖 User Context ================\n",
        make_system_prompt_from_one_sample(sample=sample),
        "",
    ]

    # 开始记录所有 trial 的运行结果
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

    # 最终保存 episode 的时候才保存 gold action
    episode_lines.append(f"Gold action: {sample.get('reflective_action')}")

    episode_str = "\n".join(episode_lines)
    (episodes_dir / episode_name).write_text(episode_str, encoding="utf-8")

    return all_states
