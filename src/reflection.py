from pathlib import Path
from typing import List, Optional

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from .state import AgentState, action_to_id
from .retry_utils import invoke_with_retry
from .utils import (
    make_system_prompt_from_one_sample,
    format_single_trial_episode,
)
from .prompts import reflection_system_prompt
from .llm_factory import get_primary_llm, get_primary_model_dirname

# 专门给 reflection 用的 LLM（和 agent/clarify 用的可以是同一个模型）
reflection_llm = get_primary_llm()
model_dirname = get_primary_model_dirname()


def summarize_clarification(state: AgentState) -> Optional[str]:
    """
    Extract the latest clarify question/answer pair and summarize the new info.
    """
    messages = state.messages or []

    # Last human reply (simulated user answer)
    last_human_idx: Optional[int] = None
    for idx in range(len(messages) - 1, -1, -1):
        if isinstance(messages[idx], HumanMessage):
            last_human_idx = idx
            break
    if last_human_idx is None:
        return None

    reply_msg = messages[last_human_idx]
    question_text = ""

    # Find the most recent AI question before that reply
    for idx in range(last_human_idx - 1, -1, -1):
        msg = messages[idx]
        if isinstance(msg, SystemMessage):
            continue
        if isinstance(msg, AIMessage):
            question_text = msg.content or ""
            break

    if not question_text:
        return None

    system_msg = SystemMessage(content=reflection_system_prompt())
    content = (
        "以下は現在のユーザ状況です：\n"
        f"{state.user_context}\n\n"
        "直近の確認質問とユーザの回答は次の通りです。\n"
        f"Q: {question_text}\n"
        f"A: {reply_msg.content}\n\n"
        "この回答から得られる新しい情報を整理し、次の意思決定に活用できるメモを箇条書きで作成してください。"
    )

    resp = invoke_with_retry(reflection_llm, [system_msg, HumanMessage(content=content)])
    return resp.content


def run_one_trial(
    sample: dict,
    mode: str = "baseline",
    use_reflection: bool = True,
    clarify_quota: int | None = None,
) -> dict:

    # 构建 system prompt：用户场景
    system_prompt = make_system_prompt_from_one_sample(sample)

    # initial_state：第一条 message 是 SystemMessage
    initial_state = AgentState(
        messages=[SystemMessage(content=system_prompt)],
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
) -> List[dict]:

    # trial 固定为 1；保留参数以兼容调用方
    final_state = run_one_trial(
        sample,
        mode=mode,
        use_reflection=use_reflection,
        clarify_quota=clarify_quota,
    )
    all_states: List[dict] = [final_state]

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
        "---------------- ⚙️ Episode ----------------\n",
        format_single_trial_episode(final_state),
        "",
        f"Gold action: {sample.get('reflective_action')}",
    ]
    episode_str = "\n".join(episode_lines)
    (episodes_dir / episode_name).write_text(episode_str, encoding="utf-8")

    return all_states
