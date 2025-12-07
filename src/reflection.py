from pathlib import Path
from datetime import datetime
from typing import List, Any, Optional

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from .state import ClarifyState, action_to_id
from .graph import graph
from .utils import (
    make_system_prompt_from_one_sample,
    format_single_trial_episode,
    format_multi_trial_episode,
)
from .prompts import reflection_system_prompt

# 专门给 reflection 用的 LLM（和 router/clarify 用的可以是同一个模型）
reflection_llm = ChatOpenAI(model="gpt-5-nano")


def run_one_trial(
    sample: dict,
    reflections: Optional[str] = None,
    lang: str = "ja",
) -> dict:

    # 构建 system prompt：用户场景 + 可选反思；没有给下一轮的模型关于之前轮次的对话信息，仅仅提供反思内容。
    system_prompt = make_system_prompt_from_one_sample(sample)
    if reflections:
        system_prompt += "\n\n【過去の試行の振り返り】\n"
        system_prompt += reflections

    # initial_state：第一条 message 是 SystemMessage
    initial_state = ClarifyState(
        messages=[SystemMessage(content=system_prompt)],
        next_decision=None,
        action_id=None,
        env=sample["annotations"],
        user_feedback=None,
        label_action_id=action_to_id[sample["reflective_action"]],
        original_utterance=sample["utterance"],
        lang=lang,
    )

    # 跑一条轨迹
    final_state = graph.invoke(initial_state)  # langgraph 返回的是 dict-like state
    return final_state


def update_reflections(
    prev_reflections: Optional[str],
    sample: dict,
    final_state: dict,
    lang: str = "ja",
) -> str:
    
    # 用户场景（简要）
    user_context = make_system_prompt_from_one_sample(sample)
    # 当前 trial 的内部轨迹（只这一条）
    episode_text = format_single_trial_episode(final_state)

    system_msg = SystemMessage(content=reflection_system_prompt(lang))

    messages: List[Any] = [system_msg]

    if prev_reflections:
        messages.append(
            HumanMessage(
                content=(
                    "以下は、これまでの複数回のトライアルから得られた反省のまとめです。"
                    "必要だと判断した場合は、これを踏まえて内容を更新してください：\n"
                    f"{prev_reflections}"
                )
            )
        )


    # 把「用户场景 + 当前 trial 的 episode」给反思模块
    messages.append(
        HumanMessage(
            content=(
                "以下は現在のサンプルに関するユーザの状況です：\n"
                f"{user_context}\n\n"
                "以下は最新のトライアルの完全なエピソードです。"
                "このエピソードに基づいて、反省内容を更新してください：\n"
                f"{episode_text}"
            )
        )
    )

    resp = reflection_llm.invoke(messages)
    # 直接使用模型输出作为新的 reflections
    return resp.content


def run_trials_for_one_sample(
    sample: dict,
    max_trials: int = 3,
    use_reflection: bool = True,
    lang: str = "ja",
) -> List[dict]:
    
    reflections: Optional[str] = None
    all_states: List[dict] = []
    trials_for_episode: List[dict] = []  # 用来喂给 format_multi_trial_episode

    # episodes 目录下再按 baseline / reflect 分子目录保存
    base_dir = Path("histories")
    episodes_dir = base_dir / ("reflect" if use_reflection else "baseline")
    episodes_dir.mkdir(parents=True, exist_ok=True)

    for _ in range(max_trials):
        # 这一轮 trial 开始前的 reflection 快照
        reflections_before = reflections

        # 跑一条 trial
        final_state = run_one_trial(sample, reflections, lang=lang)
        all_states.append(final_state)

        pred_action_id = final_state.get("action_id")
        label_action_id = final_state.get("label_action_id")
        if label_action_id is None:
            label_action_id = action_to_id[sample["reflective_action"]]
        hit_target = pred_action_id == label_action_id

        # 这一轮结束后的 reflection（命中后就不反思）
        reflections_after: Optional[str] = None
        if use_reflection and not hit_target:
            reflections_after = update_reflections(reflections, sample, final_state, lang=lang)
            reflections = reflections_after  # 用新的反思喂下一轮 trial

        # 把这一轮的信息记录下来，后面拼到「一个大的 episode」里
        trials_for_episode.append(
            {
                "final_state": final_state,
                "reflections_before": reflections_before,
                "reflections_after": reflections_after,
            }
        )

        if hit_target:
            break

    # ===== 所有 trial 跑完之后，生成一个「多 trial 的大 episode」文件 =====
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    episode_name = (
        f"sample{sample['index']}_"
        f"{timestamp}.txt"
    )
    episode_text = format_multi_trial_episode(sample, trials_for_episode)
    (episodes_dir / episode_name).write_text(episode_text, encoding="utf-8")

    return all_states
