from typing import List
from langchain_core.messages import HumanMessage, AIMessage
from .state import id_to_action


def make_system_prompt_from_one_sample(sample) -> str:
    has = sample['annotations']['has']
    near_items = sample['annotations']['near_items']
    sofa_front_table_items = sample['annotations']['sofa_front_table_items']
    kitchen_front_table_items = sample['annotations']['kitchen_front_table_items']
    kitchen_items = sample['annotations']['kitchen_items']

    user_prompt = "【ユーザの状況】\n"
    user_prompt += f"ユーザの発話: {sample['utterance']}\n"
    user_prompt += f"ユーザの位置: {sample['annotations']['position']}\n"
    user_prompt += "ユーザが手にしている物: " + (f"{has}" if has else "なし") + "\n"
    user_prompt += "ユーザの近くにある物: " + (f"{near_items}" if near_items else "なし") + "\n"
    user_prompt += "\n"

    user_prompt += "【物品配置情報】\n"
    user_prompt += "ソファ前テーブルの物品: " + (f"{sofa_front_table_items}" if sofa_front_table_items else "なし") + "\n"
    user_prompt += "キッチン前テーブルの物品: " + (f"{kitchen_front_table_items}" if kitchen_front_table_items else "なし") + "\n"
    user_prompt += "キッチンの物品: " + (f"{kitchen_items}" if kitchen_items else "なし") + "\n"
    user_prompt += "\n"

    return user_prompt


def format_single_trial_episode(final_state: dict) -> str:
    """只描述『一个 trial』发生了什么，用于 reflection / 在大 episode 里嵌入。"""
    lines: List[str] = []

    messages = final_state["messages"]
    trace = final_state.get("trace", [])
    label_action_id = final_state.get("label_action_id")
    label_action_desc = id_to_action[label_action_id]
    pred_id = final_state.get("action_id")
    pred_desc = id_to_action[pred_id]

    lines.append("-------- Episode (single trial) --------")

    # 跳过最开始的 system prompt, messages[0]
    msg_idx = 1

    # 如果你想在单个 trial 里也看到“初始用户发言”，可以加上这段（可选）
    if msg_idx < len(messages) and isinstance(messages[msg_idx], HumanMessage):
        lines.append(f"User: {messages[msg_idx].content}")
        msg_idx += 1
        lines.append("")

    # 遍历每一个 router step
    for t in trace:
        lines.append(f"【Step {t.step}】")
        lines.append(f"\tAgent: <thinking> {t.thought} </thinking>")

        if t.next_decision == "clarify":
            lines.append("\tAgent: [CLARIFY]")

            # clarify 之后：AI（澄清问题）+ Human（用户回答）
            if msg_idx < len(messages) and isinstance(messages[msg_idx], AIMessage):
                lines.append(f"\t>> Agent: {messages[msg_idx].content}")
                msg_idx += 1

            if msg_idx < len(messages) and isinstance(messages[msg_idx], HumanMessage):
                lines.append(f"\t>> User: {messages[msg_idx].content}")
                msg_idx += 1

        else:  # execute
            a_desc = id_to_action[t.action_id]
            lines.append(f"\tAgent: [EXECUTE] {a_desc}")

            # execute_node 会追加一条 AIMessage 说明执行情况
            if msg_idx < len(messages) and isinstance(messages[msg_idx], AIMessage):
                lines.append(f"\tAgent: {messages[msg_idx].content}")
                msg_idx += 1

        lines.append("")

    # 单 trial 的结果
    lines.append("---- Trial Result ----")
    lines.append(f"Gold action: {label_action_id} {label_action_desc}")
    lines.append(f"Predicted action: {pred_id} {pred_desc}")
    if pred_id == label_action_id:
        lines.append("Match or not: True ✅")
    else:
        lines.append("Match or not: False ❌")

    return "\n".join(lines)


def format_multi_trial_episode(sample, trials: List[dict]) -> str:
    """
    trials: 每个元素是 {
        "final_state": dict,
        "reflections_before": str | None,
        "reflections_after": str | None,
    }
    """
    lines: List[str] = []

    # ===== 用户信息 / 场景描述（只写一次）=====
    lines.append("======== User Context ========\n")
    lines.append(make_system_prompt_from_one_sample(sample=sample))
    lines.append("")

    # ===== 多个 trial =====
    for i, trial in enumerate(trials, start=1):
        final_state = trial["final_state"]
        reflections_before = trial.get("reflections_before")
        reflections_after = trial.get("reflections_after")

        lines.append(f"======== Trial {i} ========\n")

        # trial 前的 reflection（如果有）
        if reflections_before:
            lines.append("[Reflections BEFORE this trial]")
            lines.append(reflections_before)
            lines.append("")

        # 单 trial 的完整轨迹
        lines.append(format_single_trial_episode(final_state))
        lines.append("")

        # trial 结束后新产生的 reflection（如果有）
        if reflections_after:
            lines.append("[Reflections AFTER this trial]")
            lines.append(reflections_after)
            lines.append("")

        lines.append("")  # trial 间空一行

    return "\n".join(lines)
