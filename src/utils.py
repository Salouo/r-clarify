import re

from typing import Iterable, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
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
    # The inventory lists visible items. The agent can fetch other items elsewhere.
    user_prompt += "※以上の物品リストは「今見えている物」だけです。リストにない物でも他の場所から持ってくることができます。\n"
    user_prompt += "\n"

    return user_prompt


def _skip_system_messages(messages: list[BaseMessage], idx: int) -> int:
    while idx < len(messages) and isinstance(messages[idx], SystemMessage):
        idx += 1
    return idx


def format_single_trial_episode(
    final_state: dict,
    *,
    prior_memos: List[str] | None = None,
) -> str:
    """Format a trial log for reflection or inclusion in an episode.

    Show the memos supplied to the trial before its steps.
    """
    lines: List[str] = []

    messages = final_state["messages"]
    trace = final_state.get("trace", [])
    label_action_id = final_state.get("label_action_id")
    pred_id = final_state.get("action_id")
    pred_desc = id_to_action.get(pred_id, f"Unknown action_id={pred_id}")

    # Skip the initial system prompt.
    msg_idx = 1

    reflection_memos = [
        m.content
        for m in messages
        if isinstance(m, SystemMessage) and str(m.content).startswith("【明確化質問後のメモ】")
    ]

    def _emit_clarify_system(idx: int) -> int:
        """Log clarification notes and return the index of the next dialogue message."""
        while idx < len(messages) and isinstance(messages[idx], SystemMessage):
            content = str(messages[idx].content or "")
            if content.startswith("【Clarify思考】"):
                clean = content.replace("【Clarify思考】", "", 1).strip()
                lines.append(f"\tClarify: <thinking> {clean} </thinking>")
            elif content.startswith("【Agentの明確化意図】"):
                clean = content.replace("【Agentの明確化意図】", "", 1).strip()
                lines.append(f"\tAgent: <clarify-intent> {clean} </clarify-intent>")
            idx += 1
        return idx

    msg_idx = _skip_system_messages(messages, msg_idx)
    if msg_idx < len(messages) and isinstance(messages[msg_idx], HumanMessage):
        lines.append(f"User: {messages[msg_idx].content}")
        msg_idx += 1
        lines.append("")

    memos_to_show = prior_memos or []
    if memos_to_show:
        lines.append("【Prior reflection memos given to this trial】")
        for memo_idx, memo_text in enumerate(memos_to_show, start=1):
            lines.append(f"[{memo_idx}] {memo_text}")
        lines.append("")

    failed_actions = final_state.get("failed_actions_before_trial") or []
    if failed_actions:
        lines.append("[Failed actions given to this trial]")
        for action_text in failed_actions:
            lines.append(f"- {action_text}")
        lines.append("")

    for t in trace:
        lines.append(f"【Step {t.step}】")
        if t.thought is not None:
            lines.append(f"\tAgent: <thinking> {t.thought} </thinking>")

        if t.next_decision == "clarify":
            lines.append("\tAgent: [CLARIFY]")

            msg_idx = _emit_clarify_system(msg_idx)
            if msg_idx < len(messages) and isinstance(messages[msg_idx], AIMessage):
                lines.append(f"\t>> Agent: {messages[msg_idx].content}")
                msg_idx += 1

            msg_idx = _emit_clarify_system(msg_idx)
            if msg_idx < len(messages) and isinstance(messages[msg_idx], HumanMessage):
                lines.append(f"\t>> User: {messages[msg_idx].content}")
                msg_idx += 1

            if reflection_memos:
                memo_text = reflection_memos.pop(0)
                lines.append("\tAgent: [REFLECT]")
                for memo_line in memo_text.splitlines():
                    lines.append(f"\t   {memo_line}")

        else:
            a_desc = id_to_action.get(t.action_id, f"Unknown action_id={t.action_id}")
            lines.append(f"\tAgent: [EXECUTE] {a_desc}")

            # The execution node appends the action description to the dialogue.
            msg_idx = _skip_system_messages(messages, msg_idx)
            if msg_idx < len(messages) and isinstance(messages[msg_idx], AIMessage):
                lines.append(f"\t>> Agent は {messages[msg_idx].content}")
                msg_idx += 1

        lines.append("")

    lines.append("================ 🚀 Trial Result ================")
    lines.append(f"Predicted action {pred_desc}")
    if pred_id == label_action_id:
        lines.append("Outcome: Correct ✅")
    else:
        lines.append("Outcome: Incorrect ❌")

    return "\n".join(lines)


def format_full_messages(history: Iterable[BaseMessage]) -> str:
    """Format the dialogue with a role and sequence number for each message."""
    lines: List[str] = []
    for idx, msg in enumerate(history, start=1):
        role = "System"
        if isinstance(msg, HumanMessage):
            role = "User"
        elif isinstance(msg, AIMessage):
            role = "Agent"
        content = str(getattr(msg, "content", ""))
        lines.append(f"[{idx}] {role}: {content}")
    return "\n".join(lines)


def _count_words(text: str) -> int:
    """Count Latin words and individual CJK ideographs and kana characters.

    An ASCII alphanumeric word can include an internal apostrophe.
    """
    if not text:
        return 0

    pattern = re.compile(
        r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?"
        r"|[\u3040-\u309F\u30A0-\u30FF\u31F0-\u31FF\uFF65-\uFF9F\u4E00-\u9FFFー]+"
    )
    tokens = pattern.findall(text)
    total = 0
    for tok in tokens:
        if re.search(r"[\u3040-\u30FF\u31F0-\u31FF\uFF65-\uFF9F\u4E00-\u9FFFー]", tok):
            total += len(tok)
        else:
            total += 1
    return total


def extract_clarification_questions(final_state: dict) -> List[str]:
    """Return the trial's clarification questions in chronological order."""
    messages = final_state.get("messages") or []
    trace = final_state.get("trace") or []

    questions: List[str] = []
    msg_idx = 1  # Skip the system prompt.

    for step in trace:
        next_decision = getattr(step, "next_decision", None)
        if isinstance(step, dict):
            next_decision = step.get("next_decision")

        if next_decision == "clarify":
            if isinstance(step, dict):
                q_text = step.get("clarification_question") or ""
            else:
                q_text = getattr(step, "clarification_question", "") or ""
            if q_text:
                questions.append(q_text)
            msg_idx = _skip_system_messages(messages, msg_idx)
            if msg_idx < len(messages) and isinstance(messages[msg_idx], AIMessage):
                if not q_text:
                    questions.append(messages[msg_idx].content or "")
                msg_idx += 1
            msg_idx = _skip_system_messages(messages, msg_idx)
            if msg_idx < len(messages) and isinstance(messages[msg_idx], HumanMessage):
                msg_idx += 1
        else:
            msg_idx = _skip_system_messages(messages, msg_idx)
            if msg_idx < len(messages) and isinstance(messages[msg_idx], AIMessage):
                msg_idx += 1

    return questions


def extract_clarification_word_counts(final_state: dict) -> List[int]:
    """Return a word count for each clarification question in trial order."""
    questions = extract_clarification_questions(final_state)
    return [_count_words(q) for q in questions]


def extract_trial_steps(final_state: dict) -> List[dict]:
    """Serialize trial steps and their token usage for the results file."""
    messages = final_state.get("messages") or []
    trace = final_state.get("trace") or []
    reflection_usage = final_state.get("reflection_token_usage")

    steps: List[dict] = []
    msg_idx = 1  # Skip the system prompt.

    for step_idx, step in enumerate(trace):
        t = getattr(step, "step", None)
        if isinstance(step, dict):
            t = step.get("step", t)
        next_decision = getattr(step, "next_decision", None)
        if isinstance(step, dict):
            next_decision = step.get("next_decision", next_decision)
        token_usage_agent = getattr(step, "token_usage", None)
        if isinstance(step, dict):
            token_usage_agent = step.get("token_usage", token_usage_agent)

        if next_decision == "clarify":
            question = ""
            reply = ""
            if isinstance(step, dict):
                question = step.get("clarification_question") or ""
                reply = step.get("user_reply") or ""
            else:
                question = getattr(step, "clarification_question", "") or ""
                reply = getattr(step, "user_reply", "") or ""
            msg_idx = _skip_system_messages(messages, msg_idx)
            if msg_idx < len(messages) and isinstance(messages[msg_idx], AIMessage):
                if not question:
                    question = messages[msg_idx].content or ""
                msg_idx += 1
            msg_idx = _skip_system_messages(messages, msg_idx)
            if msg_idx < len(messages) and isinstance(messages[msg_idx], HumanMessage):
                if not reply:
                    reply = messages[msg_idx].content or ""
                msg_idx += 1
            steps.append(
                {
                    "t": t,
                    "action": "clarify",
                    "clarification_question": question,
                    "clarification_question_words": _count_words(question),
                    "user_reply": reply,
                    "user_reply_words": _count_words(reply),
                }
            )
        else:
            action_id = getattr(step, "action_id", None)
            if isinstance(step, dict):
                action_id = step.get("action_id", action_id)
            # Consume the execution message to keep the next step aligned.
            msg_idx = _skip_system_messages(messages, msg_idx)
            if msg_idx < len(messages) and isinstance(messages[msg_idx], AIMessage):
                msg_idx += 1
            steps.append(
                {
                    "t": t,
                    "action": "execute",
                    "chosen_action": action_id,
                }
            )

        if token_usage_agent:
            steps[-1]["token_usage_agent"] = token_usage_agent

        # Attribute trial-level reflection usage to the final step.
        is_last_step = (step_idx == len(trace) - 1)
        if is_last_step and reflection_usage:
            steps[-1]["token_usage_reflect"] = reflection_usage

    return steps
