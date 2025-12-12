import re
import tiktoken

from typing import List
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
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
    # 明示：ここに列挙しているのは「現在見えている物」だけで、リストにない物も他の場所から持ってこられる
    user_prompt += "※以上の物品リストは「今見えている物」だけです。リストにない物でも他の場所から持ってくることができます。\n"
    user_prompt += "\n"

    return user_prompt


def format_wrong_actions_block(wrong_actions) -> str:
    """Render a bullet list of known-wrong actions (id, desc)."""
    if not wrong_actions:
        return ""
    lines: List[str] = ["【これまでに誤って選んだ行動（避けるべき）】"]
    for aid, desc in wrong_actions:
        lines.append(f"- {desc}")
    return "\n".join(lines)


def format_single_trial_episode(final_state: dict) -> str:
    """只描述『一个 trial』发生了什么，用于 reflection / 在大 episode 里嵌入。"""
    lines: List[str] = []

    messages = final_state["messages"]
    trace = final_state.get("trace", [])
    label_action_id = final_state.get("label_action_id")
    pred_id = final_state.get("action_id")
    pred_desc = id_to_action.get(pred_id, f"Unknown action_id={pred_id}")

    # 跳过最开始的 system prompt, messages[0]
    msg_idx = 1

    # 收集明確化質問后的反思小结（SystemMessage，开头是特定标记）
    reflection_memos = [
        m.content
        for m in messages
        if isinstance(m, SystemMessage) and str(m.content).startswith("【明確化質問後のメモ】")
    ]

    def _skip_system(idx: int) -> int:
        """Advance index past any SystemMessage blocks."""
        while idx < len(messages) and isinstance(messages[idx], SystemMessage):
            idx += 1
        return idx

    def _emit_clarify_system(idx: int) -> int:
        """
        Consume连续的 SystemMessage，如果是明確化関連的思考/意图则写到日志里。
        返回消费后新的 idx。
        """
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

    # 如果你想在单个 trial 里也看到“初始用户发言”，可以加上这段（可选）
    msg_idx = _skip_system(msg_idx)
    if msg_idx < len(messages) and isinstance(messages[msg_idx], HumanMessage):
        lines.append(f"User: {messages[msg_idx].content}")
        msg_idx += 1
        lines.append("")

    # 遍历每一个 agent step
    for t in trace:
        lines.append(f"【Step {t.step}】")
        if t.thought is not None:
            lines.append(f"\tAgent: <thinking> {t.thought} </thinking>")

        if t.next_decision == "clarify":
            lines.append("\tAgent: [CLARIFY]")

            # clarify 之后：AI（明確化質問）+ Human（用户回答）
            msg_idx = _emit_clarify_system(msg_idx)
            if msg_idx < len(messages) and isinstance(messages[msg_idx], AIMessage):
                lines.append(f"\t>> Agent: {messages[msg_idx].content}")
                msg_idx += 1

            msg_idx = _emit_clarify_system(msg_idx)
            if msg_idx < len(messages) and isinstance(messages[msg_idx], HumanMessage):
                lines.append(f"\t>> User: {messages[msg_idx].content}")
                msg_idx += 1

            # 明確化質問后，若有反思总结则在这里展示
            if reflection_memos:
                memo_text = reflection_memos.pop(0)
                lines.append("\tAgent: [REFLECT]")
                for memo_line in memo_text.splitlines():
                    lines.append(f"\t   {memo_line}")

        else:  # execute
            a_desc = id_to_action.get(t.action_id, f"Unknown action_id={t.action_id}")
            lines.append(f"\tAgent: [EXECUTE] {a_desc}")

            # execute_node 会追加一条 AIMessage 说明执行情况
            msg_idx = _skip_system(msg_idx)
            if msg_idx < len(messages) and isinstance(messages[msg_idx], AIMessage):
                lines.append(f"\t>> Agent は {messages[msg_idx].content}")
                msg_idx += 1

        lines.append("")

    # 单 trial 的结果 ()
    lines.append("================ 🚀 Trial Result ================")
    lines.append(f"Predicted action {pred_desc}")
    if pred_id == label_action_id:
        lines.append("Outcome: Correct ✅")
    else:
        lines.append("Outcome: Incorrect ❌")

    return "\n".join(lines)


def _count_words(text: str) -> int:
    """
    Count tokens in a prompt.

    - Latin text: split on standard word tokens (letters/digits, keep contractions).
    - Japanese/Chinese: treat contiguous CJK/Hiragana/Katakana blocks as one token.
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
        # 如果包含日/中/假名字符，就用字符数近似“词数”
        if re.search(r"[\u3040-\u30FF\u31F0-\u31FF\uFF65-\uFF9F\u4E00-\u9FFFー]", tok):
            total += len(tok)
        else:
            total += 1
    return total


def _count_tokens(text: str) -> int:
    """
    粗略 token 计数：优先用 tiktoken(cl100k_base)，失败则退回词数。
    """
    if not text:
        return 0

    if tiktoken:
        try:
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception:
            pass
    return _count_words(text)


def extract_clarification_questions(final_state: dict) -> List[str]:
    """
    提取单个 trial 中的所有明確化質問（按时间顺序返回文本列表）。
    """
    messages = final_state.get("messages") or []
    trace = final_state.get("trace") or []

    questions: List[str] = []
    msg_idx = 1  # 跳过 system prompt

    def _skip_system(idx: int) -> int:
        while idx < len(messages) and isinstance(messages[idx], SystemMessage):
            idx += 1
        return idx

    for step in trace:
        next_decision = getattr(step, "next_decision", None)
        if isinstance(step, dict):
            next_decision = step.get("next_decision")

        if next_decision == "clarify":
            msg_idx = _skip_system(msg_idx)
            if msg_idx < len(messages) and isinstance(messages[msg_idx], AIMessage):
                questions.append(messages[msg_idx].content or "")

            if msg_idx < len(messages) and isinstance(messages[msg_idx], AIMessage):
                msg_idx += 1
            msg_idx = _skip_system(msg_idx)
            if msg_idx < len(messages) and isinstance(messages[msg_idx], HumanMessage):
                msg_idx += 1
        else:
            msg_idx = _skip_system(msg_idx)
            if msg_idx < len(messages) and isinstance(messages[msg_idx], AIMessage):
                msg_idx += 1

    return questions


def extract_clarification_word_counts(final_state: dict) -> List[int]:
    """
    从单个 trial 的最终 state 中提取所有明確化質問的词数列表。
    返回值是按时间顺序排列的明確化質問词数。
    """
    questions = extract_clarification_questions(final_state)
    return [_count_words(q) for q in questions]


def extract_trial_steps(final_state: dict) -> List[dict]:
    """
    把单个 trial 的完整步骤序列化，便于直接写入 results.json。
    结构示例：
    [
      {"t": 1, "action": "clarify", "question": "...", "question_words": 23, "user_reply": "..."},
      {"t": 2, "action": "execute", "chosen_action": 12},
    ]
    """
    messages = final_state.get("messages") or []
    trace = final_state.get("trace") or []

    steps: List[dict] = []
    msg_idx = 1  # 跳过 system prompt

    def _skip_system(idx: int) -> int:
        while idx < len(messages) and isinstance(messages[idx], SystemMessage):
            idx += 1
        return idx

    for step in trace:
        t = getattr(step, "step", None)
        if isinstance(step, dict):
            t = step.get("step", t)
        next_decision = getattr(step, "next_decision", None)
        if isinstance(step, dict):
            next_decision = step.get("next_decision", next_decision)

        if next_decision == "clarify":
            question = ""
            reply = ""
            msg_idx = _skip_system(msg_idx)
            if msg_idx < len(messages) and isinstance(messages[msg_idx], AIMessage):
                question = messages[msg_idx].content or ""
            if msg_idx < len(messages) and isinstance(messages[msg_idx], AIMessage):
                msg_idx += 1
            msg_idx = _skip_system(msg_idx)
            if msg_idx < len(messages) and isinstance(messages[msg_idx], HumanMessage):
                reply = messages[msg_idx].content or ""
                msg_idx += 1
            steps.append(
                {
                    "t": t,
                    "action": "clarify",
                    "question": question,
                    "question_words": _count_words(question),
                    "question_tokens": _count_tokens(question),
                    "user_reply": reply,
                    "user_reply_words": _count_words(reply),
                    "user_reply_tokens": _count_tokens(reply),
                }
            )
        else:
            action_id = getattr(step, "action_id", None)
            if isinstance(step, dict):
                action_id = step.get("action_id", action_id)
            # execute_node 会在 messages 中追加一条 AIMessage，跳过它
            msg_idx = _skip_system(msg_idx)
            if msg_idx < len(messages) and isinstance(messages[msg_idx], AIMessage):
                msg_idx += 1
            steps.append(
                {
                    "t": t,
                    "action": "execute",
                    "chosen_action": action_id,
                }
            )

    return steps
