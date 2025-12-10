import re
from typing import List
from langchain_core.messages import HumanMessage, AIMessage
from .state import id_to_action

try:
    import tiktoken
except ImportError:  # pragma: no cover - optional dependency
    tiktoken = None



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
    user_prompt += "※以下の物品リストは「今見えている物」だけです。リストにない物でも他の場所から持ってくることができます。\n"
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

    # 如果你想在单个 trial 里也看到“初始用户发言”，可以加上这段（可选）
    if msg_idx < len(messages) and isinstance(messages[msg_idx], HumanMessage):
        lines.append(f"User: {messages[msg_idx].content}")
        msg_idx += 1
        lines.append("")

    # 遍历每一个 router step
    for t in trace:
        lines.append(f"【Step {t.step}】")
        if t.thought is not None:
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
            a_desc = id_to_action.get(t.action_id, f"Unknown action_id={t.action_id}")
            lines.append(f"\tAgent: [EXECUTE] {a_desc}")

            # execute_node 会追加一条 AIMessage 说明执行情况
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


def format_multi_trial_episode(sample, trials: List[dict]) -> str:
    """
    trials: 每个元素是 {
        "final_state": dict,
        "reflections_before": str | None,
        "reflections_after": str | None,
    }
    """
    lines: List[str] = []

    # Gold action（标签）从 sample 中恢复：
    # 优先使用 sample["reflective_action"] 的描述，如果没有再退回到 id_to_action。
    gold_desc = sample.get("reflective_action")

    # ===== 用户信息 / 场景描述（只写一次）=====
    lines.append("================ 📖 User Context ================\n")
    lines.append(make_system_prompt_from_one_sample(sample=sample))
    lines.append("")

    # ===== 多个 trial =====
    for i, trial in enumerate(trials, start=1):
        final_state = trial["final_state"]
        reflections_before = trial.get("reflections_before")
        reflections_after = trial.get("reflections_after")
        wrong_actions = trial.get("wrong_actions")
        mode = trial.get("mode", "baseline")

        lines.append(f"---------------- ⚙️ Trial {i} ----------------\n")

        # 累积的错误动作列表（用于提醒下一轮需要避开什么）
        wrong_block = format_wrong_actions_block(wrong_actions)
        if wrong_block and mode in {"reflect", "reflect_k"}:
            lines.append(wrong_block)
            lines.append("")

        # trial 前的 reflection（如果有）
        if reflections_before:
            # 现在由 reflections_before 自带标题（例如：
            # [Reflection Before this trial 前 2 次]）
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

    lines.append(f"Gold action: {gold_desc}")

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
    提取单个 trial 中的所有澄清问题（按时间顺序返回文本列表）。
    """
    messages = final_state.get("messages") or []
    trace = final_state.get("trace") or []

    questions: List[str] = []
    msg_idx = 1  # 跳过 system prompt

    for step in trace:
        next_decision = getattr(step, "next_decision", None)
        if isinstance(step, dict):
            next_decision = step.get("next_decision")

        if next_decision == "clarify":
            if msg_idx < len(messages) and isinstance(messages[msg_idx], AIMessage):
                questions.append(messages[msg_idx].content or "")

            if msg_idx < len(messages) and isinstance(messages[msg_idx], AIMessage):
                msg_idx += 1
            if msg_idx < len(messages) and isinstance(messages[msg_idx], HumanMessage):
                msg_idx += 1
        else:
            if msg_idx < len(messages) and isinstance(messages[msg_idx], AIMessage):
                msg_idx += 1

    return questions


def extract_clarification_word_counts(final_state: dict) -> List[int]:
    """
    从单个 trial 的最终 state 中提取所有澄清问题的词数列表。
    返回值是按时间顺序排列的澄清问题词数。
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
            if msg_idx < len(messages) and isinstance(messages[msg_idx], AIMessage):
                question = messages[msg_idx].content or ""
            if msg_idx < len(messages) and isinstance(messages[msg_idx], AIMessage):
                msg_idx += 1
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
