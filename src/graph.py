import json
import re

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from pydantic import ValidationError

from .state import AgentState, id_to_action, AgentOutput, AgentStep
from .prompts import agent_system_prompt, user_sim_system_prompt
from .llm_factory import get_primary_llm
from .gpt_llm import OpenAIChatModel

load_dotenv(".env", override=True)

llm = get_primary_llm()

# The user simulator uses a fixed model across agent backends.
user_llm = OpenAIChatModel(model="gpt-5.2")

_FENCE_RE = re.compile(
    r"```[a-zA-Z0-9_-]*\s*([\s\S]*?)\s*```",
    re.MULTILINE,
)


def _trace_to_messages(trace: list[AgentStep]) -> list[SystemMessage]:
    """Format earlier decisions in the current trial as system messages."""
    msgs: list[SystemMessage] = []
    for t in trace:
        thought = t.thought or "（thought なし）"
        if t.next_decision == "clarify":
            question_text = t.clarification_question or "（clarification_question なし）"
            reply_text = t.user_reply or "（user_reply なし）"
            decision_detail = (
                f"next_decision: clarify; action_id: None（確認を選択、まだ行動未決定）; "
                f"clarification_question: {question_text}; user_reply: {reply_text}"
            )
        else:
            action_desc = id_to_action.get(t.action_id, "不明な行動")
            decision_detail = (
                f"next_decision: {t.next_decision}; action_id: {t.action_id}（{action_desc}）"
            )
        msgs.append(
            SystemMessage(
                content=(
                    f"【Agent前回ログ step {t.step}】thought: {thought}; "
                    f"{decision_detail}"
                )
            )
        )
    return msgs


def _strip_code_fence(text: str) -> str:
    text = (text or "").strip()
    blocks = _FENCE_RE.findall(text)
    if blocks:
        # Model replies can include more than one fenced block.
        for b in blocks:
            b2 = b.strip()
            if "{" in b2 and "}" in b2:
                return b2
        return blocks[0].strip()
    return text


def _sanitize_json_text(text: str) -> str:
    text = _strip_code_fence(text)
    # Strip control characters that are invalid in JSON strings.
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)
    # Model replies can contain unescaped line breaks inside JSON strings.
    text = text.replace("\r", " ").replace("\n", " ")
    # Some replies add a stray single quote after a JSON string.
    text = text.replace("\"',", "\",")
    return text.strip()


def _extract_first_json_object(text: str) -> str | None:
    """Return the first balanced JSON object, ignoring braces inside strings."""
    start = None
    depth = 0
    in_str = False
    esc = False

    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            continue

        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    return text[start : i + 1]

    return None


def _safe_agent_output(message: AIMessage) -> AgentOutput:
    raw_text = str(getattr(message, "content", "")).strip()

    def _parse_json(text: str) -> dict:
        cleaned = _sanitize_json_text(text)
        try:
            return json.loads(cleaned)
        except Exception:
            try:
                return json.loads(cleaned, strict=False)
            except Exception:
                # Recover an object embedded in surrounding prose.
                snippet = _extract_first_json_object(cleaned)
                if snippet:
                    return json.loads(snippet, strict=False)
                raise

    try:
        data = _parse_json(raw_text)
        if not isinstance(data, dict):
            raise ValueError("Agent output must be a JSON object.")
        if "clarification_question" not in data and "question" in data:
            data["clarification_question"] = data.get("question")
        if isinstance(data.get("action_id"), str) and data["action_id"].strip().isdigit():
            data["action_id"] = int(data["action_id"].strip())
        if isinstance(data.get("clarification_question"), str):
            data["clarification_question"] = data["clarification_question"].strip()
        return AgentOutput(**data)
    except (json.JSONDecodeError, ValidationError, ValueError):
        # A failed parse is recorded as action -1.
        return AgentOutput(
            thought="解析失败。execute -1 にフォールバック。",
            next_decision="execute",
            action_id=-1,
        )


def agent(state: AgentState) -> AgentState:
    """Choose the next action within the trial limits."""
    if state.step >= state.max_steps:
        return {
            "next_decision": "execute",
            "action_id": -1,
            "step": state.max_steps,
            "clarification_question": None,
            "trace": state.trace + [AgentStep(
                step=state.step,
                thought="最大ステップ数を超過したため、失敗を判定します。",
                next_decision="execute",
                action_id=-1,
            )]
        }

    system_prompt = agent_system_prompt(
        state.mode,
        clarify_quota_total=state.clarify_quota_total,
        clarify_quota_left=state.clarify_quota_left,
    )

    trace_messages = _trace_to_messages(state.trace)

    base_messages = list(state.messages)

    has_same_system: bool = (
        base_messages
        and isinstance(base_messages[0], SystemMessage)
        and base_messages[0].content == system_prompt
    )
    # Place the trial trace immediately after the system prompt.
    if has_same_system:
        messages = [base_messages[0], *trace_messages, *base_messages[1:]]
    else:
        messages = [SystemMessage(content=system_prompt), *trace_messages, *base_messages]
    
    raw_resp = llm.invoke(messages)

    try:
        token_usage = getattr(raw_resp, "additional_kwargs", {}).get("token_usage") or {}
    except Exception:
        token_usage = {}
    
    resp = _safe_agent_output(raw_resp)

    valid_decisions = {"clarify", "execute"}
    next_decision = (resp.next_decision or "").strip().lower()
    if next_decision not in valid_decisions:
        next_decision = "execute"

    if isinstance(resp.action_id, int) and 1 <= resp.action_id <= 40:
        action_id = resp.action_id
    else:
        action_id = -1

    clarification_question = (
        (resp.clarification_question or "").strip() if next_decision == "clarify" else None
    )

    allow_clarify = state.mode in {
        "clarify",
        "r-clarify",
        "r-clarify_reflexion",
        "action_elimination",
        "non_thinking_clarify",
        "r-clarify_non_cot",
    }
    if allow_clarify and state.clarify_quota_left is not None:
        allow_clarify = state.clarify_quota_left > 0

    if not allow_clarify:
        next_decision = "execute"
        clarification_question = None

    # Action elimination rules out actions that failed in earlier trials.
    blocked_repeated_action_id = None
    if (
        state.mode == "action_elimination"
        and next_decision == "execute"
        and action_id in set(state.failed_action_ids or [])
    ):
        blocked_repeated_action_id = action_id
        action_id = -1

    clarify_quota_left = state.clarify_quota_left
    if next_decision == "clarify" and clarify_quota_left is not None:
        clarify_quota_left = max(clarify_quota_left - 1, 0)

    return {
        "next_decision": next_decision,
        "action_id": action_id,
        "step": state.step,  # Clarification and execution advance the step counter.
        "clarify_quota_left": clarify_quota_left,
        "clarify_quota_total": state.clarify_quota_total,
        "messages": state.messages,
        "clarification_question": clarification_question,
        "trace": state.trace
        + [
            AgentStep(
                step=state.step + 1,
                thought=resp.thought,
                next_decision=next_decision,
                action_id=action_id,
                clarification_question=clarification_question,
                token_usage={
                    **(token_usage or {}),
                    "blocked_repeated_action_id": blocked_repeated_action_id,
                } if blocked_repeated_action_id is not None else (token_usage or None),
            )
        ],
    }


def clarify_node(state: AgentState) -> AgentState:
    """Append the clarification question to the dialogue."""
    question_text = (state.clarification_question or "").strip()

    new_messages = list(state.messages)
    if not question_text:
        question_text = "エージェントが明確化質問を生成できなかったため、失敗として扱います。"
        return {
            "messages": new_messages + [AIMessage(content=question_text)],
            "next_decision": "execute",
            "action_id": -1,
            "step": state.max_steps,
            "clarification_question": None,
        }

    new_messages.append(AIMessage(content=question_text))

    return {
        "messages": new_messages,
        "step": state.step + 1,
        "next_decision": None,
        "action_id": None,
        "clarification_question": None,
    }


def user_sim_node(state: AgentState) -> AgentState:
    """Generate the simulated user reply and attach it to the trial trace."""
    action_id = state.label_action_id
    action_desc = id_to_action.get(action_id, "不明な行動")
    if action_id not in id_to_action:
        fallback_msg = "ラベルが不明のため、失敗として扱います。"
        return {
            "messages": state.messages + [HumanMessage(content=fallback_msg)],
            "step": state.max_steps,
            "next_decision": "execute",
            "action_id": -1,
        }

    # The latest agent message contains the clarification question.
    last_ai_msg = None
    for msg in reversed(state.messages):
        if isinstance(msg, AIMessage):
            last_ai_msg = msg
            break

    system_prompt = user_sim_system_prompt(
        action_desc=action_desc,
        utterance=state.original_utterance,
        env=state.env,
    )

    if last_ai_msg:
        question_text = last_ai_msg.content
    else:
        question_text = "エージェントが明確化質問が生成されなかったため、失敗として扱います。"
        return {
            "messages": state.messages + [HumanMessage(content=question_text)],
            "step": state.max_steps,  # End the trial on the next agent step.
            "next_decision": "execute",
            "action_id": -1,
        }

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=question_text),
    ]

    resp = user_llm.invoke(messages)

    new_trace = list(state.trace)
    if new_trace:
        last = new_trace[-1]
        if isinstance(last, AgentStep):
            new_trace[-1] = last.copy(update={"user_reply": resp.content})
        elif isinstance(last, dict):
            last = dict(last)
            last["user_reply"] = resp.content
            new_trace[-1] = last

    return {
        "messages": state.messages + [HumanMessage(content=resp.content)],
        "trace": new_trace,
    }


def execute_node(state: AgentState) -> AgentState:
    """Record the selected action and advance the trial by one step."""
    action_id = state.action_id if state.action_id in id_to_action else -1
    desc = id_to_action[action_id]
    if action_id == -1:
        result_text = "動作を特定できません。"
    else:
        result_text = f"{desc}を执行しました。"

    return {
        "messages": state.messages + [AIMessage(content=result_text)],
        "step": state.step + 1,
    }


builder = StateGraph(AgentState)

builder.add_node("agent", agent)
builder.add_node("clarify", clarify_node)
builder.add_node("user_sim", user_sim_node)
builder.add_node("execute", execute_node)

builder.set_entry_point("agent")

builder.add_conditional_edges(
    "agent",
    lambda s: s.next_decision if s.next_decision in {"clarify","execute"} else "execute",
    {"clarify": "clarify", "execute": "execute"},
)

builder.add_edge("clarify", "user_sim")
builder.add_edge("user_sim", "agent")

builder.add_edge("execute", END)

graph = builder.compile()
