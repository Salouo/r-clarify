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

# Main agent (决策/clarify/execute)
llm = get_primary_llm()

# User simulator（保持 OpenAI，确保回答稳定）
user_llm = OpenAIChatModel(model="gpt-5.2")

_FENCE_RE = re.compile(
    r"```[a-zA-Z0-9_-]*\s*([\s\S]*?)\s*```",
    re.MULTILINE,
)


def _trace_to_messages(trace: list[AgentStep]) -> list[SystemMessage]:
    """Expose prior agent thinking/decisions to the next turn as system notes."""
    msgs: list[SystemMessage] = []
    for t in trace:
        action_desc = id_to_action.get(t.action_id, "不明な行動")
        thought = t.thought or "（thought なし）"
        msgs.append(
            SystemMessage(
                content=(
                    f"【Agent前回ログ step {t.step}】thought: {thought}; "
                    f"next_decision: {t.next_decision}; action_id: {t.action_id}（{action_desc}）"
                )
            )
        )
    return msgs


def _strip_code_fence(text: str) -> str:
    text = (text or "").strip()
    blocks = _FENCE_RE.findall(text)
    if blocks:
        # 优先挑看起来像 JSON 的 block
        for b in blocks:
            b2 = b.strip()
            if "{" in b2 and "}" in b2:
                return b2
        return blocks[0].strip()
    return text


def _sanitize_json_text(text: str) -> str:
    text = _strip_code_fence(text)
    # 清理控制字符，避免 json.loads 报错
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)
    # 模型有时在字符串里直接换行，JSON 不允许裸换行
    text = text.replace("\r", " ").replace("\n", " ")
    # 模型有时会在字符串结束后多输出一个单引号再跟逗号，破坏 JSON
    text = text.replace("\"',", "\",")
    return text.strip()


def _extract_first_json_object(text: str) -> str | None:
    """
    找到第一个括号平衡的 {...}，并且忽略 JSON 字符串中的 { }。
    """
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

        # not in string
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
        # 先直接解析
        try:
            return json.loads(cleaned)
        except Exception:
            try:
                return json.loads(cleaned, strict=False)
            except Exception:
                # 抽取第一个平衡的 {...}
                snippet = _extract_first_json_object(cleaned)
                if snippet:
                    return json.loads(snippet, strict=False)
                raise

    try:
        data = _parse_json(raw_text)
        if not isinstance(data, dict):
            raise ValueError("Agent 输出必须是 JSON 对象。")
        if isinstance(data.get("action_id"), str) and data["action_id"].strip().isdigit():
            data["action_id"] = int(data["action_id"].strip())
        if isinstance(data.get("question"), str):
            data["question"] = data["question"].strip()
        return AgentOutput(**data)
    except (json.JSONDecodeError, ValidationError, ValueError) as exc:
        print(
            "[Agent] 解析模型输出失败，将回退到失败动作。"
            f"错误: {exc}\n原始输出: {raw_text}"
        )
        # 关键：这里依赖 schema 允许 -1
        return AgentOutput(
            thought="解析失败。execute -1 にフォールバック。",
            next_decision="execute",
            action_id=-1,
        )


# ========== Agent Node ==========

def agent(state: AgentState) -> AgentState:

    # Simple step limit: if max_steps exceeded, force execute failure action -1
    if state.step >= state.max_steps:
        return {
            "next_decision": "execute",
            "action_id": -1,
            "step": state.max_steps,
            "clarify_question": None,
            "trace": state.trace + [AgentStep(
                step=state.step,
                thought="最大ステップ数を超過したため、失敗を判定します。",
                next_decision="execute",
                action_id=-1,
            )]
        }

    # 根据不同的 mode 选择不同的 prompt
    system_prompt = agent_system_prompt(
        state.mode,
        clarify_quota_total=state.clarify_quota_total,
        clarify_quota_left=state.clarify_quota_left,
    )

    # 让 agent 在同一 trial 内能看到自己历史的思考/决策
    trace_messages = _trace_to_messages(state.trace)

    base_messages = list(state.messages)
    has_same_system = (
        base_messages
        and isinstance(base_messages[0], SystemMessage)
        and base_messages[0].content == system_prompt
    )
    if has_same_system:
        # 保持原有的系统 prompt，不重复；trace 插在其后
        messages = [base_messages[0], *trace_messages, *base_messages[1:]]
    else:
        messages = [SystemMessage(content=system_prompt), *trace_messages, *base_messages]
    
    # 得到干净 agent 输出
    raw_resp = llm.invoke(messages)
    token_usage = {}
    try:
        token_usage = getattr(raw_resp, "additional_kwargs", {}).get("token_usage") or {}
    except Exception:
        token_usage = {}
    resp = _safe_agent_output(raw_resp)

    valid_decisions = {"clarify", "execute"}
    next_decision = (resp.next_decision or "").strip().lower()
    if next_decision not in valid_decisions:
        next_decision = "execute"

    # 如果模型没有给出合法的 action_id（例如为 null 或不在 1〜40 范围内），
    # 就回退到失败动作 -1，避免后续节点报错。
    if isinstance(resp.action_id, int) and 1 <= resp.action_id <= 40:
        action_id = resp.action_id
    else:
        action_id = -1

    clarify_question = (resp.question or "").strip() if next_decision == "clarify" else None

    # 根据当前 mode 决定是否允许 clarify：
    # - clarify / reflect: 使用模型预测的 clarify / execute
    # - cot / direct: 强制只执行 execute（纯 CoT / 纯 direct 一步到位）
    allow_clarify = state.mode in {"clarify", "reflect", "reflect_k"}
    if allow_clarify and state.clarify_quota_left is not None:
        allow_clarify = state.clarify_quota_left > 0

    if not allow_clarify:
        next_decision = "execute"
        clarify_question = None


    clarify_quota_left = state.clarify_quota_left

    new_messages = state.messages
    if next_decision == "clarify":
        if clarify_quota_left is not None:
            clarify_quota_left = max(clarify_quota_left - 1, 0)


    return {
        "next_decision": next_decision,
        "action_id": action_id,
        "step": state.step,  # clarify/execute 节点才记步
        "clarify_quota_left": clarify_quota_left,
        "clarify_quota_total": state.clarify_quota_total,
        "messages": new_messages,
        "clarify_question": clarify_question,
        "trace": state.trace
        + [
            AgentStep(
                step=state.step + 1,  # 这里标注将要执行的动作序号
                thought=resp.thought,
                next_decision=next_decision,
                action_id=action_id,
                question=clarify_question,
                token_usage=token_usage or None,
            )
        ],
    }


# ========== Clarify Node ==========

def clarify_node(state: AgentState) -> AgentState:

    question_text = (state.clarify_question or "").strip()

    new_messages = list(state.messages)
    if not question_text:
        question_text = "エージェントが明確化質問を生成できなかったため、失敗として扱います。"
        return {
            "messages": new_messages + [AIMessage(content=question_text)],
            "next_decision": "execute",
            "action_id": -1,
            "step": state.max_steps,
            "clarify_question": None,
        }

    new_messages.append(AIMessage(content=question_text))

    return {
        "messages": new_messages,
        "step": state.step + 1,
        "next_decision": None,
        "action_id": None,
        "clarify_question": None,
    }


# ========== User Simulation Node ==========

def user_sim_node(state: AgentState) -> AgentState:

    action_id = state.label_action_id
    action_desc = id_to_action.get(action_id, "不明な行動")
    if action_id not in id_to_action:
        # ラベルが不明な場合は即座に失敗扱い
        fallback_msg = "ラベルが不明のため、失敗として扱います。"
        return {
            "messages": state.messages + [HumanMessage(content=fallback_msg)],
            "step": state.max_steps,
            "next_decision": "execute",
            "action_id": -1,
        }

    # Find the last AIMessage as the "clarification question"
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
        # Agent が clarify を選んだのに質問が生成されていない場合は即座に失敗判定へ
        question_text = "エージェントが明確化質問が生成されなかったため、失敗として扱います。"
        return {
            "messages": state.messages + [HumanMessage(content=question_text)],
            "step": state.max_steps,       # 次の agent 呼び出しで強制終了させる
            "next_decision": "execute",
            "action_id": -1,
        }

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=question_text),
    ]

    resp = user_llm.invoke(messages)

    return {
        "messages": state.messages + [HumanMessage(content=resp.content)],
    }


# ========== Reflection Node（明確化質問後のまとめ） ==========

# ========== Execute Node (END) ==========

def execute_node(state: AgentState) -> AgentState:

    # 如果 action_id 异常（None 等），回退到失败动作 -1
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


# ========== Build Graph ==========

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
