import json
import re

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from pydantic import ValidationError

from .state import AgentState, id_to_action, AgentOutput, AgentStep
from .prompts import agent_system_prompt, clarify_system_prompt, user_sim_system_prompt
from .llm_factory import get_primary_llm
from .gpt_llm import OpenAIChatModel

load_dotenv(".env", override=True)

# Main agent (决策/clarify/execute)
llm = get_primary_llm()

# User simulator（保持 OpenAI，确保回答稳定）
user_llm = OpenAIChatModel(model="gpt-5-mini")

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


def _parse_clarify_output(raw_resp: AIMessage) -> tuple[str | None, str]:
    raw_text = str(getattr(raw_resp, "content", "")).strip()
    cleaned = _sanitize_json_text(raw_text)

    # 先尝试整段 JSON
    for candidate in (cleaned, _extract_first_json_object(cleaned)):
        if not candidate:
            continue
        try:
            data = json.loads(candidate, strict=False)
            if isinstance(data, dict):
                thought = data.get("thought")
                question = data.get("question")
                if not isinstance(thought, str):
                    thought = None
                if isinstance(question, str) and question.strip():
                    return thought, question.strip()
        except Exception:
            pass

    # 兜底：把清洗后的文本当作问题（避免把 ```json 带下去）
    return None, cleaned


# ========== Agent Node ==========

def agent(state: AgentState) -> AgentState:

    # Simple step limit: if max_steps exceeded, force execute failure action -1
    if state.step >= state.max_steps:
        return {
            "next_decision": "execute",
            "action_id": -1,
            "step": state.step + 1,
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

    base_messages = [m for m in state.messages if not (
        isinstance(m, SystemMessage)
        and isinstance(m.content, str)
        and m.content.startswith("【Agentの明確化意図】")
    )]
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
    raw_resp = llm.invoke(messages)
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

    # 根据当前 mode 决定是否允许 clarify：
    # - clarify / reflect: 使用模型预测的 clarify / execute
    # - cot / direct: 强制只执行 execute（纯 CoT / 纯 direct 一步到位）
    allow_clarify = state.mode in {"clarify", "reflect", "reflect_k"}
    if allow_clarify and state.clarify_quota_left is not None:
        allow_clarify = state.clarify_quota_left > 0

    if not allow_clarify:
        next_decision = "execute"


    clarify_quota_left = state.clarify_quota_left

    new_messages = state.messages
    if next_decision == "clarify" and resp.thought:
        # 明确化意图写入对话，供 clarify 节点提醒；agent 侧会过滤掉这类消息
        new_messages = state.messages + [
            SystemMessage(content="【Agentの明確化意図】\n" + resp.thought)
        ]
        if clarify_quota_left is not None:
            clarify_quota_left = max(clarify_quota_left - 1, 0)

    return {
        "next_decision": next_decision,
        "action_id": action_id,
        "step": state.step + 1,
        "clarify_quota_left": clarify_quota_left,
        "clarify_quota_total": state.clarify_quota_total,
        "messages": new_messages,
        "trace": state.trace
        + [
            AgentStep(
                step=state.step + 1,
                thought=resp.thought,
                next_decision=next_decision,
                action_id=action_id,
            )
        ],
    }


# ========== Clarify Node ==========

def clarify_node(state: AgentState) -> AgentState:

    system_prompt = clarify_system_prompt()

    messages = [SystemMessage(content=system_prompt)]

    # 将 agent 上一步的思考传递给 clarify 模型，减少意图偏差
    agent_thought = state.trace[-1].thought if state.trace else None
    if agent_thought:
        messages.append(SystemMessage(content="【Agent思考メモ】\n" + agent_thought))

    messages.extend(state.messages)

    resp = llm.invoke(messages)
    thought, question = _parse_clarify_output(resp)

    new_messages = list(state.messages)
    if thought:
        new_messages.append(SystemMessage(content="【Clarify思考】\n" + thought))
    new_messages.append(AIMessage(content=question))

    return {
        "messages": new_messages,
        "next_decision": None,
        "action_id": None,
    }


# ========== User Simulation Node ==========

def user_sim_node(state: AgentState) -> AgentState:

    action_id = state.label_action_id
    action_desc = id_to_action[action_id]

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

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=last_ai_msg.content),
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
    lambda state: state.next_decision,
    {
        "clarify": "clarify",
        "execute": "execute",
    },
)

builder.add_edge("clarify", "user_sim")
builder.add_edge("user_sim", "agent")

builder.add_edge("execute", END)

graph = builder.compile()
