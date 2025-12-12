import json
import re

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from pydantic import ValidationError

from .state import AgentState, id_to_action, AgentOutput, AgentStep
from .retry_utils import invoke_with_retry
from .prompts import agent_system_prompt, clarify_system_prompt, user_sim_system_prompt
from .llm_factory import get_primary_llm
from .gpt_llm import OpenAIChatModel
from .reflection import summarize_clarification

load_dotenv(".env", override=True)

# Main agent (决策/clarify/execute)
llm = get_primary_llm()

# User simulator（保持 OpenAI，确保回答稳定）
user_llm = OpenAIChatModel(model="gpt-5-mini")


def _safe_agent_output(message: AIMessage) -> AgentOutput:
    """Parse agent JSON output; fallback to failure when parsing breaks."""
    raw_text = str(getattr(message, "content", "")).strip()

    def _strip_code_fence(text: str) -> str:
        # 去掉 ``` 或 ```json 样式的包裹
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        return text.strip()

    def _parse_json(text: str) -> dict:
        cleaned = _strip_code_fence(text)
        try:
            return json.loads(cleaned)
        except Exception:
            # 尝试对 thought 字段内的未转义引号做修复
            thought_match = re.search(
                r'"thought"\s*:\s*"(.*?)"\s*,\s*"next_decision"', cleaned, flags=re.S
            )
            if thought_match:
                thought_content = thought_match.group(1)
                escaped_thought = thought_content.replace("\\", "\\\\").replace('"', '\\"')
                cleaned = (
                    cleaned[:thought_match.start(1)]
                    + escaped_thought
                    + cleaned[thought_match.end(1):]
                )
                try:
                    return json.loads(cleaned)
                except Exception:
                    pass
            # 尝试提取第一个 { ... } 块再解析
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end != -1 and end > start:
                snippet = cleaned[start : end + 1]
                return json.loads(snippet)
            raise

    try:
        data = _parse_json(raw_text) # 解析LLM的输出为json字典
        if not isinstance(data, dict):
            raise ValueError("Agent 输出必须是 JSON 对象。")
        return AgentOutput(**data)
    except (json.JSONDecodeError, ValidationError, ValueError) as exc:
        print(
            "[Agent] 解析模型输出失败，将回退到失败动作。"
            f"错误: {exc}\n原始输出: {raw_text}"
        )
        return AgentOutput(
            thought="解析失败，回退到 execute -1",
            next_decision="execute",
            action_id=-1,
        )


def _parse_clarify_output(raw_resp: AIMessage) -> tuple[str | None, str]:
    """Parse clarify model output (thought + question JSON); fallback to raw text."""
    raw_text = str(getattr(raw_resp, "content", "")).strip()
    try:
        data = json.loads(raw_text)
        if isinstance(data, dict):
            thought = data.get("thought")
            question = data.get("question")
            if not isinstance(thought, str):
                thought = None
            if isinstance(question, str) and question.strip():
                return thought, question.strip()
    except Exception:
        pass
    # 兜底：直接把输出当作问题
    return None, raw_text


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

    messages = [SystemMessage(content=system_prompt), *state.messages]
    raw_resp = invoke_with_retry(llm, messages)
    resp = _safe_agent_output(raw_resp)

    # 如果模型没有给出合法的 action_id（例如为 null 或不在 1〜40 范围内），
    # 就回退到失败动作 -1，避免后续节点报错。
    if isinstance(resp.action_id, int) and 1 <= resp.action_id <= 40:
        action_id = resp.action_id
    else:
        action_id = -1

    # 根据当前 mode 决定是否允许 clarify：
    # - baseline / reflect: 使用模型预测的 clarify / execute
    # - cot / direct: 强制只执行 execute（纯 CoT / 纯 direct 一步到位）
    allow_clarify = state.mode in {"baseline", "reflect"}
    if allow_clarify and state.clarify_quota_left is not None:
        allow_clarify = state.clarify_quota_left > 0

    next_decision = resp.next_decision
    if not allow_clarify:
        next_decision = "execute"

    clarify_quota_left = state.clarify_quota_left

    new_messages = state.messages
    if next_decision == "clarify" and resp.thought:
        # 将 agent 的明確化意图显式写入对话，供 clarify 节点参考
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

    resp = invoke_with_retry(llm, messages)
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

    resp = invoke_with_retry(user_llm, messages)

    return {
        "messages": state.messages + [HumanMessage(content=resp.content)],
    }

# ========== Reflection Node（明確化質問後のまとめ） ==========

def reflection_node(state: AgentState) -> AgentState:
    if not state.enable_reflection:
        return {"messages": state.messages}

    summary = summarize_clarification(state)
    if not summary:
        return {"messages": state.messages}

    reflection_msg = "【明確化質問後のメモ】\n" + summary

    return {
        "messages": state.messages + [SystemMessage(content=reflection_msg)],
        "next_decision": None,
        "action_id": None,
    }


# ========== Execute Node (END) ==========

def execute_node(state: AgentState) -> AgentState:

    # 如果 action_id 异常（None 等），回退到失败动作 -1
    action_id = state.action_id if state.action_id in id_to_action else -1
    desc = id_to_action[action_id]
    result_text = f"{desc}を执行しました。"

    return {
        "messages": state.messages + [AIMessage(content=result_text)],
    }


# ========== Build Graph ==========

builder = StateGraph(AgentState)

builder.add_node("agent", agent)
builder.add_node("clarify", clarify_node)
builder.add_node("user_sim", user_sim_node)
builder.add_node("clarify_reflection", reflection_node)
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
builder.add_edge("user_sim", "clarify_reflection")
builder.add_edge("clarify_reflection", "agent")

builder.add_edge("execute", END)

graph = builder.compile()
