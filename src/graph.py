from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_openai import ChatOpenAI

from .state import ClarifyState, id_to_action, RouterOutput, RouterStep
from .retry_utils import invoke_with_retry
from .token_usage import token_usage_callback
from .prompts import router_system_prompt, clarify_system_prompt, user_sim_system_prompt

load_dotenv(".env", override=True)

# Main agent (router / clarify / user_sim / execute)
llm = ChatOpenAI(
    model="gpt-4o-mini",
    callbacks=[token_usage_callback],
)

# User simulator
user_llm = ChatOpenAI(
    model="gpt-5.1",
    callbacks=[token_usage_callback]
)


# ========== Router Node ==========

def router(state: ClarifyState) -> ClarifyState:

    # Simple step limit: if max_steps exceeded, force execute failure action -1
    if state.step >= state.max_steps:
        return {
            "next_decision": "execute",
            "action_id": -1,
            "step": state.step + 1,
            "trace": state.trace + [RouterStep(
                step=state.step,
                thought="最大ステップ数を超過したため、失敗を判定します。",
                next_decision="execute",
                action_id=-1,
            )]
        }

    # 根据不同的 mode 选择不同的 prompt
    system_prompt = router_system_prompt(state.mode)

    messages = [SystemMessage(content=system_prompt), *state.messages]
    resp: RouterOutput = invoke_with_retry(
        llm.with_structured_output(RouterOutput),
        messages,
    )

    # 如果模型没有给出合法的 action_id（例如为 null 或不在 1〜40 范围内），
    # 就回退到失败动作 -1，避免后续节点报错。
    if isinstance(resp.action_id, int) and 1 <= resp.action_id <= 40:
        action_id = resp.action_id
    else:
        action_id = -1

    # 根据当前 mode 决定是否允许 clarify：
    # - baseline / reflect / reflect_k: 使用模型预测的 clarify / execute
    # - cot / direct: 强制只执行 execute（纯 CoT / 纯 direct 一步到位）
    allow_clarify = state.mode in {"baseline", "reflect", "reflect_k"}
    next_decision = resp.next_decision
    if not allow_clarify:
        next_decision = "execute"

    return {
        "next_decision": next_decision,
        "action_id": action_id,
        "step": state.step + 1,
        "trace": state.trace
        + [
            RouterStep(
                step=state.step + 1,
                thought=resp.thought,
                next_decision=next_decision,
                action_id=action_id,
            )
        ],
    }


# ========== Clarify Node ==========

def clarify_node(state: ClarifyState) -> ClarifyState:

    system_prompt = clarify_system_prompt()

    messages = [SystemMessage(content=system_prompt), *state.messages]
    resp = invoke_with_retry(llm, messages)

    return {
        "messages": state.messages + [AIMessage(content=resp.content)],
        "next_decision": None,
        "action_id": None,
    }


# ========== User Simulation Node ==========

def user_sim_node(state: ClarifyState) -> ClarifyState:

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
        original_utterance=state.original_utterance,
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


# ========== Execute Node (END) ==========

def execute_node(state: ClarifyState) -> ClarifyState:

    # 如果 action_id 异常（None 等），回退到失败动作 -1
    action_id = state.action_id if state.action_id in id_to_action else -1
    desc = id_to_action[action_id]
    result_text = f"{desc}を执行しました。"

    return {
        "messages": state.messages + [AIMessage(content=result_text)],
    }


# ========== Build Graph ==========

builder = StateGraph(ClarifyState)

builder.add_node("router", router)
builder.add_node("clarify", clarify_node)
builder.add_node("user_sim", user_sim_node)
builder.add_node("execute", execute_node)

builder.set_entry_point("router")

builder.add_conditional_edges(
    "router",
    lambda state: state.next_decision,
    {
        "clarify": "clarify",
        "execute": "execute",
    },
)

builder.add_edge("clarify", "user_sim")
builder.add_edge("user_sim", "router")

builder.add_edge("execute", END)

graph = builder.compile()
