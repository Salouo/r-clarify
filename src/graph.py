from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_openai import ChatOpenAI

from .state import ClarifyState, DecisionType, id_to_action, RouterOutput, RouterStep

load_dotenv(".env", override=True)

# Main agent (router / clarify / user_sim / execute)
llm = ChatOpenAI(model="gpt-5.1")

# User simulator
user_llm = ChatOpenAI(model="gpt-5.1")


# ========== Router Node ==========

def router(state: ClarifyState) -> ClarifyState:
    """
    Read current conversation + environment + user feedback.
    Decide: clarify or execute.
    If execute, also select action_id.
    """

    # Simple step limit: if max_steps exceeded, force execute failure action -1
    if state.step >= state.max_steps:
        return {
            "next_decision": "execute",
            "action_id": -1,
            "step": state.step + 1,
            "trace": state.trace + [RouterStep(
                step=state.step,
                thought="Exceeded maximum steps. Forcing failure action -1.",
                next_decision="execute",
                action_id=-1,
            )]
        }

    system_prompt = """
    你是一个家庭 Agent ，能够基于用户的情况执行反思性行动，即使用户的指令不清楚也没关系。
    反思性行动是指能够满足用户潜在需求而非明确要求的有益行为。
    在下面40个可用行动中，恰好有一个反思性行动能够满足用户的潜在意图。请仔细考虑。

    重要提示：你对最佳行动的判断可能与用户真正想要的不一致。在做决定前请仔细思考。
    例如：如果用户饿了，他们可能需要零食，但实际上他们可能更想要饮料。你必须通过澄清来发现他们的真实意图，做出正确决定。

    【過去の試行の振り返り】
    - 在对话历史中，你可能会看到一段以「【過去の試行の振り返り】」开头的文本。
    - 那是你在相似情境下多次尝试后的总结，里面包含了你今后应该遵守的经验/教训。
    - 你在做 thought 和决策时，必须参考并尽量遵守这些反思要点。

    你的目标是用最少的步数发现用户的潜在意图，然后执行那个反思性行动。

    你有2个选项可以选择：

    1. "clarify"（提出澄清问题）：
    - 当你认为当前信息不足以确定应该执行哪个行动，直接执行可能出错时，提出1个澄清问题。
    - 如果用户回答"我不知道怎么回答"，意味着你的问题没有帮助或帮助很小以至于无法揭示他们的意图。
    - 你可以直接询问用户是否需要这个反思性行动。

    2. "execute"（执行1-40中的一个特定行动）：

    [1]バナナを持ってくる
    [2]充電ケーブルを持ってくる
    [3]コップを持ってくる
    [4]ケチャップを持ってくる
    [5]宅配便を持ってくる
    [6]ペットボトルを持ってくる
    [7]リモコンを持ってくる
    [8]スマホを持ってくる
    [9]お菓子を持ってくる
    [10]ティッシュ箱を持ってくる
    [11]充電ケーブルを片付ける
    [12]コップを片付ける
    [13]ケチャップを片付ける
    [14]ミニカーを片付ける
    [15]ペットボトルを片付ける
    [16]リモコンを片付ける
    [17]スマホを片付ける
    [18]お菓子を片付ける
    [19]ティッシュ箱を片付ける
    [20]ゴミをゴミ箱に入れる
    [21]缶切りを持ってくる
    [22]クッキングシートを持ってくる
    [23]グラスを持ってくる
    [24]おろし器を持ってくる
    [25]キッチンペーパーを持ってくる
    [26]レモンを持ってくる
    [27]オリーブオイルを持ってくる
    [28]じゃがいもを持ってくる
    [29]サランラップを持ってくる
    [30]水筒を持ってくる
    [31]缶切りを棚にしまう
    [32]クッキングシートを棚にしまう
    [33]グラスを棚にしまう
    [34]おろし器を棚にしまう
    [35]キッチンペーパーを棚にしまう
    [36]ペットボトルを冷蔵庫にしまう
    [37]サランラップを棚にしまう
    [38]タッパーをレンジに入れる
    [39]タッパーを冷蔵庫にしまう
    [40]水筒を棚にしまう

    仅输出以下格式的JSON对象（无需其他文本）：
    {
    "thought": "<你当前的推理，2-3句话>",
    "next_decision": "clarify" 或 "execute",
    "action_id": <当next_decision为"execute"时为1-40的整数，否则为null>
    }
    """

    messages = [SystemMessage(content=system_prompt), *state.messages]
    resp: RouterOutput = llm.with_structured_output(RouterOutput).invoke(messages)
    return {
        "next_decision": resp.next_decision,
        "action_id": resp.action_id,
        "step": state.step+1,
        "trace": state.trace + [RouterStep(
            step=state.step+1,
            thought=resp.thought,
            next_decision=resp.next_decision,
            action_id=resp.action_id,
        )],
    }


# ========== Clarify Node ==========

def clarify_node(state: ClarifyState) -> ClarifyState:
    """
    Router chose clarify:
    - Generate 1 high-value clarification question (AI speaks)
    """
    system_prompt = """
    Router 已为你设定决策为"澄清"。

    你的目标是提出1个澄清问题，能够显著减少对用户潜在需求的误解。
    - 保持问题简洁，以获得更好的用户体验。
    - 满足用户潜在意图的行动必须是下面列出的40个反思性行动之一，所以你的澄清问题应该保持专注。
    - 澄清问题要符合日常对话形式，而不是像在聊天框聊天一样。不要含有括号，序号①，等文本说明性信息。

    直接输出自然语言供用户查看。不需要JSON格式。
    注意：你的输出不需要换行。
    """

    messages = [SystemMessage(content=system_prompt), *state.messages]
    resp = llm.invoke(messages)

    return {
        "messages": state.messages + [AIMessage(content=resp.content)],
        "next_decision": None,
        "action_id": None,
    }


# ========== User Simulation Node ==========

def user_sim_node(state: ClarifyState) -> ClarifyState:
    """
    Simulate a user with the goal of action label_action_id answering the clarification question.
    - Expose only label_action_id, original_utterance, and env in system prompt.
    - For conversation, only show the user_llm the last clarification question; they can't see prior chat history.
    """
    action_id = state.label_action_id
    action_desc = id_to_action.get(action_id, f"アクション {action_id}")

    # Find the last AIMessage as the "clarification question"
    last_ai_msg = None
    for msg in reversed(state.messages):
        if isinstance(msg, AIMessage):
            last_ai_msg = msg
            break

    system_prompt = f"""
    你现在要扮演一个【真实用户】角色，而不是智能体或系统。

    你的“真实需求”（不要说出口）是：希望 Agent 最终帮你执行：
    「{action_desc}」

    你一开始对 Agent 说的话是：
    「{state.original_utterance}」

    当前你所在场景的客观环境信息是：
    {state.env}

    现在，Agent 刚刚向你提出了一些澄清问题。
    请你继续作为同一个用户，在这个场景中，自然、简洁地回答这些问题。

    对话时请遵守以下规则：

    1. 不要暴露你的真实需求
    - 不要说“我想让你执行某个动作”“我的目标是让你做××”之类的话。
    - 只能像普通用户一样，围绕你看到的物品、遇到的不便、想解决的问题来表达。

    2. 像一个真实的人说话
    - 用自然口语，通常 1～2 句话即可。
    - 不要提到“环境信息”“动作编号”“Agent”“系统提示”等技术性词汇。
    - 不要复述本提示的内容。

    3. 决定容器“空 / 有液体”的方式
    - 当 Agent 问你某个容器（杯子、水壶、塑料瓶等）是不是空的时候：
    * 先根据【客观环境信息】描述的情景，想一想在这个场景下，这个容器是“空”还是“有液体”更自然；
    * 再考虑哪种回答更有助于你最终得到「{action_desc}」这样的帮助；
    * 然后给出一个自然的回答，例如：“已经没了”“还有一些”“是满的”等。
    - 不要说“因为我想让你做××所以说它是空的”这种元信息，只给出结果。

    4. 面对几乎无关的问题
    - 如果你觉得 Agent 当前的澄清问题：
    * 和你的真实需求几乎无关，且
    * 换成现实中，一个普通用户也会觉得“很难回答”或“不知道怎么答”，
    那就直接回答：**“我不知道怎么回答”**。

    5. 保持场景的一致性
    - 你的回答要尽量与之前的对话内容以及 {state.env} 保持一致，不要自相矛盾。
    - 如果之前已经暗示过某个容器是空的/有水的，后续也尽量保持一致；
    只有在你明确说明发生了变化时才改变（例如：“我刚又装满了水”）。

    现在，请你以上述规则，站在“真正需要这个行动的用户”的立场上，简洁自然地回答 Agent 刚才的问题。
    注意：你的回复不需要换行。
    """

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=last_ai_msg.content),
    ]

    resp = user_llm.invoke(messages)

    return {
        "messages": state.messages + [HumanMessage(content=resp.content)],
    }


# ========== Execute Node (END) ==========

def execute_node(state: ClarifyState) -> ClarifyState:
    """
    Router chose execute:
    - Read state.action_id (1-40)
    - Call real robot interface to execute this action (print for now)
    - Write execution result to messages (AI speaks), then END
    """

    desc = id_to_action[state.action_id]
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
