from typing import Any, Optional, Literal
from typing_extensions import Annotated
from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field


# 动作表放这里，供 graph / utils / main 共用
id_to_action: dict[int, str] = {
    -1: "動作を特定できない",
    1: "[1]バナナを持ってくる",
    2: "[2]充電ケーブルを持ってくる",
    3: "[3]コップを持ってくる",
    4: "[4]ケチャップを持ってくる",
    5: "[5]宅配便を持ってくる",
    6: "[6]ペットボトルを持ってくる",
    7: "[7]リモコンを持ってくる",
    8: "[8]スマホを持ってくる",
    9: "[9]お菓子を持ってくる",
    10: "[10]ティッシュ箱を持ってくる",
    11: "[11]充電ケーブルを片付ける",
    12: "[12]コップを片付ける",
    13: "[13]ケチャップを片付ける",
    14: "[14]ミニカーを片付ける",
    15: "[15]ペットボトルを片付ける",
    16: "[16]リモコンを片付ける",
    17: "[17]スマホを片付ける",
    18: "[18]お菓子を片付ける",
    19: "[19]ティッシュ箱を片付ける",
    20: "[20]ゴミをゴミ箱に入れる",
    21: "[21]缶切りを持ってくる",
    22: "[22]クッキングシートを持ってくる",
    23: "[23]グラスを持ってくる",
    24: "[24]おろし器を持ってくる",
    25: "[25]キッチンペーパーを持ってくる",
    26: "[26]レモンを持ってくる",
    27: "[27]オリーブオイルを持ってくる",
    28: "[28]じゃがいもを持ってくる",
    29: "[29]サランラップを持ってくる",
    30: "[30]水筒を持ってくる",
    31: "[31]缶切りを棚にしまう",
    32: "[32]クッキングシートを棚にしまう",
    33: "[33]グラスを棚にしまう",
    34: "[34]おろし器を棚にしまう",
    35: "[35]キッチンペーパーを棚にしまう",
    36: "[36]ペットボトルを冷蔵庫にしまう",
    37: "[37]サランラップを棚にしまう",
    38: "[38]タッパーをレンジに入れる",
    39: "[39]タッパーを冷蔵庫にしまう",
    40: "[40]水筒を棚にしまう",
}

action_to_id: dict[str, int] = {v: k for k, v in id_to_action.items()}

# 所有可用决策
DecisionType = Literal["clarify", "execute"]

# 每一步 agent 的输出
class AgentStep(BaseModel):
    step: int                       # 第几步
    thought: Optional[str] = None   # LLM 的思考（thinking），direct 模式下可以为 None
    next_decision: DecisionType     # clarify / execute
    action_id: Optional[int] = None # execute 时选的动作
    question: Optional[str] = None  # clarify 时由 agent 生成的问题文本
    answer: Optional[str] = None    # clarify 时用户的回答文本
    token_usage: Optional[dict] = None  # 本步 agent 模型调用的 token 消耗（prompt/completion/total）


class AgentOutput(BaseModel):
    thought: Optional[str] = Field(
        default=None,
        description="你对于当前局面，以及对于接下来应该如何决策的思考；direct 模式下可以为 null"
    )
    next_decision: DecisionType = Field(
        description="你接下来要执行的决策，只能是 clarify / execute 之一"
    )
    action_id: Optional[int] = Field(
        default=None,
        description="当你选择 next_decision 为 execute 时，必须是 1 到 40 的整数; clarify 时为 None",
    )
    question: Optional[str] = Field(
        default=None,
        description="当你选择 next_decision 为 clarify 时，必须填写 1 条澄清问题; execute 时为 null",
    )


# ========== Graph State ==========

class AgentState(BaseModel):

    # 对话历史：真实/模拟用户 + agent 的所有对话
    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)

    # agent 决定的下一步决策
    next_decision: Optional[DecisionType] = Field(default=None,)

    # 如果 next_decision == "execute"，这里是 1~40 的编号；否则就是null
    action_id: Optional[int] = Field(default=None)

    # 外部环境信息（sample 里的 env），现在用不用都行，先留着
    env: dict[str, Any] = Field(default_factory=dict)

    # 用户反馈（之后做 reflection 可以用，现在先不用）
    user_feedback: Optional[str] = Field(default=None)

    # 这个 sample 的真实目标动作（label）
    label_action_id: int = Field(default=None)

    # 原始模糊 utterance（方便在 user_sim 里用）
    original_utterance: str = Field(default="")

    # 当前运行模式（clarify / reflect / reflect_k / cot / cot_reflect / direct）
    mode: str = Field(
        default="clarify",
        description="当前样本使用的运行模式，例如 'clarify' / 'reflect' / 'reflect_k' / 'cot' / 'cot_reflect' / 'direct'",
    )

    # 步数
    step: int = Field(default=0, description="当前执行的 step 计数（从 0 开始）")

    # 最大步数
    max_steps: int = Field(default=1000,description="最大允许的 step 数，超过后应强制结束或执行")

    # 每一个 trial 的轨迹
    trace: list[AgentStep] = Field(default_factory=list, description="agent 每一步的思考和决策轨迹")

    # 是否在澄清后使用反思模块对用户回答进行整理
    enable_reflection: bool = Field(default=False)

    # 预先整理好的用户场景文本，供澄清后的反思使用
    user_context: str = Field(default="")

    # 澄清问题的额度（总数和剩余次数）；None 表示不限
    clarify_quota_total: Optional[int] = Field(
        default=None, description="本次 trial 允许澄清的最大次数（None 表示不限）"
    )
    clarify_quota_left: Optional[int] = Field(
        default=None, description="当前剩余的澄清次数（None 表示不限）"
    )

    # Agent 选择 clarify 时生成的问题，会由 clarify 节点直接记录
    clarification_question: Optional[str] = Field(
        default=None, description="agent 选择 clarify 时生成的一句话澄清问题"
    )
