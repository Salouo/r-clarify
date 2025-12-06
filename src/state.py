from typing import Any, Optional, Literal
from typing_extensions import Annotated
from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field


# 所有可用决策
DecisionType = Literal["clarify", "execute"]

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

# 每一步 agent 的输出
class RouterStep(BaseModel):
    step: int                       # 第几步
    thought: str                    # LLM 的思考（thinking）
    next_decision: DecisionType     # clarify / execute
    action_id: Optional[int] = None # execute 时选的动作


class RouterOutput(BaseModel):
    thought: str = Field(
        description="你对于当前局面，以及对于接下来应该如何决策的仔细思考"
    )
    next_decision: DecisionType = Field(
        description="你接下来要执行的决策，只能是 clarify / execute 之一"
    )
    action_id: Optional[int] = Field(
        default=None,
        description="当你选择 next_decision 为 execute 时，必须是 1 到 40 的整数; clarify 时为 None",
    )


# ========== Graph State ==========

class ClarifyState(BaseModel):
    """Graph state for clarify/execute agent with simulated user."""

    # 对话历史：真实/模拟用户 + agent 的所有对话
    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)

    # router 决定的下一步决策
    next_decision: Optional[DecisionType] = Field(default=None,)

    # 如果 next_decision == "execute"，这里是 1~40 的编号；否则就是null
    action_id: Optional[int] = Field(default=None)

    # 外部环境信息（sample 里的 env），现在用不用都行，先留着
    env: dict[str, Any] = Field(default_factory=dict)

    # 用户反馈（之后做 reflection 可以用，现在先不用）
    user_feedback: Optional[str] = Field(
        default=None,
        description="用户对上一步行为的反馈，用于之后的 reflection",
    )

    # 这个 sample 的真实目标动作（label）
    label_action_id: int = Field(
        default=None,
        description="sample 的标注动作 ID（reflective action），只给 user_sim 看",
    )

    # 原始模糊 utterance（方便在 user_sim 里用）
    original_utterance: str = Field(
        default="",
        description="用户最初的模糊请求文本",
    )

    # 使用的 prompt 语言（ja / zh）
    lang: str = Field(
        default="zh",
        description="当前对话使用的提示语言，例如 'ja' 或 'zh'",
    )

    # 步数
    step: int = Field(
        default=0,
        description="当前执行的 step 计数（从 0 开始）",
    )

    # 最大步数
    max_steps: int = Field(
        default=1000,
        description="最大允许的 step 数，超过后应强制结束或执行",
    )

    trace: list[RouterStep] = Field(
        default_factory=list,
        description="router 每一步的思考和决策轨迹"
    )
