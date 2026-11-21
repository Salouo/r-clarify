from typing import Any, Optional, Literal
from typing_extensions import Annotated
from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field


# Action IDs match the dataset labels. The sentinel -1 denotes a failed action.
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

DecisionType = Literal["clarify", "execute"]


class AgentStep(BaseModel):
    """One agent decision and its associated dialogue and token usage."""

    step: int
    thought: Optional[str] = None  # Direct mode may omit the reasoning.
    next_decision: DecisionType
    action_id: Optional[int] = None
    clarification_question: Optional[str] = None
    user_reply: Optional[str] = None
    token_usage: Optional[dict] = None


class AgentOutput(BaseModel):
    thought: Optional[str] = Field(
        default=None,
        description="Reasoning behind the next decision. Direct mode may omit this field."
    )
    next_decision: DecisionType = Field(
        description="The next decision, either clarify or execute."
    )
    action_id: Optional[int] = Field(
        default=None,
        description="Action ID from 1 to 40 for execution, or None for clarification.",
    )
    clarification_question: Optional[str] = Field(
        default=None,
        description="One question when requesting clarification, or None when executing.",
    )


class AgentState(BaseModel):
    """Dialogue, limits, and decision history for the current trial."""

    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)

    next_decision: Optional[DecisionType] = Field(default=None,)

    # Action IDs 1 through 40 select a candidate. The sentinel -1 records failure.
    action_id: Optional[int] = Field(default=None)

    # Scene annotations from the current sample.
    env: dict[str, Any] = Field(default_factory=dict)

    user_feedback: Optional[str] = Field(default=None)

    label_action_id: int = Field(default=None)

    original_utterance: str = Field(default="")

    mode: str = Field(
        default="clarify",
        description="Experiment mode for the current sample.",
    )

    step: int = Field(default=0, description="Number of steps completed in this trial, starting at zero.")

    max_steps: int = Field(default=1000, description="Step limit at which the agent must end the trial.")

    trace: list[AgentStep] = Field(default_factory=list, description="Agent reasoning and decisions recorded at each step.")

    # Action elimination excludes these IDs in subsequent trials.
    failed_action_ids: list[int] = Field(default_factory=list)

    enable_reflection: bool = Field(default=False)

    user_context: str = Field(default="")

    clarify_quota_total: Optional[int] = Field(
        default=None, description="Maximum clarification turns allowed in this trial. None means unlimited."
    )
    clarify_quota_left: Optional[int] = Field(
        default=None, description="Clarification turns remaining in this trial. None means unlimited."
    )

    clarification_question: Optional[str] = Field(
        default=None, description="Question generated when the agent requests clarification."
    )
