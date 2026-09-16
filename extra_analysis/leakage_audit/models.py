from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


LABELS = (
    "NO_TARGET_SIGNAL",
    "LEGITIMATE_INTENT_SIGNAL",
    "NEAR_DIRECT_TARGET_EXPOSURE",
    "DIRECT_TARGET_LEAKAGE",
)


@dataclass(frozen=True)
class ClarificationRecord:
    record_id: str
    instance_id: str
    source: str
    gold_action: str
    clarification_question: str
    response: str
    original_utterance: str | None = None
    participant_id: str | None = None
    original_condition: str | None = None
    trial: int | None = None
    clarification_step: int | None = None
    step_t: int | None = None
    sample_index: int | None = None
    model_dir: str | None = None
    log_path: str | None = None

    def metadata(self) -> dict[str, Any]:
        return asdict(self)

    def judge_payload(self) -> dict[str, str]:
        payload = {
            "original_user_utterance": self.original_utterance or "",
            "clarification_question": self.clarification_question or "",
            "simulated_or_roleplayed_user_response": self.response or "",
            "gold_action": self.gold_action or "",
        }
        return payload

