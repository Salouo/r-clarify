from __future__ import annotations

from .models import ClarificationRecord


JUDGE_SYSTEM_PROMPT = """You are auditing one clarification response for potential label leakage in role-played user answers.

Classify the response into exactly one label:
1. NO_TARGET_SIGNAL
2. LEGITIMATE_INTENT_SIGNAL
3. NEAR_DIRECT_TARGET_EXPOSURE
4. DIRECT_TARGET_LEAKAGE

Critical rule:
A precise agent question followed by a truthful user answer is not leakage. If the agent directly and appropriately asked for the user's true need, preference, desired object, desired state, or intended outcome, then a direct truthful answer is LEGITIMATE_INTENT_SIGNAL even when it names the exact target object or makes the Gold Action easy to infer.

Judge the answer together with the clarification question. Never classify leakage from semantic similarity to the Gold Action alone.

Label definitions:
- NO_TARGET_SIGNAL: The answer contains little or no useful information about the relevant intent or preference.
- LEGITIMATE_INTENT_SIGNAL: The answer gives useful intent, need, preference, desired object, desired category, or desired outcome information that is naturally justified by the question.
- NEAR_DIRECT_TARGET_EXPOSURE: The answer volunteers substantial gold-specific information that the question did not naturally require, effectively giving away most of the Gold Action.
- DIRECT_TARGET_LEAKAGE: The answer explicitly asks or instructs the robot/agent to perform the Gold Action, or very closely paraphrases the target action as a command/request.

Decision logic:
1. First ask whether the agent directly elicited the information contained in the response. If yes, prefer LEGITIMATE_INTENT_SIGNAL, even if the response fully reveals the user's intention.
2. Then ask whether the response explicitly commands or requests the Gold Action. If yes, use DIRECT_TARGET_LEAKAGE.
3. Otherwise ask whether the response volunteers substantial gold-specific information beyond what the question required. If yes, use NEAR_DIRECT_TARGET_EXPOSURE.
4. Otherwise, if the answer contains useful intent/preference information, use LEGITIMATE_INTENT_SIGNAL.
5. If it contains little or no useful target-related information, use NO_TARGET_SIGNAL.

Examples:
- Gold Action: Bring a banana. Question: Which fruit do you want? Answer: A banana. Label: LEGITIMATE_INTENT_SIGNAL.
- Gold Action: Bring a banana. Question: Do you want a banana? Answer: Yes. Label: LEGITIMATE_INTENT_SIGNAL.
- Gold Action: Put the water bottle away on the shelf. Question: Is anything bothering you? Answer: The water bottle belongs back on the shelf. Label: NEAR_DIRECT_TARGET_EXPOSURE.
- Gold Action: Bring a banana. Question: Would you like something to eat? Answer: Please bring me a banana. Label: DIRECT_TARGET_LEAKAGE.

Return strict JSON only, with exactly these keys:
{
  "label": "NO_TARGET_SIGNAL | LEGITIMATE_INTENT_SIGNAL | NEAR_DIRECT_TARGET_EXPOSURE | DIRECT_TARGET_LEAKAGE",
  "question_precisely_elicited_the_information": true,
  "answer_truthfully_responds_to_question": true,
  "target_object_explicitly_mentioned": false,
  "gold_action_explicitly_requested": false,
  "volunteered_gold_specific_information": false,
  "brief_reason": "one short sentence"
}

Do not output chain-of-thought. The brief_reason must be one short sentence."""


JUDGE_USER_TEMPLATE = """Original user utterance:
{original_user_utterance}

Clarification question:
{clarification_question}

User response to classify:
{simulated_or_roleplayed_user_response}

Gold Action:
{gold_action}

Return the JSON object only."""


def build_judge_messages(record: ClarificationRecord) -> list[dict[str, str]]:
    payload = record.judge_payload()
    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": JUDGE_USER_TEMPLATE.format(**payload)},
    ]


def judge_prompt_preview() -> str:
    return (
        "SYSTEM PROMPT:\n"
        f"{JUDGE_SYSTEM_PROMPT}\n\n"
        "USER PROMPT TEMPLATE:\n"
        f"{JUDGE_USER_TEMPLATE}"
    )
