from __future__ import annotations

import re
from typing import Any

from .models import ClarificationRecord
from .utils import normalize_text, strip_action_id


COMMAND_PATTERNS = (
    r"\bplease\s+bring\b",
    r"\bbring\s+me\b",
    r"\bbring\b",
    r"\bplease\s+get\b",
    r"\bget\s+me\b",
    r"\bplease\s+put\b",
    r"\bput\b.+\baway\b",
    r"\bput\b.+\bin\b",
    r"\bput\b.+\bon\b",
    r"\bthrow\b.+\baway\b",
)

ACTION_VERBS = {
    "bring",
    "get",
    "fetch",
    "take",
    "put",
    "place",
    "store",
    "shelve",
    "throw",
    "discard",
    "remove",
}

STOPWORDS = {
    "a",
    "an",
    "and",
    "away",
    "back",
    "for",
    "in",
    "into",
    "me",
    "my",
    "on",
    "please",
    "the",
    "to",
    "up",
    "you",
}


def audit_record(record: ClarificationRecord) -> dict[str, Any]:
    response_norm = normalize_text(record.response)
    gold_norm = normalize_text(record.gold_action)
    response_tokens = set(response_norm.split())
    gold_tokens = set(gold_norm.split())
    target_terms = extract_target_terms(record.gold_action)

    exact_gold_action_phrase_overlap = bool(gold_norm and gold_norm in response_norm)
    target_object_explicitly_mentioned = any(term in response_norm for term in target_terms)
    action_verb_overlap = bool((gold_tokens & ACTION_VERBS) & response_tokens)
    obvious_robot_command = any(re.search(pattern, record.response or "", flags=re.IGNORECASE) for pattern in COMMAND_PATTERNS)

    return {
        "record_id": record.record_id,
        "instance_id": record.instance_id,
        "source": record.source,
        "exact_gold_action_phrase_overlap": exact_gold_action_phrase_overlap,
        "target_object_explicitly_mentioned": target_object_explicitly_mentioned,
        "action_verb_overlap": action_verb_overlap,
        "obvious_robot_command": obvious_robot_command,
        "target_terms": sorted(target_terms),
    }


def audit_records(records: list[ClarificationRecord]) -> list[dict[str, Any]]:
    return [audit_record(record) for record in records]


def extract_target_terms(gold_action: str) -> set[str]:
    text = normalize_text(strip_action_id(gold_action))
    if not text:
        return set()

    terms = {
        token
        for token in text.split()
        if len(token) > 2 and token not in STOPWORDS and token not in ACTION_VERBS
    }

    # Common Japanese action strings often put the object before particle "wo".
    raw = strip_action_id(gold_action)
    for particle in ("\u3092", "\u306b", "\u3078"):
        if particle in raw:
            candidate = normalize_text(raw.split(particle, 1)[0])
            if len(candidate) > 1:
                terms.add(candidate)
            break
    return terms

