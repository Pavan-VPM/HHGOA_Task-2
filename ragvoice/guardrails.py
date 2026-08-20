from __future__ import annotations

from dataclasses import dataclass

from ragvoice.embed import tokenize


UNSAFE_TERMS = {
    "bomb",
    "kill",
    "murder",
    "explosive",
    "terrorist",
}


@dataclass
class GuardrailResult:
    allowed: bool
    reason: str


def check_query(question: str) -> GuardrailResult:
    tokens = set(tokenize(question))
    if not tokens:
        return GuardrailResult(False, "Please provide a spoken question or transcript.")
    if tokens & UNSAFE_TERMS:
        return GuardrailResult(False, "The request was blocked by the safety guardrail.")
    if len(tokens) < 3:
        return GuardrailResult(False, "The question is too short to retrieve grounded context reliably.")
    return GuardrailResult(True, "ok")


def grounded_enough(retrievals: list[dict]) -> bool:
    if not retrievals:
        return False
    return retrievals[0]["score"] >= 0.16
