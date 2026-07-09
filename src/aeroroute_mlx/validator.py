"""Reject model text that introduces unsupported numeric claims."""

from __future__ import annotations

import re

_NUMERIC_TOKEN = re.compile(r"[-+]?\d+(?:[.,]\d+)?%?")
# HLD SS13.1: MLX output must never state a route is legal, safe,
# dispatchable, or ATC-compliant.
_BANNED_OPERATIONAL_CLAIMS = (
    "atc clearance",
    "atc-compliant",
    "cleared for",
    "dispatchable",
    "guarantees safety",
    "operational flight plan",
    "safe route",
)

# HLD SS13.3 requires a maximum length as part of output validation. Real
# generated explanations in this session's quality corpus top out around
# 340 characters (docs/quality-corpus-2026-07-09.json); 1000 is generous
# enough to never trip on normal one-paragraph output while still catching
# a genuinely runaway/malformed generation.
MAX_EXPLANATION_TEXT_LENGTH = 1000


def validate_text_length(text: str) -> bool:
    return len(text) <= MAX_EXPLANATION_TEXT_LENGTH


def validate_numeric_claims(text: str, allowed_values: list[str]) -> bool:
    allowed = {_normalize(value) for value in allowed_values}
    return all(
        _normalize(token) in allowed for token in _NUMERIC_TOKEN.findall(text)
    )


def validate_operational_claims(text: str) -> bool:
    normalized = text.casefold()
    return not any(claim in normalized for claim in _BANNED_OPERATIONAL_CLAIMS)


def _normalize(value: str) -> str:
    return value.strip().replace(",", ".").removesuffix(".0")
