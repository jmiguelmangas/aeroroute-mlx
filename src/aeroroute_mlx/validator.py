"""Reject model text that introduces unsupported numeric claims."""

from __future__ import annotations

import re

_NUMERIC_TOKEN = re.compile(r"[-+]?\d+(?:[.,]\d+)?%?")
_BANNED_OPERATIONAL_CLAIMS = (
    "atc clearance",
    "cleared for",
    "guarantees safety",
    "operational flight plan",
    "safe route",
)


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
