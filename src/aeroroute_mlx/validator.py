"""Reject model text that introduces unsupported numeric claims."""

from __future__ import annotations

import re

_NUMERIC_TOKEN = re.compile(r"[-+]?\d+(?:[.,]\d+)?%?")


def validate_numeric_claims(text: str, allowed_values: list[str]) -> bool:
    allowed = {_normalize(value) for value in allowed_values}
    return all(
        _normalize(token) in allowed for token in _NUMERIC_TOKEN.findall(text)
    )


def _normalize(value: str) -> str:
    return value.strip().replace(",", ".").removesuffix(".0")
