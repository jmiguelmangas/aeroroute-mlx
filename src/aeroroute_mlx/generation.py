"""Structured generation orchestration with deterministic fallback."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass

from aeroroute_mlx.contracts import ExplanationRequest, ExplanationResponse
from aeroroute_mlx.fallback import render_fallback
from aeroroute_mlx.model import TextGenerationProvider
from aeroroute_mlx.prompt_builder import build_prompt
from aeroroute_mlx.validator import (
    validate_numeric_claims,
    validate_operational_claims,
)


@dataclass(frozen=True, slots=True)
class GenerationSettings:
    timeout_s: float = 30.0
    max_tokens: int = 180


async def generate_explanation(
    request: ExplanationRequest,
    provider: TextGenerationProvider | None,
    settings: GenerationSettings,
) -> ExplanationResponse:
    if provider is not None:
        try:
            raw = await asyncio.wait_for(
                asyncio.to_thread(
                    provider.generate,
                    build_prompt(request),
                    max_tokens=settings.max_tokens,
                ),
                timeout=settings.timeout_s,
            )
            text = parse_structured_text(raw)
            if validate_numeric_claims(
                text, request.allowed_numeric_values
            ) and validate_operational_claims(text):
                return ExplanationResponse(
                    contract_version=request.contract_version,
                    provider=provider.name,
                    text=text,
                    fallback_used=False,
                )
        except (Exception, asyncio.TimeoutError):
            pass
    return ExplanationResponse(
        contract_version=request.contract_version,
        provider="template",
        text=render_fallback(request.summary),
        fallback_used=True,
    )


def parse_structured_text(raw: str) -> str:
    candidate = raw.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        candidate = "\n".join(lines[1:-1]).strip()
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("model output is not a JSON object")
    payload = json.loads(candidate[start : end + 1])
    if set(payload) != {"text"} or not isinstance(payload["text"], str):
        raise ValueError("model output does not match the explanation schema")
    text = payload["text"].strip()
    if not text:
        raise ValueError("model explanation is empty")
    return text
