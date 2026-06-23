"""Typed cross-service boundary for explanation-only requests."""

from pydantic import BaseModel, Field


class ExplanationRequest(BaseModel):
    contract_version: str = Field(pattern=r"^1\.0\.0$")
    summary: str = Field(min_length=1, max_length=2_000)
    allowed_numeric_values: list[str] = Field(default_factory=list)


class ExplanationResponse(BaseModel):
    contract_version: str
    provider: str
    text: str
    fallback_used: bool
