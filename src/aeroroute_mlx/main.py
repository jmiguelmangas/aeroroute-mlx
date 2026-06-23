"""Native explanation service; the deterministic solver is never invoked here."""

from fastapi import FastAPI

from aeroroute_mlx.contracts import ExplanationRequest, ExplanationResponse
from aeroroute_mlx.fallback import render_fallback
from aeroroute_mlx.validator import validate_numeric_claims

app = FastAPI(title="AeroRoute MLX", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "provider": "fallback"}


@app.post("/v1/explanations", response_model=ExplanationResponse)
def explain(request: ExplanationRequest) -> ExplanationResponse:
    text = render_fallback(request.summary)
    if not validate_numeric_claims(text, request.allowed_numeric_values):
        text = "Deterministic explanation unavailable because numeric validation failed."
    return ExplanationResponse(
        contract_version=request.contract_version,
        provider="template",
        text=text,
        fallback_used=True,
    )
