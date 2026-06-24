"""Native explanation service; the deterministic solver is never invoked here."""

from fastapi import FastAPI

from aeroroute_mlx.contracts import ExplanationRequest, ExplanationResponse
from aeroroute_mlx.fallback import render_fallback
from aeroroute_mlx.local_adapter import configured_adapter
from aeroroute_mlx.validator import validate_numeric_claims

app = FastAPI(title="AeroRoute MLX", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    provider = "mlx_adapter" if configured_adapter() else "fallback"
    return {"status": "ok", "provider": provider}


@app.post("/v1/explanations", response_model=ExplanationResponse)
def explain(request: ExplanationRequest) -> ExplanationResponse:
    text = render_fallback(request.summary)
    provider = "template"
    fallback_used = True
    adapter = configured_adapter()
    if adapter:
        try:
            text = adapter.explain(request)
            provider = "mlx"
            fallback_used = False
        except Exception:
            text = render_fallback(request.summary)
    if not validate_numeric_claims(text, request.allowed_numeric_values):
        text = "Deterministic explanation unavailable because numeric validation failed."
        provider = "template"
        fallback_used = True
    return ExplanationResponse(
        contract_version=request.contract_version,
        provider=provider,
        text=text,
        fallback_used=fallback_used,
    )
