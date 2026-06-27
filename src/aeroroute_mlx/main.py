"""Native explanation service; the deterministic solver is never invoked here."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI

from aeroroute_mlx.contracts import ExplanationRequest, ExplanationResponse
from aeroroute_mlx.generation import GenerationSettings, generate_explanation
from aeroroute_mlx.model import (
    MlxLmProvider,
    ModelManifest,
    TextGenerationProvider,
)


def create_app(
    provider: TextGenerationProvider | None = None,
    settings: GenerationSettings = GenerationSettings(),
) -> FastAPI:
    application = FastAPI(title="AeroRoute MLX", version="0.2.0")

    @application.get("/health")
    def health() -> dict[str, str]:
        return {
            "status": "ok",
            "provider": provider.name if provider is not None else "fallback",
        }

    @application.post("/v1/explanations", response_model=ExplanationResponse)
    async def explain(request: ExplanationRequest) -> ExplanationResponse:
        return await generate_explanation(request, provider, settings)

    return application


def provider_from_environment() -> TextGenerationProvider | None:
    if os.getenv("AEROROUTE_MLX_ENABLED", "0") != "1":
        return None
    manifest_path = Path(
        os.getenv("AEROROUTE_MLX_MODEL_MANIFEST", "model-manifest.json")
    )
    return MlxLmProvider(ModelManifest.load(manifest_path))


app = create_app(provider_from_environment())
