"""Optional local MLX adapter with deterministic fallback boundaries."""

from __future__ import annotations

import os
from functools import lru_cache

from aeroroute_mlx.contracts import ExplanationRequest

_SYSTEM_PROMPT = (
    "Choose the alternative with the lowest estimated fuel. State its route id, "
    "fuel and time only. Do not number or describe other alternatives. Do not "
    "give operational flight advice."
)


@lru_cache(maxsize=1)
def configured_adapter() -> "LocalAdapter | None":
    model_path = os.getenv("AEROROUTE_MLX_MODEL_PATH")
    adapter_path = os.getenv("AEROROUTE_MLX_ADAPTER_PATH")
    if not model_path or not adapter_path:
        return None
    return LocalAdapter(model_path, adapter_path)


class LocalAdapter:
    def __init__(self, model_path: str, adapter_path: str) -> None:
        self._model_path = model_path
        self._adapter_path = adapter_path
        self._loaded: tuple[object, object] | None = None

    def explain(self, request: ExplanationRequest) -> str:
        if self._loaded is None:
            from mlx_lm import load

            self._loaded = load(
                self._model_path, adapter_path=self._adapter_path
            )
        from mlx_lm import generate

        model, tokenizer = self._loaded
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": request.summary},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        return generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=100,
            verbose=False,
        )
