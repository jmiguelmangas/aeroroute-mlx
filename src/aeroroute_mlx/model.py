"""Validated local-model configuration and lazy MLX-LM adapter."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Protocol, cast


@dataclass(frozen=True, slots=True)
class ModelManifest:
    base_model: str
    architecture: str
    modality: str
    quantization: str
    base_revision: str
    local_path: Path
    application_context_limit: int

    @classmethod
    def load(cls, path: Path) -> ModelManifest:
        payload = json.loads(path.read_text())
        manifest = cls(
            base_model=str(payload["base_model"]),
            architecture=str(payload["architecture"]),
            modality=str(payload["modality"]),
            quantization=str(payload["quantization"]),
            base_revision=str(payload["base_revision"]),
            local_path=(path.parent / str(payload["local_path"])).resolve(),
            application_context_limit=int(payload["application_context_limit"]),
        )
        manifest.validate()
        return manifest

    def validate(self) -> None:
        if self.architecture != "gemma3_text" or self.modality != "text":
            raise ValueError(
                "only the text-only Gemma 3 architecture is supported"
            )
        if self.quantization != "mlx-4bit":
            raise ValueError(
                "only the pinned MLX 4-bit quantization is supported"
            )
        if len(self.base_revision) != 40:
            raise ValueError("base revision must be an immutable commit SHA")
        if not 256 <= self.application_context_limit <= 4_096:
            raise ValueError("application context limit is outside policy")
        if not self.local_path.is_dir():
            raise ValueError("local model directory does not exist")


class TextGenerationProvider(Protocol):
    @property
    def name(self) -> str: ...

    def generate(self, prompt: str, *, max_tokens: int) -> str: ...


class MlxLmProvider:
    """Load a pinned local model on first use and generate bounded text."""

    def __init__(self, manifest: ModelManifest) -> None:
        self.manifest = manifest
        self._model: Any | None = None
        self._tokenizer: Any | None = None

    @property
    def name(self) -> str:
        return "mlx"

    def generate(self, prompt: str, *, max_tokens: int) -> str:
        self._load()
        chat_prompt = self._tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
        module = import_module("mlx_lm")
        generate = cast(Any, getattr(module, "generate"))
        return cast(
            str,
            generate(
                self._model,
                self._tokenizer,
                prompt=chat_prompt,
                max_tokens=max_tokens,
                verbose=False,
            ),
        )

    def _load(self) -> None:
        if self._model is not None:
            return
        module = import_module("mlx_lm")
        load = cast(Any, getattr(module, "load"))
        self._model, self._tokenizer = load(
            str(self.manifest.local_path), lazy=False
        )
