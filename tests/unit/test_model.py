import json
from pathlib import Path

import pytest

from aeroroute_mlx.model import ModelManifest


def test_manifest_requires_pinned_text_only_local_model(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "base_model": "mlx-community/gemma-3-text-4b-it-4bit",
                "architecture": "gemma3_text",
                "modality": "text",
                "quantization": "mlx-4bit",
                "base_revision": "4f665a4c50ecfe4ecdc34056ab52fe3e3c4abf9e",
                "local_path": "model",
                "application_context_limit": 4096,
            }
        )
    )

    manifest = ModelManifest.load(manifest_path)

    assert manifest.local_path == model
    assert manifest.base_revision.startswith("4f665a4c")


def test_manifest_rejects_unpinned_revision(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "base_model": "model",
                "architecture": "gemma3_text",
                "modality": "text",
                "quantization": "mlx-4bit",
                "base_revision": "main",
                "local_path": "model",
                "application_context_limit": 4096,
            }
        )
    )

    with pytest.raises(ValueError, match="immutable commit SHA"):
        ModelManifest.load(manifest_path)
