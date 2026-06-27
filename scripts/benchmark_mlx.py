"""Run a bounded native MLX smoke benchmark and write machine-readable evidence."""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import resource
import time
from pathlib import Path

from aeroroute_mlx.contracts import ExplanationRequest
from aeroroute_mlx.generation import GenerationSettings, generate_explanation
from aeroroute_mlx.model import MlxLmProvider, ModelManifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest = ModelManifest.load(args.manifest.resolve())
    provider = MlxLmProvider(manifest)
    request = ExplanationRequest(
        contract_version="1.0.0",
        summary=(
            "The minimum-fuel route uses 69239 kg and is an educational "
            "synthetic trajectory only."
        ),
        allowed_numeric_values=["69239"],
    )
    timings: list[float] = []
    responses: list[dict[str, object]] = []
    for _ in range(args.runs):
        started = time.perf_counter()
        response = asyncio.run(
            generate_explanation(
                request,
                provider,
                GenerationSettings(timeout_s=180, max_tokens=120),
            )
        )
        timings.append(time.perf_counter() - started)
        responses.append(response.model_dump())
    report = {
        "model": manifest.base_model,
        "revision": manifest.base_revision,
        "platform": platform.platform(),
        "runs": args.runs,
        "cold_latency_s": round(timings[0], 3),
        "warm_latency_s": [round(value, 3) for value in timings[1:]],
        "peak_resident_memory_mb": round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024,
            1,
        ),
        "contract_passes": sum(
            not bool(response["fallback_used"]) for response in responses
        ),
        "responses": responses,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
