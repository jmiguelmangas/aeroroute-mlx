"""Broader prompt-only quality corpus for the Gemma 3 4B explanation model.

`scripts/benchmark_mlx.py` is a 3-run compatibility smoke test with a single
synthetic request. This script is a model-*quality* pass: it builds a
diverse corpus of real optimization results from the live local API (varied
routes, aircraft, and profiles), reduces each to the exact `summary` +
`allowed_numeric_values` pair the API sends to this service in production
(mirroring `aeroroute_api.application.services.explanation`, duplicated here
deliberately -- this is a benchmarking script, not a shipped runtime
dependency, and aeroroute-mlx must not import aeroroute-api), runs each
through the real MLX provider once (model loaded once, reused across all
cases), and reports pass rate and latency distribution.

Requires a running local API (``make dev-up`` + seeded catalogue) for corpus
generation, and native MLX (``uv sync --extra mlx``) for generation.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aeroroute_mlx.contracts import ExplanationRequest
from aeroroute_mlx.generation import GenerationSettings, generate_explanation
from aeroroute_mlx.model import MlxLmProvider, ModelManifest

API_BASE_URL = "http://127.0.0.1:8000"

# (origin, destination, aircraft_type, profile) -- deliberately diverse:
# short/medium/long haul, all 6 supported aircraft types, all 3 profiles,
# a mix of the original 45-airport catalogue and the 30 airports added in
# the Phase 7 catalogue expansion (marked with a trailing comment).
CORPUS_ROUTES: tuple[tuple[str, str, str, str], ...] = (
    ("LEMD", "LFPG", "A320", "balanced"),
    ("EGLL", "EDDF", "A320", "minimum_time"),
    ("EHAM", "LOWW", "B738", "minimum_fuel"),
    ("LEMD", "EDDM", "A320", "balanced"),  # EDDM new
    ("LFPG", "LIMC", "B738", "minimum_time"),  # LIMC new
    ("EGLL", "EBBR", "A320", "minimum_fuel"),  # EBBR new
    ("LEMD", "KJFK", "B77W", "balanced"),
    ("KJFK", "LEMD", "B788", "minimum_fuel"),
    ("EDDF", "KATL", "B77W", "minimum_time"),  # KATL new
    ("EGLL", "KBOS", "B788", "balanced"),  # KBOS new
    ("LEMD", "MMMX", "A359", "minimum_fuel"),  # MMMX new
    ("KJFK", "CYUL", "A320", "minimum_time"),  # CYUL new
    ("OMDB", "LEMD", "A359", "balanced"),
    ("OMDB", "SKBO", "B77W", "minimum_time"),  # SKBO new
    ("OTHH", "HKJK", "B788", "minimum_fuel"),  # HKJK new
    ("HECA", "FAOR", "A359", "balanced"),  # FAOR new
    ("OMDB", "LTFM", "A320", "minimum_time"),  # LTFM new
    ("RJAA", "KSFO", "B788", "minimum_fuel"),
    ("RJTT", "ZBAA", "A359", "balanced"),  # ZBAA new
    ("RKSI", "WSSS", "B738", "minimum_time"),
    ("VHHH", "WMKK", "A320", "minimum_fuel"),  # WMKK new
    ("WSSS", "VTBS", "A320", "balanced"),  # VTBS new
    ("YSSY", "NZCH", "B738", "minimum_time"),  # NZCH new
    ("SBGR", "SBGL", "A320", "minimum_fuel"),  # SBGL new
)


@dataclass(frozen=True, slots=True)
class ExplanationFacts:
    """Mirrors aeroroute_api.application.services.explanation.ExplanationFacts."""

    origin_icao: str
    destination_icao: str
    profile: str
    distance_m: float
    time_s: float
    fuel_kg: float
    data_degraded: bool = False
    baseline_time_s: float | None = None
    baseline_fuel_kg: float | None = None


DEGRADED_CODES = {
    "WEATHER_FALLBACK",
    "WEATHER_STALE",
    "WEATHER_STILL_AIR",
    "FUEL_NOT_CONVERGED",
}


def facts_from_optimization(doc: dict[str, Any]) -> ExplanationFacts:
    request = doc["request"]
    winner = doc["winner"]
    baseline = doc.get("baseline")
    return ExplanationFacts(
        origin_icao=request["origin_icao"],
        destination_icao=request["destination_icao"],
        profile=request["profile"],
        distance_m=winner["distance_m"],
        time_s=winner["time_s"],
        fuel_kg=winner["fuel_kg"],
        data_degraded=any(
            flag["code"] in DEGRADED_CODES for flag in doc.get("data_quality", [])
        ),
        baseline_time_s=(baseline["time_s"] if baseline else None),
        baseline_fuel_kg=(baseline["fuel_kg"] if baseline else None),
    )


def render_deterministic_explanation(facts: ExplanationFacts) -> str:
    distance_km = facts.distance_m / 1_000
    time_minutes = facts.time_s / 60
    comparison = _comparison_text(facts)
    return (
        f"For the {facts.profile} profile, the selected synthetic trajectory "
        f"from {facts.origin_icao} to {facts.destination_icao} covers "
        f"{distance_km:.0f} km, takes an estimated {time_minutes:.0f} minutes, "
        f"and uses {facts.fuel_kg:.0f} kg of modeled trip fuel. {comparison}"
        "This is an educational trajectory-efficiency estimate, not "
        "operational flight-planning advice."
    )


def allowed_numeric_values(facts: ExplanationFacts) -> list[str]:
    values = {
        f"{facts.distance_m / 1_000:.0f}",
        f"{facts.time_s / 60:.0f}",
        f"{facts.fuel_kg:.0f}",
    }
    if facts.baseline_time_s is not None:
        values.add(f"{abs(facts.time_s - facts.baseline_time_s) / 60:.0f}")
    if facts.baseline_fuel_kg is not None:
        values.add(f"{abs(facts.fuel_kg - facts.baseline_fuel_kg):.0f}")
    return sorted(values)


def _comparison_text(facts: ExplanationFacts) -> str:
    if facts.baseline_fuel_kg is None or facts.baseline_time_s is None:
        return ""
    fuel_delta = facts.fuel_kg - facts.baseline_fuel_kg
    time_delta_minutes = (facts.time_s - facts.baseline_time_s) / 60
    fuel_text = _delta_text(fuel_delta, "kg of modeled trip fuel")
    time_text = _delta_text(time_delta_minutes, "minutes")
    return f"Compared with the baseline, it {fuel_text} and {time_text}. "


def _delta_text(delta: float, unit: str) -> str:
    rounded = abs(delta)
    if rounded < 0.5:
        return f"has a negligible difference in {unit}"
    if delta < 0:
        return f"saves {rounded:.0f} {unit}"
    return f"uses {rounded:.0f} more {unit}"


def fetch_optimization(
    origin: str, destination: str, aircraft: str, profile: str
) -> dict[str, Any]:
    payload = json.dumps(
        {
            "origin_icao": origin,
            "destination_icao": destination,
            "aircraft_type": aircraft,
            "profile": profile,
        }
    ).encode()
    request = urllib.request.Request(
        f"{API_BASE_URL}/api/v1/optimizations",
        data=payload,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round(quantile * (len(ordered) - 1))))
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    manifest = ModelManifest.load(args.manifest.resolve())
    provider = MlxLmProvider(manifest)
    settings = GenerationSettings(timeout_s=60, max_tokens=180)

    cases: list[dict[str, Any]] = []
    for index, (origin, destination, aircraft, profile) in enumerate(
        CORPUS_ROUTES, start=1
    ):
        case_id = f"{origin}-{destination}-{aircraft}-{profile}"
        print(f"[{index}/{len(CORPUS_ROUTES)}] {case_id} ...", flush=True)
        try:
            doc = fetch_optimization(origin, destination, aircraft, profile)
        except Exception as error:  # noqa: BLE001 -- record and continue
            cases.append(
                {"case_id": case_id, "api_error": str(error), "passed": False}
            )
            continue
        facts = facts_from_optimization(doc)
        summary = render_deterministic_explanation(facts)
        numeric_values = allowed_numeric_values(facts)
        request = ExplanationRequest(
            contract_version="1.0.0",
            summary=summary,
            allowed_numeric_values=numeric_values,
        )
        started = time.perf_counter()
        response = asyncio.run(generate_explanation(request, provider, settings))
        elapsed_s = time.perf_counter() - started
        cases.append(
            {
                "case_id": case_id,
                "data_degraded": facts.data_degraded,
                "fallback_used": response.fallback_used,
                "provider": response.provider,
                "latency_s": round(elapsed_s, 3),
                "passed": not response.fallback_used,
                "text": response.text,
            }
        )

    evaluated = [case for case in cases if "latency_s" in case]
    passed = [case for case in evaluated if case["passed"]]
    latencies = [case["latency_s"] for case in evaluated]
    report = {
        "model": manifest.base_model,
        "revision": manifest.base_revision,
        "total_cases": len(cases),
        "api_errors": len(cases) - len(evaluated),
        "evaluated_cases": len(evaluated),
        "passed_cases": len(passed),
        "pass_rate": round(len(passed) / len(evaluated), 4) if evaluated else 0.0,
        "latency_p50_s": round(percentile(latencies, 0.5), 3) if latencies else None,
        "latency_p95_s": round(percentile(latencies, 0.95), 3) if latencies else None,
        "cases": cases,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
