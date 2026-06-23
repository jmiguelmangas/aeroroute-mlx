"""Constrained prompt construction for the optional local model."""

from aeroroute_mlx.contracts import ExplanationRequest


def build_prompt(request: ExplanationRequest) -> str:
    allowed = ", ".join(request.allowed_numeric_values) or "no numeric claims"
    return (
        "Explain only the supplied deterministic result. Do not calculate, rank, "
        "or add facts. Allowed numeric values: "
        f"{allowed}.\n\nDeterministic summary:\n{request.summary}"
    )
