"""Constrained prompt construction for the optional local model."""

from aeroroute_mlx.contracts import ExplanationRequest


def build_prompt(request: ExplanationRequest) -> str:
    allowed = ", ".join(request.allowed_numeric_values) or "no numeric claims"
    return (
        "Return exactly one JSON object with this schema: "
        '{"text":"brief explanation"}. Explain only the supplied '
        "deterministic result. Do not calculate, rank, add facts, make safety "
        "claims, or mention ATC clearance. Allowed numeric values: "
        f"{allowed}.\n\nDeterministic summary:\n{request.summary}"
    )
