from aeroroute_mlx.fallback import render_fallback


def test_fallback_is_deterministic() -> None:
    assert render_fallback("tailwind reduced time") == (
        "Deterministic explanation: tailwind reduced time"
    )
