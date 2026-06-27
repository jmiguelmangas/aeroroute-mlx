from aeroroute_mlx.validator import (
    validate_numeric_claims,
    validate_operational_claims,
)


def test_accepts_only_supplied_numeric_claims() -> None:
    assert validate_numeric_claims(
        "Fuel is 1200 kg and time is 90 min.", ["1200", "90"]
    )


def test_rejects_unsupported_numeric_claim() -> None:
    assert not validate_numeric_claims("Fuel is 999 kg.", ["1200"])


def test_rejects_operational_claim() -> None:
    assert not validate_operational_claims(
        "This safe route guarantees safety and ATC clearance."
    )
