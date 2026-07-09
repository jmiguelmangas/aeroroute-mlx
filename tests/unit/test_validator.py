from aeroroute_mlx.validator import (
    MAX_EXPLANATION_TEXT_LENGTH,
    validate_numeric_claims,
    validate_operational_claims,
    validate_text_length,
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


def test_rejects_dispatchable_claim() -> None:
    assert not validate_operational_claims(
        "This trajectory is dispatchable as-is."
    )


def test_rejects_atc_compliant_claim() -> None:
    assert not validate_operational_claims(
        "The route is fully ATC-compliant."
    )


def test_accepts_text_within_max_length() -> None:
    assert validate_text_length("x" * MAX_EXPLANATION_TEXT_LENGTH)


def test_rejects_text_over_max_length() -> None:
    assert not validate_text_length("x" * (MAX_EXPLANATION_TEXT_LENGTH + 1))
