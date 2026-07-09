import json
import logging
import time

from fastapi.testclient import TestClient

from aeroroute_mlx.generation import GenerationSettings
from aeroroute_mlx.main import create_app
from aeroroute_mlx.validator import MAX_EXPLANATION_TEXT_LENGTH


class FakeProvider:
    name = "fake-mlx"

    def __init__(self, output: str, delay_s: float = 0) -> None:
        self.output = output
        self.delay_s = delay_s

    def generate(self, prompt: str, *, max_tokens: int) -> str:
        time.sleep(self.delay_s)
        assert "Return exactly one JSON object" in prompt
        assert max_tokens == 64
        return self.output


def _request(
    provider: FakeProvider, *, timeout_s: float = 1
) -> dict[str, object]:
    response = TestClient(
        create_app(
            provider,
            GenerationSettings(timeout_s=timeout_s, max_tokens=64),
        )
    ).post(
        "/v1/explanations",
        json={
            "contract_version": "1.0.0",
            "summary": "Fuel is 1200 kg.",
            "allowed_numeric_values": ["1200"],
        },
    )
    assert response.status_code == 200
    return response.json()


def test_valid_structured_generation_is_returned() -> None:
    payload = _request(FakeProvider('{"text":"Fuel is 1200 kg."}'))

    assert payload["provider"] == "fake-mlx"
    assert payload["fallback_used"] is False


def test_malformed_json_falls_back() -> None:
    payload = _request(FakeProvider("not json"))

    assert payload["provider"] == "template"
    assert payload["fallback_used"] is True


def test_unsupported_numeric_claim_falls_back() -> None:
    payload = _request(FakeProvider('{"text":"Fuel is 999 kg."}'))

    assert payload["provider"] == "template"


def test_operational_claim_falls_back() -> None:
    payload = _request(
        FakeProvider('{"text":"This is a safe route using 1200 kg."}')
    )

    assert payload["provider"] == "template"


def test_timeout_falls_back() -> None:
    payload = _request(
        FakeProvider('{"text":"Fuel is 1200 kg."}', delay_s=0.05),
        timeout_s=0.001,
    )

    assert payload["provider"] == "template"


def test_overlong_text_falls_back() -> None:
    overlong = "x" * (MAX_EXPLANATION_TEXT_LENGTH + 1)
    payload = _request(FakeProvider(json.dumps({"text": overlong})))

    assert payload["provider"] == "template"
    assert payload["fallback_used"] is True


def test_fallback_logs_the_cause(caplog) -> None:
    with caplog.at_level(logging.WARNING, logger="aeroroute.mlx"):
        _request(FakeProvider("not json"))

    records = [r for r in caplog.records if r.name == "aeroroute.mlx"]
    assert records, "expected a fallback log line from aeroroute.mlx"
    payload = json.loads(records[-1].message)
    assert payload["event"] == "mlx_generation_fallback"
    assert "ValueError" in payload["reason"]
