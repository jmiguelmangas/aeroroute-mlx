from fastapi.testclient import TestClient

from aeroroute_mlx import main


class _Adapter:
    def explain(self, request):
        return "Selected candidate: minimum_fuel. Fuel: 18000 kg. Time: 420 minutes."


def test_endpoint_uses_configured_adapter(monkeypatch) -> None:
    monkeypatch.setattr(main, "configured_adapter", lambda: _Adapter())
    response = TestClient(main.app).post(
        "/v1/explanations",
        json={
            "contract_version": "1.0.0",
            "summary": "synthetic comparison",
            "allowed_numeric_values": ["18000", "420"],
        },
    )
    assert response.json()["provider"] == "mlx"
    assert response.json()["fallback_used"] is False


def test_endpoint_falls_back_when_adapter_adds_numbers(monkeypatch) -> None:
    monkeypatch.setattr(
        main,
        "configured_adapter",
        lambda: type("Adapter", (), {"explain": lambda *_: "Fuel 999 kg."})(),
    )
    response = TestClient(main.app).post(
        "/v1/explanations",
        json={
            "contract_version": "1.0.0",
            "summary": "Fuel is 1200 kg.",
            "allowed_numeric_values": ["1200"],
        },
    )
    assert response.json()["provider"] == "template"
    assert response.json()["fallback_used"] is True
