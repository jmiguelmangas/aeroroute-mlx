from fastapi.testclient import TestClient

from aeroroute_mlx.main import app


def test_explanation_endpoint_uses_deterministic_fallback() -> None:
    response = TestClient(app).post(
        "/v1/explanations",
        json={
            "contract_version": "1.0.0",
            "summary": "Fuel is 1200 kg.",
            "allowed_numeric_values": ["1200"],
        },
    )

    assert response.status_code == 200
    assert response.json()["provider"] == "template"
    assert response.json()["fallback_used"]
