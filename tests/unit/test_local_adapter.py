from aeroroute_mlx.local_adapter import configured_adapter


def test_adapter_is_disabled_without_local_paths(monkeypatch) -> None:
    monkeypatch.delenv("AEROROUTE_MLX_MODEL_PATH", raising=False)
    monkeypatch.delenv("AEROROUTE_MLX_ADAPTER_PATH", raising=False)
    configured_adapter.cache_clear()
    assert configured_adapter() is None
