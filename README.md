# aeroroute-mlx

Native macOS MLX explanation service. It owns model lifecycle, constrained
generation, numeric-claim validation, and deterministic fallback. It does not
calculate or rank trajectories, access the database, or train models.

The default process uses deterministic fallback and does not import MLX. To run
the validated local Gemma 3 4B checkpoint on Apple Silicon:

```bash
uv sync --all-groups --extra mlx
AEROROUTE_MLX_ENABLED=1 \
  AEROROUTE_MLX_MODEL_MANIFEST=./configs/gemma3-4b-smoke.json \
  uv run uvicorn aeroroute_mlx.main:app --host 127.0.0.1 --port 8765
```

Model weights remain local under `models/` and are never committed. See
`docs/NATIVE_BENCHMARK_2026-06-27.md` for the recorded M4 smoke result.
