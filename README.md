# aeroroute-mlx

Native macOS MLX explanation service. It owns model lifecycle, constrained
generation, numeric-claim validation, and deterministic fallback. It does not
calculate or rank trajectories, access the database, or train models.

The default process uses deterministic fallback and does not import MLX. To run
the validated local Gemma 3 4B checkpoint on Apple Silicon:

```bash
uv sync --all-groups --extra mlx
find .venv -name '._*' -delete   # only needed on an external-volume workspace, see below
AEROROUTE_MLX_ENABLED=1 \
  AEROROUTE_MLX_MODEL_MANIFEST=./configs/gemma3-4b-smoke.json \
  uv run uvicorn aeroroute_mlx.main:app --host 127.0.0.1 --port 8765
```

If this checkout lives on a volume that does not natively support macOS
resource forks (e.g. an external exFAT-formatted drive), `uv sync --extra mlx`
mirrors every file `transformers` installs with a `._<name>` AppleDouble
sidecar. `transformers`' dynamic module discovery then tries to read one of
those sidecars as UTF-8 source and raises `UnicodeDecodeError` on model load.
Because `generate_explanation()` intentionally catches broad exceptions to
degrade to the template provider, this failure is silent: the service keeps
responding with `provider: "template"` and `fallback_used: true` and nothing
surfaces the real cause. Run the `find .venv -name '._*' -delete` step above
after every `uv sync --extra mlx` on such a workspace, or check
`fallback_used` in a response if native MLX explanations unexpectedly stop
appearing. See `docs/NATIVE_BENCHMARK_2026-07-09.md` for the full writeup.

Model weights remain local under `models/` and are never committed. See
`docs/NATIVE_BENCHMARK_2026-06-27.md` and `docs/NATIVE_BENCHMARK_2026-07-09.md`
for the recorded M4 smoke results.
