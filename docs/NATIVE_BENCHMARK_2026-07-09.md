# Gemma 3 4B native MLX benchmark -- reproduction and a real bug found

Date: 2026-07-09

## Environment

- Same machine as the 2026-06-27 benchmark: Mac mini, Apple M4, 16 GB unified memory
- macOS 26.5.1 arm64
- MLX 0.31.2, MLX-LM 0.31.3, transformers 5.12.1 (freshly installed via
  `uv sync --all-groups --extra mlx`)
- `mlx-community/gemma-3-text-4b-it-4bit`, revision
  `4f665a4c50ecfe4ecdc34056ab52fe3e3c4abf9e`, local quantized weights 2.4 GB

## Bug found: silent fallback caused by AppleDouble sidecar files

The first benchmark run after a fresh `uv sync --all-groups --extra mlx`
returned `fallback_used: true` on all 3 runs with `peak_resident_memory_mb`
around 104 MB -- the model was never actually loaded, but
`generate_explanation()`'s broad `except (Exception, asyncio.TimeoutError):
pass` (intentional graceful-degradation behavior, see `generation.py`)
silently swallowed the real error and returned a template response instead.

Bypassing that handler and calling `MlxLmProvider._load()` directly surfaced
the real cause: `transformers`' dynamic module-discovery walks every entry
under `transformers/models/` and tried to read a `._dots1`-style AppleDouble
sidecar file as UTF-8 source, raising `UnicodeDecodeError`. This workspace
lives on an external volume that does not natively support macOS resource
forks, so macOS mirrors every file and directory the package installer
touches with a `._<name>` sidecar. `uv sync --extra mlx` had created 2,681
such sidecar files inside `transformers/models/` alone -- more than the
2,196 real `.py` files in that tree.

Fix applied for this session: `find .venv -name '._*' -delete` inside
`aeroroute-mlx/`, then the model loaded and generated correctly. This is a
local-environment issue, not a code or model defect, but it is worth
documenting because the existing fallback design means a broken native
provider fails *silently* -- the API keeps returning template explanations
with `provider: "template"` and no error surfaces anywhere unless someone
inspects `fallback_used` or reproduces the load outside the try/except. The
README now carries a one-line mitigation note for this workspace.

## Result (after the fix)

Three bounded generations used the same representative AeroRoute explanation
request as the 2026-06-27 benchmark. All three returned the required JSON
schema, introduced no unsupported numeric claims, contained no banned
operational claim, and passed the service contract without fallback.

| Metric | 2026-06-27 | 2026-07-09 |
| --- | ---: | ---: |
| Contract pass rate | 3/3 | 3/3 |
| Cold latency | 6.015 s | 6.023 s |
| Warm latency | 1.335 s, 1.322 s | 1.342 s, 1.341 s |
| Peak resident memory | 2,202.4 MB | 3,369.8 MB |

Latency reproduces closely. Peak RSS is higher this run, plausibly from the
newer `transformers`/`mlx-lm` releases pulled by `uv sync` (0.31.2/0.31.3
pinned either way) rather than a regression in the pinned model itself; still
well within the 16 GB budget on this Mac.

Reproduce with:

```bash
cd aeroroute-mlx
uv sync --all-groups --extra mlx
find .venv -name '._*' -delete   # only needed on this external-volume workspace
uv run python scripts/benchmark_mlx.py configs/gemma3-4b-smoke.json --runs 3
```

## Decision

Unchanged from 2026-06-27: the 4B checkpoint is the validated local
fallback and default on this Mac. See
`aeroroute-mlx-training/docs/COMPATIBILITY_12B_2026-07-09.md` for why the
12B checkpoint is not attempted as the primary model on this hardware.
