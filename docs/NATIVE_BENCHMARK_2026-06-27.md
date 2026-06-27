# Gemma 3 4B native MLX benchmark

Date: 2026-06-27

## Environment

- Mac mini with Apple M4, 10 CPU cores and 16 GB unified memory
- macOS 26.5.1 arm64
- MLX 0.31.2 and MLX-LM 0.31.3
- `mlx-community/gemma-3-text-4b-it-4bit`
- revision `4f665a4c50ecfe4ecdc34056ab52fe3e3c4abf9e`
- local quantized weights: 2.4 GB

## Result

Three bounded generations used the same representative AeroRoute explanation
request. All three returned the required JSON schema, introduced no unsupported
numeric claims, contained no banned operational claim and passed the service
contract without fallback.

| Metric | Result |
| --- | ---: |
| Contract pass rate | 3/3 |
| Cold latency | 6.015 s |
| Warm latency | 1.335 s, 1.322 s |
| Peak resident memory | 2,202.4 MB |

Reproduce with:

```bash
uv sync --all-groups --extra mlx
uv run python scripts/benchmark_mlx.py configs/gemma3-4b-smoke.json --runs 3
```

## Decision

The 4B checkpoint is suitable as the lower-memory local fallback on this Mac.
This does not promote Gemma 3 12B: its local download is incomplete, it has not
been benchmarked, and Gemma terms acceptance has not been recorded. The 4B
result is a compatibility smoke benchmark, not a model-quality evaluation.
