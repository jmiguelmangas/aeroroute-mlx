# Gemma 3 4B prompt-only quality corpus

Date: 2026-07-09
Machine: Mac mini, Apple M4, 16 GB unified memory (same as the compatibility
benchmarks)

## Purpose

`docs/NATIVE_BENCHMARK_2026-06-27.md` and `docs/NATIVE_BENCHMARK_2026-07-09.md`
are compatibility smoke tests: 3 runs of one synthetic request, proving the
service contract works at all. Neither is a model-quality signal. This corpus
is a first, broader quality pass on the prompt-only 4B baseline, per the
HLD SS13.9 evaluation intent (config A: base model plus strict prompt) --
still well short of the full held-out/gold-set evaluation HLD SS13.8-13.9
describe for a real promotion decision, but real signal beyond N=3.

## Method

`scripts/quality_corpus.py` builds 24 diverse `(origin, destination,
aircraft_type, profile)` cases -- short/medium/long haul, all 6 supported
aircraft types, all 3 optimization profiles, mixing the original 45-airport
catalogue with the 30 airports added in the Phase 7 catalogue expansion.
For each case it:

1. requests a real optimization from the live local API;
2. reduces the result to the exact `summary` + `allowed_numeric_values` pair
   `aeroroute-api` sends to this service in production (the deterministic
   template text and its permitted numbers -- see the script's module
   docstring for why this logic is duplicated here rather than imported);
3. runs it through the real `MlxLmProvider` (model loaded once, reused
   across all 24 cases) and the real validators
   (`validate_numeric_claims`, `validate_operational_claims`,
   `parse_structured_text`) -- the identical code path production uses;
4. records whether the native response was accepted (`fallback_used=False`)
   or silently fell back to the template provider, plus latency.

Reproduce with:

```bash
cd aeroroute-mlx
uv sync --all-groups --extra mlx
find .venv -name '._*' -delete   # see NATIVE_BENCHMARK_2026-07-09.md
uv run python scripts/quality_corpus.py configs/gemma3-4b-smoke.json \
  --output docs/quality-corpus-2026-07-09.json
```

(Requires the local dev stack up: `make dev-up` in `aeroroute-platform`,
migrations and an imported airport bundle in `aeroroute-api`.)

## Result

| Metric | Value |
| --- | ---: |
| Total cases | 24 |
| API errors (excluded from pass rate) | 1 |
| Evaluated cases | 23 |
| Passed (native MLX response accepted, no fallback) | 23 |
| **Pass rate** | **100%** |
| Latency p50 | 2.595 s |
| Latency p95 | 2.973 s |

The one API error (`OMDB-SKBO-B77W-minimum_time`, Dubai-Bogota) is not an
MLX or model-quality issue: the deterministic optimizer correctly returned
`aircraft_mass_outside_profile` (422) because that route/aircraft/payload
combination exceeds the supported mass profile -- exactly the kind of
actionable-problem response HLD SS11.5.1 requires for infeasible cases, not a
degraded or invented result. This is expected behavior, not a failure to
investigate.

All 23 evaluated explanations passed schema parsing, numeric-claim
validation (no number appeared that wasn't in the deterministic facts), and
operational-claim validation (no "filed", "approved", "cleared" language) on
the first native attempt, with no fallback triggered. Spot-checked text
samples read as natural, grounded restatements of the deterministic facts
(see `docs/quality-corpus-2026-07-09.json` for all 23 full texts), consistent
with the model's role as an explanation layer, not a fact generator.

## Limitations of this pass

- 24 cases is a quality *signal*, not the "larger repeated quality corpus"
  or held-out/gold-set evaluation HLD SS13.8 describes for an actual
  promotion decision -- there is no promotion decision to make here, since
  there is no adapter or challenger result to compare against yet.
- All requests used `/api/v1/optimizations` (still-air, no live weather, no
  terminal/AIRAC enrichment) for speed; the production explanation path is
  identical regardless (it only ever sees the reduced `summary` +
  `allowed_numeric_values`), so this does not affect validity of the MLX
  quality signal, only route diversity.
- No adversarial or edge-case prompts (extreme deltas, degraded-data
  wording, near-zero fuel savings phrased ambiguously) were included; this
  is representative-case coverage, not stress testing.

## Decision

No promotion decision applies (prompt-only baseline, no adapter). This
corpus adds real evidence that the 4B prompt-only baseline is reliable
across varied routes/aircraft/profiles on this hardware, ahead of the
Mistral/Qwen bake-off described in HLD SS13.5's model portfolio table.
