# Alpha Sensitivity Analysis

Post-hoc sensitivity analysis for `DPass@k(alpha)`.

The main experiment uses `alpha = 0.8`. This tool recomputes `DPass` from existing
result JSON files across multiple alpha values, so it does not rerun model
inference.

## Quick Start

```bash
uv run python -m extra_analysis.alpha_sensitivity.run
```

By default this scans:

```text
outputs/**/results/*.json
```

and writes:

```text
extra_analysis/alpha_sensitivity/results/
```

## Common Usage

Analyze only the main `gpt-5/r-clarify` result:

```bash
uv run python -m extra_analysis.alpha_sensitivity.run \
  --results-glob "outputs/gpt-5/results/r-clarify.json"
```

Use a denser grid around the experimental value `0.8`:

```bash
uv run python -m extra_analysis.alpha_sensitivity.run \
  --alphas "0.70,0.75,0.80,0.85,0.90"
```

Match a different trial budget:

```bash
uv run python -m extra_analysis.alpha_sensitivity.run --trial 5
```

Evaluate only the first `k` samples:

```bash
uv run python -m extra_analysis.alpha_sensitivity.run --k 100
```

## Outputs

- `alpha_sensitivity.csv`: one row per result file and alpha value.
- `summary.csv`: per-file min/max/range summary across alpha values.
- `rankings.csv`: method ranking at each alpha value.
- `summary.md`: compact Markdown tables for reporting.
- `config.json`: exact analysis configuration.

Key columns:

- `pass_at_trial`: normal success rate under the selected trial budget.
- `dpass`: step-discounted success score at the given `alpha`.
- `baseline_dpass`: score at the baseline alpha, default `0.8`.
- `delta_from_baseline`: absolute score change from the baseline.
- `relative_delta_from_baseline_pct`: percentage change from the baseline.
