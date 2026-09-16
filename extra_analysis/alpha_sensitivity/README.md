# Alpha Sensitivity Analysis

Offline sensitivity analysis for `DPass@k(alpha)` and `TDPass@k(alpha)`.

The script only reads an existing result JSON file. It does not rerun the agent,
user simulator, reflection, or any API/model call.

## Quick Start

```bash
uv run python -m extra_analysis.alpha_sensitivity.run --input "outputs/gpt-5/results/r-clarify.json"
```

Default alpha values are `0.5 0.6 0.7 0.8 0.9`. Default trial budgets are `3 5`.
By default, the script prints a readable table and a CSV block to the console,
and does not write any file.

## Save CSV

```bash
uv run python -m extra_analysis.alpha_sensitivity.run \
  --input "outputs/gpt-5/results/r-clarify.json" \
  --csv "extra_analysis/alpha_sensitivity/results/gpt-5_r-clarify.csv"
```

## Options

```bash
uv run python -m extra_analysis.alpha_sensitivity.run \
  --input "outputs/gpt-5/results/r-clarify.json" \
  --alphas 0.70 0.75 0.80 0.85 0.90 \
  --trials 3 5 \
  --percent
```

Columns:

- `DPass@3`, `DPass@5`: discount only the first successful trial's steps.
- `TDPass@3`, `TDPass@5`: discount cumulative steps through the first successful trial.
