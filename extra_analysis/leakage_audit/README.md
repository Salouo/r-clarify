# Leakage Audit

Post-hoc audit tooling for simulator label leakage in generated clarification responses.

This package never reruns the main R-Clarify experiments and only writes under
`extra_analysis/leakage_audit/results/`.

## Inspect First

```bash
python -m extra_analysis.leakage_audit.run inspect
```

The inspection step prints:

- discovered simulator result logs;
- discovered human response files, including auto-detected `human_runs*` logs;
- clarification response counts;
- the schema mapping from result JSON fields to audit fields;
- the exact LLM judge prompt;
- estimated judge API call count.

The default result glob is:

```text
outputs/**/results/*.json
```

Human comparison files are auto-detected by `user_source: human` or folders whose
names start with `human_runs`, such as:

```text
outputs/human_runs_gpt-5/results/*.json
```

## Pilot

```bash
python -m extra_analysis.leakage_audit.run pilot --pilot-size 10
```

Use `--dry-run` to print sampled inputs without API calls.

## Full Run

The full run is intentionally guarded:

```bash
python -m extra_analysis.leakage_audit.run run --approve-full-run
```

Useful options:

```text
--judge-model MODEL
--temperature 0
--num-judge-runs 1
--num-judge-runs 3
--bootstrap-samples 10000
--seed 42
--human-log-glob "path/to/human/*.csv"
--no-human
--no-progress
--output-name gpt-5/r-clarify
```

For one specific model/condition result:

```bash
python -m extra_analysis.leakage_audit.run pilot \
  --results-glob "outputs/gpt-5/results/r-clarify.json" \
  --no-human \
  --pilot-size 40 \
  --judge-model gpt-4o
```

That pilot writes to:

```text
extra_analysis/leakage_audit/results/gpt-5/r-clarify/pilot/
```

Pilot and full runs show a tqdm progress bar by default for judge calls, including
percentage, current record/sample, run index, and whether the row came from cache
or the API.

## Outputs

Outputs are written under `extra_analysis/leakage_audit/results/`:

- `audit_labels.jsonl`
- `audit_final_labels.jsonl`
- `lexical_audit.jsonl`
- `summary.json`
- `summary.csv`
- `examples_no_target_signal.csv`
- `examples_legitimate_intent.csv`
- `examples_near_direct_exposure.csv`
- `examples_direct_leakage.csv`

## Quick Label Counts

Print the four-class counts from a result file or directory:

```bash
python -m extra_analysis.leakage_audit.summarize_labels extra_analysis/leakage_audit/results/pilot
```

Optional breakdowns:

```bash
python -m extra_analysis.leakage_audit.summarize_labels extra_analysis/leakage_audit/results --by-source --by-model --by-condition
```
