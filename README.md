# R-Clarify

R-Clarify studies how a household assistant can infer what a user wants from an ambiguous utterance and a description of the room. The agent chooses from 40 household actions. It can ask a simulated user a clarification question before acting. After an unsuccessful trial, it can use a reflection memo to guide its next attempt.

## Dataset

The experiments use the extended **Do-I-Demand** annotations described by Yamasaki et al. (SIGDIAL 2025). A household scene can have several acceptable actions. In the R-Clarify evaluation, each scene–action pair is a separate sample with one target action. The simulated user receives that target, and a trial succeeds when the agent selects it. [Dataset paper](https://aclanthology.org/2025.sigdial-1.45/) · [Upstream repository](https://github.com/riken-grp/dataloader_reflective_stretch)

| File                                                         | Contents                                                     |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [raw_data.json](data/raw_data.json)                          | 355 scenes with at least one accepted action, comprising 547 action labels in the `reflective_actions` field. |
| [processed_data.json](data/processed_data.json)              | The same 355 scenes with structured scene annotations and 545 action labels. The action lists differ from the raw file at zero-based indices 2 and 112. |
| [processed_data_expanded.json](data/processed_data_expanded.json) | The 545 single-target samples used by the experiment runner, indexed from 0 to 544. |
| [processed_data_expanded_en.json](data/processed_data_expanded_en.json) | Contains the same JSON data as the expanded file, including the Japanese utterances and action labels. |
| [options.txt](data/options.txt)                              | The 40 candidate household actions.                          |

## Run an experiment

From the repository root, install the locked dependencies with `uv`. The project requires Python 3.11 or later and pins Python 3.11 in `.python-version`.

```bash
uv sync --locked
```

Create a `.env` file in the repository root with your OpenAI API key.

```dotenv
OPENAI_API_KEY=your_key_here
```

The default agent and reflection model is `gpt-5`. To use a local Hugging Face model, set `DEFAULT_BACKEND = "hf"` and choose `HF_MODEL_ID` in [llm_factory.py](src/llm_factory.py). The user simulator uses OpenAI `gpt-5.2` with either agent backend, and the current runner initializes an OpenAI client at startup. [User simulator implementation](src/graph.py)

To run one sample with up to five trials and two clarification questions per trial:

```bash
uv run python -m src.run --mode r-clarify --max-trials 5 --clarify-quota 2 --start 0 --end 1
```

To run the same experiment on all 545 samples:

```bash
uv run python -m src.run --mode r-clarify --max-trials 5 --clarify-quota 2
```

Trials stop at the first correct action or at `--max-trials`. Sample ranges include `--start` and exclude `--end`. Omit `--clarify-quota` for unlimited clarification turns per trial. Use `--memo-window` to set how many recent reflection memos the next trial receives.

### Experiment modes

| `--mode`                 | Can ask questions | Information carried into the next trial                      |
| ------------------------ | ----------------- | ------------------------------------------------------------ |
| `r-clarify`              | Yes               | Most recent reflection memo by default.                      |
| `r-clarify_reflexion`    | Yes               | Three most recent reflection memos by default.               |
| `r-clarify_non_cot`      | Yes               | Most recent reflection memo by default. The prompt omits the reasoning output. |
| `clarify`                | Yes               | None.                                                        |
| `non_thinking_clarify`   | Yes               | None. The prompt omits the reasoning output.                 |
| `action_elimination`     | Yes               | IDs of previously failed actions. Repeating a blocked action counts as a failed execution. |
| `reflect`, `cot_reflect` | No                | Most recent reflection memo by default. Both use the same execute-only reasoning prompt. |
| `cot`                    | No                | None. The prompt requests reasoning before action selection. |
| `direct`                 | No                | None. The prompt requests a direct action choice.            |

## Results and evaluation

With the default agent, the full-dataset command writes results to `outputs/gpt-5/results/r-clarify.json`. The one-sample command writes `outputs/gpt-5/results/r-clarify_0_1.json`. Per-sample dialogue transcripts are saved to `outputs/gpt-5/episodes/r-clarify/sample<index>.txt`.

Each result records the target action, trial outcomes, step counts, clarification exchanges, and agent and reflection token usage. Rerunning a command skips indices already present in its result file. The filenames depend on the model, mode, and sample range. Before changing the trial limit, clarification quota, or memo window for the same range, move the previous results and transcripts to a separate directory.

To compute DPass and TDPass for the full-dataset example at trial budgets of three and five with a step discount of 0.8, run:

```bash
uv run python -m extra_analysis.alpha_sensitivity.run --input outputs/gpt-5/results/r-clarify.json --trials 3 5 --alphas 0.8
```

- **Pass@k** is the fraction of evaluated samples that succeed within the first *k* trials.
- **DPass@k** assigns a successful sample a score of `alpha ** (steps - 1)`, using the steps in its successful trial. Samples that do not succeed within *k* trials score zero. The reported metric averages over all evaluated samples.
- **TDPass@k** uses the same discount with the cumulative step count across all trials up to and including success.

For Pass@k, mean trial lengths, and token usage through success, set `metrics_path` in [eval.py](src/eval.py) to the result file you want to evaluate, then run `uv run python -m src.eval`.

## Additional analyses and utilities

| Resource                                                     | Purpose                                                      |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [Alpha sensitivity analysis](extra_analysis/alpha_sensitivity/README.md) | Recompute discounted scores from saved results across trial budgets and discount factors. |
| [Leakage audit](extra_analysis/leakage_audit/README.md)      | Inspect whether clarification replies reveal the target action. Includes simulator and human comparisons. |
| [Human paired statistics](extra_analysis/human_paired_statistics.py) | Compare conditions on matched participant–sample pairs. Pass the two sets of human results with `--r-clarify` and `--wo-tur`. [Summary tables](extra_analysis/human_paired_statistics/results) |
| [merge_results.py](merge_results.py)                         | Merge result shards by sample index. Set the input directory, filename pattern, and output path in `main()`. |
| [process_dataset.py](process_dataset.py)                     | Rebuild single-target samples. Set the input path in `main()` to `data/processed_data.json` and choose an output path before running. |
| [test.py](test.py)                                           | Compute clarification usage among successful trials in an existing results file. Set the result path and trial limit in `main()`. |
