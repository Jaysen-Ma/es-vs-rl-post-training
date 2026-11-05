# Reproduction and saved-result analysis

Start here to explore the package without downloading models or running training.
These workflows consume saved predictions or histogram CSVs. The synthetic
pass@k example needs neither a GPU nor study data. Obtain the full repository
for the paper and committed study summaries; the wheel and source archive do
not include those research artifacts.

For checkpoints and model execution, use [Models and evaluation](model_evaluation.md).
For code changes and CI, use the [contributor guide](development.md).

## Analysis environment

Analysis means processing saved model outputs: computing pass@k, making paired
statistical comparisons, and producing tables and plots. It does not generate
new model responses. The `analysis` extra contains NumPy, SciPy and Matplotlib
and does not install PyTorch, vLLM or model checkpoints. The local `analysis`
dependency group selects that same extra, keeping the existing setup command.

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and use Python
3.12, as selected by `.python-version`. From the repository root:

```bash
uv sync --locked --group analysis
uv run --locked --group analysis es-capacity-passk --help
```

`uv sync` installs `es_capacity` in editable mode, so changes under
`src/es_capacity/` are available to the existing script commands.

`pyproject.toml` declares the dependencies; `uv.lock` records exact resolved
versions and artifact hashes, including transitive dependencies. `--locked`
rejects a stale lockfile instead of silently resolving a different environment.
Use a separate virtual environment from any existing GPU training environment;
`uv sync` manages this checkout's `.venv` and can remove packages outside the
selected dependency groups.

The validated baseline is CPython 3.12 on Linux with uv 0.10.3. The declared Python
minimum is 3.10 because the scripts use Python 3.10 syntax. Core model
configuration and runner CLI wiring include the `tomli` fallback on Python 3.10.
The full analysis suite and package checks run on Python 3.10 and 3.12 in CI.
Other Python versions and platforms have not yet been validated.

To deliberately update dependencies, use `uv lock --upgrade-package NAME`, repeat
the relevant validation, and commit the lockfile change with its reason. Routine
setup should use `--locked`.

## What can be reproduced from a clone

| Task | Command | Required inputs |
|---|---|---|
| Paper's pass@k figure | `uv run --locked --group analysis python scripts/plot_hero.py` | The committed three-way summary JSON files |
| pass@k example | `uv run --locked --group analysis es-capacity-passk --example --out /tmp/passk-example --plot` | Synthetic correctness scores bundled with the package |
| pass@k summaries | `uv run --locked --group analysis es-capacity-passk --results-dir PATH --out OUTPUT --label base --label trained --label rl --baseline base --plot` | Scored per-question evaluation JSONL files under `PATH/LABEL/BENCHMARK/` |
| MMLU tables | `uv run --locked --group analysis es-capacity-mmlu --results-dir PATH --out OUTPUT` | `PATH/ARM/results.json` and `PATH/ARM/samples.jsonl` for each arm |
| ETHICS commonsense summary | `uv run --locked --group analysis es-capacity-ethics --results-dir PATH --subset commonsense` | `PATH/ARM/samples.jsonl` for each model arm |
| Other four ETHICS summaries | `uv run --locked --group analysis es-capacity-ethics --results-dir PATH --subset rest` | `PATH/ARM/rest_samples.jsonl` for each model arm |
| GPQA summaries | `uv run --locked --group analysis es-capacity-gpqa summarize --results-dir PATH --out OUTPUT --pair 7B-ES:7B-base` | Saved GPQA seed-sweep results and samples |
| GPQA tables | `uv run --locked --group analysis es-capacity-gpqa tables --results-dir PATH` | Reduced GPQA CSV files |
| Weight-change plots | `uv run --locked --group analysis es-capacity-weights --scale 7B --results-dir PATH --out OUTPUT` | `PATH/SCALE-ES/` and `PATH/SCALE-RL/`, each containing `stats.csv` and `hist.csv` |

The figure command overwrites `writeup/figures/hero_passk.png` and
`writeup/figures/hero_passk.pdf`. Its numerical inputs are committed, but rendered
files can differ across plotting environments. Apart from the bundled synthetic
example, the other commands require data that is excluded from Git; a fresh clone cannot regenerate every paper result
without obtaining or regenerating those inputs. The source raw evaluation data
should be retained separately.

The ETHICS commands read all model-arm directories containing the selected
sample file and replace `summary.csv` or `rest_summary.csv` in `--results-dir`.
That path is explicit and can be outside the checkout. They need saved predictions,
not checkpoints, the original dataset archive, or a GPU. A fresh clone does not
contain these sample files. The existing ETHICS runners call the same packaged
summary functions after evaluation; the scoring and CSV formats are unchanged.

Checkpoint comparison (`es-capacity-compare-weights`) requires PyTorch and
SafeTensors and is outside this analysis environment. See the
[checkpoint comparison setup](model_evaluation.md#checkpoint-comparison).

## pass@k analysis

`es-capacity-passk` reads the scored JSONL shards produced by the existing
limit-of-RLVR maths evaluator. It does not generate answers or grade them again.
The numerical functions are in `src/es_capacity/analysis/passk.py`; the installed
command is in `src/es_capacity/cli/passk.py`.

Install the analysis environment above, then run a self-contained example:

```bash
uv run --locked --group analysis es-capacity-passk --example --out /tmp/passk-example --plot
```

The bundled `src/es_capacity/examples/passk.json` contains synthetic correctness
indicators for four questions and four attempts under the labels `reference` and
`candidate`. It includes one question in each coverage category: solved by both,
solved by just the reference, solved by just the candidate, and solved by neither.
The example makes no claims about model performance and includes no benchmark
questions or generated answers. Its summary and plot remain in the requested
output directory after the temporary example inputs are removed.

For actual scored outputs, use this layout:

```text
RUN/
  base/math500/shard_0.jsonl
  trained/math500/shard_0.jsonl
  rl/math500/shard_0.jsonl
```

Every nonempty JSONL line needs an `idx` question ID and a `score` list of boolean
correctness indicators, for example:

```json
{"idx": 0, "score": [true, false, false, true]}
```

Shard files should cover disjoint question IDs. All compared labels must cover
the same IDs. Each label must have a positive, uniform sample count across its
questions. The four-way breakdown additionally requires equal counts between
each compared pair; unequal-budget pairs retain their curves and omit that
breakdown. The existing shard-loading rules are retained in this extraction.

```bash
es-capacity-passk --results-dir /data/run --out /data/summaries --label base --label trained --label rl --baseline base --benchmark math500 --plot
```

`--results-dir` and `--out` are explicit, so installed commands work outside a
checkout. Labels are unique directory names; `--baseline` defaults to the first
label. Repeat `--benchmark` to select from `aime24`, `math500`, `minerva_math` and
`olympiadbench`. Without that option the command processes the four study
benchmarks and reports missing label directories as skipped. An explicitly
selected missing benchmark is an error, as is a run with no available benchmarks.

Outputs are `BENCHMARK_summary.json` and, with `--plot`, `BENCHMARK_passk.png`.
Existing files of those names are replaced. The JSON schema remains unchanged:
`baseline`, per-label `n_sampling`, `ks`, `pass_at_k`, and `four_way_breakdown`.
Pass@k values are fractions from 0 to 1; `fractions_pct` contains percentages.
Scoring rules, power-of-two k values and comparison guards are unchanged.

The estimator matches the formula in the external
[evaluate.py](https://github.com/Jaysen-Ma/limit-of-RLVR/blob/569825a3f8c68dc6d25415917498db634ea73bdb/math/examples/math_eval/evaluate.py#L16):
`1 - C(n-c, k) / C(n, k)`, where `c` of `n` completions are correct. Tests compare
against an independent exact-combination reference, including all-correct,
all-incorrect, non-power-of-two and larger sample budgets. CI requires no external
checkout to run these tests.

The old `python scripts/passk/analyze.py --run-tag RUN --label ...` interface is a
compatibility launcher. It still reads `results/RUN/` relative to the checkout and
defaults to writing summaries there. It delegates to the same packaged functions;
new code should use the installed command with explicit paths.

## MMLU analysis

`es-capacity-mmlu` consumes saved outputs from the MMLU model runner; it does not
load models or run the benchmark. The analysis functions live in
`src/es_capacity/analysis/mmlu.py` and the command in `src/es_capacity/cli/mmlu.py`.
With the analysis dependencies installed:

```bash
es-capacity-mmlu --results-dir /data/mmlu-run --out /data/mmlu-summary
```

The input directory contains one subdirectory per model arm:

```text
mmlu-run/
  7B-base/results.json
  7B-base/samples.jsonl
  7B-ES/results.json
  7B-ES/samples.jsonl
```

Each `results.json` supplies the aggregate `results.mmlu["acc,none"]`, effective
sample counts under `n-samples`, the `group_subtasks` hierarchy and optional
configuration metadata. Each sample line supplies `subtask`, `doc_id` and `acc`.
All directories with `results.json` are included and must have `samples.jsonl`.
Question keys are `(subtask, doc_id)`; record order can differ between arms.
Duplicate keys and mismatched question sets in a trained/base pair are errors.

The command creates `--out` if needed and replaces only its `scores.csv`, leaving
the input files untouched when output is separate. It prints the study's existing
zero-shot Markdown table. The CSV retains accuracy, counts, gained/lost questions,
raw and adjusted McNemar p-values, few-shot count and elapsed time. The study's
six arms come first; additional arm directories follow alphabetically. A trained
arm is compared with the `SCALE-base` arm inferred from its name when present.
Bonferroni correction covers the trained/base comparisons available in that run.
Overall accuracies, statistical calculations, rounding and table format are
unchanged. Use the CSV metadata when interpreting runs with other shot counts.

The legacy `python scripts/mmlu/tables.py --sweep-root PATH` launcher delegates
to the packaged command and still writes `PATH/scores.csv`. Without that argument
it uses `results/mmlu_results` relative to the current directory. New commands
should use the installed entry point with explicit input/output paths. Saved MMLU
inputs are excluded from Git; CI exercises synthetic results and samples.

## GPQA analysis

`es-capacity-gpqa` analyses the saved seed sweeps from `es-capacity-evaluate gpqa`.
It uses deterministic likelihood scores already in those files; it does not run
models or generate answers. The functions are in `src/es_capacity/analysis/gpqa.py`
and the command is in `src/es_capacity/cli/gpqa.py`.

```bash
es-capacity-gpqa check --results-dir /data/gpqa-sweep
es-capacity-gpqa summarize --results-dir /data/gpqa-sweep --out /data/gpqa-summary --pair 7B-ES:7B-base --pair 7B-RL:7B-base
es-capacity-gpqa tables --results-dir /data/gpqa-summary
```

Inputs follow `SWEEP/seedN/ARM/`, including any nested harness output directories.
For each seed/arm the loader selects the last `results_*.json` path in lexical
order, as the existing analyzer does, then requires the matching
`samples_gpqa_diamond_zeroshot_TIMESTAMP.jsonl` beside it. It rejects missing,
truncated or duplicate sample records. Results supply `results` and `n-samples`
for `gpqa_diamond_zeroshot`, including `acc,none`, `acc_stderr,none` and the effective
count, plus `config.model_args.pretrained`. Each sample supplies `doc_id`, `acc`,
`doc_hash` and four likelihoods in `filtered_resps`.

Every seed must contain the same model arms and question IDs, with nonempty
samples and question hashes. Within a seed, arms must have identical question
hashes, so their paired comparisons refer to the same shuffled questions. Across
seeds, the hash vectors must differ. These checks detect repeated permutations;
they do not establish statistical independence of the seeds. `check` writes
nothing and returns status 1 for an invalid sweep. Missing files, malformed input
and invalid arguments return status 2.

`summarize` validates the inputs and computes all tables before creating `--out`.
It writes four CSVs, replacing files of those names:

- `scores.csv`: seed/arm accuracy, counts, standard error and model path.
- `scores_summary.csv`: each arm's mean, standard deviation, minimum and maximum.
- `per_question.csv`: how many seeds each arm answered each question correctly.
- `position_bias.csv`: pooled choice-position frequencies across seeds.

Repeat `--pair ARM:BASELINE` to add `mcnemar.csv` (per-seed exact paired tests) and
`pairs_summary.csv` (accuracy deltas and the existing per-question Wilcoxon test).
Pairs must name arms in the sweep. With no pairs, those two generated files are
removed from the output directory if left by an earlier paired run; unrelated
files remain. Use separate input and output directories to keep summaries apart
from raw evaluations. SciPy is required when computing nonzero Wilcoxon contrasts;
missing SciPy produces installation instructions rather than a `nan` report.

The statistical calculations, CSV schemas, rounding and study table formatting
are preserved. `tables` prints Markdown from the reduced CSV directory and needs
no raw samples. It supports summaries without pair files and includes additional
arm names after the study's six arms. Raw benchmark text and saved runs are
excluded from Git; the tests use synthetic scores, hashes and likelihoods.

The old `scripts/gpqa/analyze.py --sweep-root PATH --out OUTPUT --pair ARM:BASELINE`
and `--check` forms delegate to the installed command. The old
`scripts/gpqa/tables.py --dir PATH` launcher remains available, with its existing
`results/gpqa` default. New commands should use the installed entry point with
explicit paths. None of these commands requires model configuration.

## Weight-distribution figures

`es-capacity-weights` plots the saved CSV outputs of `es-capacity-compare-weights`.
It does not open checkpoints or require model configuration, PyTorch or a GPU.
Install the analysis dependencies, then select the CSV root and a separate output
directory:

```bash
es-capacity-weights --scale 7B --results-dir /data/weight-comparisons --out /data/weight-figures
```

Inputs follow this layout, with the corresponding `1.5B-ES` and `1.5B-RL`
directories for `--scale 1.5B`:

```text
weight-comparisons/
  7B-ES/stats.csv
  7B-ES/hist.csv
  7B-RL/stats.csv
  7B-RL/hist.csv
```

`stats.csv` records parameter count, mean, standard deviation, extrema, excess
kurtosis, tail and zero-fraction statistics, keyed by `series`, `grouping` and
`group`. Series are `base`, `trained` and `delta`; groupings include pooled weights,
module type and layer depth. `hist.csv` records the nonempty bins and serialized
underflow/overflow edges on the 512-bin grid. All four files are required. The
plots need nonzero-variance pooled distributions and at least one shared module
distribution for ES and RL. Missing or incomplete required groups fail before
creating the output directory.

The command creates `--out` if needed and writes these five PNGs, replacing
existing files of the same names:

- `raw_weights_SCALE.png`: base and post-trained parameter distributions.
- `delta_pooled_SCALE.png`: pooled changes on shared and zoomed axes.
- `weight_change_SCALE.png`: the standalone shared-axis figure used in the README.
- `delta_by_module_SCALE.png`: changes and their sizes by module.
- `delta_by_depth_SCALE.png`: change size against transformer layer index.

Histogram reconstruction and numerical helpers live in
`src/es_capacity/analysis/weights.py`. Figure rendering lives in `weight_plots.py`
and uses Matplotlib's headless Agg backend; the CLI is in
`src/es_capacity/cli/weights.py`. NumPy is loaded only when histogram calculations
need it, and Matplotlib only for rendering. Command help works without either.

Bin coordinates still come from the saved edges, including the shared base and
trained grid. Counts, density normalization, rebinning, axes and styling are
unchanged. Input CSVs remain untouched. Figure pixels can vary across plotting
environments; use the locked environment for comparisons. The full saved CSVs are
excluded from Git, while tests render synthetic distributions.

The old `python scripts/weights/plot.py --scale 7B --dir PATH` launcher still
writes to `PATH/figures`; its default input root remains `results/weights`.
To generate new CSVs from checkpoints, follow
[checkpoint comparison](model_evaluation.md#checkpoint-comparison).
