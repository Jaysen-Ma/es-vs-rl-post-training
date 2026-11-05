# Development and reproduction

The `es_capacity` package provides the shared model registry, filesystem
configuration and saved-result analysis for pass@k, ETHICS, MMLU, GPQA and weight
distributions. MMLU, GPQA and all ETHICS model runners are under `evaluation/`;
checkpoint comparison is packaged separately from saved-result plotting. Model execution requires
a separate GPU environment.

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
files can differ across plotting environments. The other commands require data
that is excluded from Git; a fresh clone cannot regenerate every paper result
without obtaining or regenerating those inputs. The source raw evaluation data
should be retained separately.

The ETHICS commands read all model-arm directories containing the selected
sample file and replace `summary.csv` or `rest_summary.csv` in `--results-dir`.
That path is explicit and can be outside the checkout. They need saved predictions,
not checkpoints, the original dataset archive, or a GPU. A fresh clone does not
contain these sample files. The existing ETHICS runners call the same packaged
summary functions after evaluation; the scoring and CSV formats are unchanged.

Checkpoint comparison (`scripts/weights/compare.py`) requires PyTorch and
safetensors and is outside this analysis environment.

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

`es-capacity-weights` plots the saved CSV outputs of `scripts/weights/compare.py`.
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
Generating new comparison CSVs with `scripts/weights/compare.py` still requires
downloaded checkpoints and its separate dependencies.

## Checkpoint comparison

`es-capacity-compare-weights` compares downloaded SafeTensors checkpoints on the
CPU and writes the inputs for weight-distribution plotting:

```bash
es-capacity-compare-weights --config /data/models.toml --scale 7B --arms ES RL --out /data/weight-comparisons
es-capacity-weights --scale 7B --results-dir /data/weight-comparisons --out /data/weight-figures
```

Install this package into an environment with PyTorch and SafeTensors. These
libraries are outside the locked analysis environment; command help and model
preflight do not import them. Model paths use the shared TOML configuration.
The installed command requires `--out`; the legacy `scripts/weights/compare.py`
launcher retains its `results/weights` default.

The algorithm in `src/es_capacity/analysis/weight_comparison.py` streams the same
tensor chunks in two passes, retaining raw units, shared base/trained histogram
grids, underflow/overflow counts and separate excluded/tied-head groups. It writes
`SCALE-ARM/stats.csv`, `hist.csv` and `per_tensor.csv`, replacing those files on
reruns. No inference or training is involved.

Tiny CPU checkpoints were checked with Python 3.12.3, PyTorch 2.11.0+cu130 and
SafeTensors 0.8.0 on Linux aarch64. All three CSVs match the previous implementation
byte for byte. The optional numerical tests run with those libraries installed:

```bash
python -m unittest discover -s tests -p test_checkpoint_comparison.py -v
```

The two tensor tests are skipped in the lightweight CI environment; CLI/config
checks still run there. Full-size checkpoint comparison was not repeated during
this migration.

## Evaluation environments and study setup references

`external-repositories.toml` records the exact local checkout revisions inspected
on 2026-11-05. These are setup reference points, not evidence that the historical
experiments ran at precisely those commits. The experiment records do not yet
establish that provenance. Branch names are hints; use the recorded commit IDs
when reconstructing these checkouts.

The training repository is recorded as study background; evaluation users do not
need to install or run it. The external repositories declare incompatible
dependency versions:

| Workflow | External declaration at the recorded revision |
|---|---|
| ES training (`es-at-scale/setup.py`) | `vllm==0.11.0`, `transformers==4.57.6` |
| Maths evaluation (`limit-of-RLVR/math/requirements.txt`) | `vllm<=0.6.3`, `transformers<4.48`, `math-verify[antlr4_11_0]==0.6.0` |

These declarations have not been validated as fresh GPU installations in this
packaging step. The write-up records `math-verify` 0.9.0 for ES training; that
package is not declared by the training repository's `setup.py`. Do not treat a
current dependency file as a complete record of the environment used for the
paper. A follow-up GPU validation should record Python, package versions, CUDA
and driver versions, model and dataset revisions, repository commits and run
settings together.

Maths evaluation uses the external evaluation environment. The local MMLU,
GPQA and ETHICS runners depend on the evaluation harness (`lm_eval`),
Hugging Face Datasets and vLLM. They are not covered by `uv.lock` yet. Setup for
those workflows must be validated on suitable hardware before it is described
as reproducible.

Install the shared package into an existing GPU environment with:

```bash
python -m pip install -e .
```

The core installation has no third-party dependencies on Python 3.11 or newer;
Python 3.10 additionally installs `tomli` to read TOML configuration. This command
adds the shared package without selecting the analysis extra or GPU runtime.
ETHICS statistical summaries still require SciPy in that environment. Keep using
its interpreter for `python scripts/...` commands. Users evaluate downloaded
checkpoints; no training step is required.

## MMLU model evaluation

The installed `es-capacity-evaluate mmlu` command runs models; the separate
`es-capacity-mmlu` command analyses saved results. Install the package into the
existing GPU evaluation environment described above. The core package and analysis
extra do not install `lm_eval`, Hugging Face Datasets, vLLM or their GPU dependencies.
No training step is needed.

Download the desired checkpoints, configure their paths as described in
[Model and data paths](#model-and-data-paths), then select the model arms and an
explicit output directory:

```bash
es-capacity-models check --config /data/models.toml --arms 7B-base 7B-ES 7B-RL
es-capacity-evaluate mmlu --config /data/models.toml --arms 7B-base 7B-ES 7B-RL --out /data/mmlu-run
```

Run these commands with the evaluation environment's executables. When no arms
are selected, the runner evaluates all six registered arms. It checks every
selected local checkpoint before creating the output directory or starting any
worker. The explicit `--config` or `ES_CAPACITY_CONFIG` selection is resolved to
an absolute path and passed to each worker. Checkpoint paths remain local and do
not fall back to a Hub download. The benchmark itself may need to be downloaded
by the evaluation harness on first use.

The orchestration and worker live in `src/es_capacity/evaluation/mmlu.py`; argument
handling lives in `src/es_capacity/cli/evaluate.py`. Each arm runs sequentially in
a fresh process using the same Python interpreter and the installed module,
without a script path in the checkout. The parent reports failures, continues
with the remaining arms, and exits with status 1 if any worker failed. Each worker
rechecks its selected checkpoint before loading the model runtime.

The output remains `OUTPUT/ARM/results.json` and `OUTPUT/ARM/samples.jsonl`.
Rerunning the same arm replaces those files. Results retain the harness metrics,
sample counts and metadata for arm, checkpoint path, dtype, few-shot count, seed
and elapsed scoring time. Samples retain their subject labels for paired analysis.
After evaluation, analyse them with:

```bash
es-capacity-mmlu --results-dir /data/mmlu-run --out /data/mmlu-summary
```

Scoring still ranks the four answer letters by likelihood; it does not sample
answers. The existing defaults are unchanged:

| Setting | Default |
|---|---|
| Few-shot examples | `0` |
| Seed | `0` |
| Dtype | `bfloat16` |
| Batch size | `1024` |
| Maximum model length | `1280` |
| Maximum batched tokens | `2048` |
| GPU memory utilization | `0.25` |
| Dataset limit | Full benchmark |

Use `--limit 0.05` for a small fraction of each subject. Batch size remains an
explicit integer: the original runner avoids the harness's automatic batching
path. The length and token limits are study settings; a different few-shot count
or model can require different limits. All seed settings, eager execution and
sample logging remain unchanged. The worker defaults `VLLM_USE_FLASHINFER_SAMPLER`
to `0` before runtime imports and retains an existing user value. Importing the
package or showing help does not change the parent environment.

The legacy `python scripts/mmlu/run.py` command delegates to the installed CLI
and retains its flags and `results/mmlu_results` default output path. New installed
commands require `--out`. Help and model preflight work with the core package.
CI tests orchestration, simulated runtime calls, output formats and worker cleanup
without loading model weights. Actual GPU inference and a fresh evaluation
environment remain unvalidated in this packaging work.

## GPQA model evaluation

Use `es-capacity-evaluate gpqa` in the existing GPU evaluation environment, with
downloaded checkpoints and the same model configuration as MMLU. This command
runs models; `es-capacity-gpqa` analyses the saved outputs.

```bash
es-capacity-evaluate gpqa --config /data/models.toml --arms 7B-base 7B-ES 7B-RL --seeds 0 1 2 3 4 5 6 7 8 9 --out /data/gpqa-sweep
es-capacity-gpqa check --results-dir /data/gpqa-sweep
es-capacity-gpqa summarize --results-dir /data/gpqa-sweep --out /data/gpqa-summary --pair 7B-ES:7B-base --pair 7B-RL:7B-base
```

Without `--arms` or `--seeds`, the runner selects all six registered arms and
seeds 0 through 9. Seed lists must be nonempty, distinct and nonnegative; negative
seed-directory names would not match the saved-output analyzer's layout. The
installed command requires `--out`. It checks every selected checkpoint before
creating that directory or starting workers, and forwards the resolved absolute
configuration path to each worker.

GPQA-diamond scoring still ranks the four choices by likelihood rather than
sampling answers. Seeds vary the answer-option ordering, not a sampling budget.
Each model runs in its own process and is loaded once, then reused for every
selected seed. All four harness seed arguments are set to the same value per
evaluation. `datasets.disable_caching()` runs before task setup to prevent reuse
of a cached answer permutation, and per-question sample logging remains enabled.
Run `check` afterwards to verify that arms saw matching questions and that the
saved hash vectors actually changed between seeds; distinct seed values alone
do not establish distinct permutations or statistical independence.

The study defaults remain:

| Setting | Default |
|---|---|
| Task | `gpqa_diamond_zeroshot` |
| Seeds | `0` through `9` |
| Dtype | `bfloat16` |
| Batch size | `auto` |
| Maximum model length | `4096` |
| GPU memory utilization | `0.5` |
| Eager execution | Enabled |

The legacy `--enforce-eager` flag remains accepted; eager execution is already
enabled by default. Runtime imports and the `VLLM_USE_FLASHINFER_SAMPLER=0` fallback
occur only in workers, retaining a user-supplied value. Importing the runner or
showing help leaves the parent environment unchanged.

Each seed/arm writes `OUTPUT/seedN/ARM/results_TIMESTAMP.json` and its matching
`samples_gpqa_diamond_zeroshot_TIMESTAMP.jsonl`. Results retain the harness metrics
and counts, checkpoint/dtype metadata, arm and seed. Sample records retain their
question IDs, hashes and choice likelihoods for paired analysis. Reruns can add
timestamped files; the analyzer's existing lexical selection rule is unchanged.
The parent continues after a failed arm and returns status 1 if any worker fails.

Orchestration and model execution live in `src/es_capacity/evaluation/gpqa.py`,
with shared command handling in `src/es_capacity/cli/evaluate.py`. Workers launch
the installed module using the current interpreter, without a checkout path.
`python scripts/gpqa/run_sweep.py` remains a compatibility launcher, preserving
the flags and `results/gpqa_results` output default. The core package can show
help and check checkpoint paths, while actual evaluation needs the separate
model runtime. Tests simulate that runtime and check output compatibility with
the analyzer; actual GPU inference and a fresh GPU environment remain unvalidated.

## ETHICS commonsense evaluation

`es-capacity-evaluate ethics --subset commonsense` runs the commonsense morality
task in the existing GPU evaluation environment. It requires the model runtime
described above and SciPy for the automatic statistical summary. The separate
`es-capacity-ethics` command analyses saved predictions without loading models.

```bash
es-capacity-evaluate ethics --subset commonsense --config /data/models.toml --arms 7B-base 7B-ES 7B-RL --csv /data/ethics/commonsense/cm_test.csv --out /data/ethics-run
```

`--csv` selects an existing official test CSV with `label` and `input` columns.
With no CSV selected, the runner reuses `OUTPUT/cm_test.jsonl` if present;
otherwise it reuses or downloads the official `ethics.tar` archive under `--out`
and reads its commonsense test split. Checkpoints are checked before data
preparation or worker startup. An explicit missing CSV is also rejected before
creating output. No dataset download is needed when supplying a CSV.

Task preparation writes `cm_test.jsonl` and `ethics_cm.yaml` under `--out`. The
template is included in the installed package at
`src/es_capacity/evaluation/tasks/ethics_cm.yaml`; it does not depend on a file in
the checkout. The generated YAML uses a quoted absolute data path. Its prompt,
two choices (`no`, `yes`), label mapping and accuracy metric are unchanged.
Scoring ranks those replies by likelihood rather than generating answers.

The orchestration and worker live in
`src/es_capacity/evaluation/ethics_commonsense.py`, with arguments handled in
`src/es_capacity/cli/evaluate.py`. The default arms remain `1.5B-base`, `1.5B-ES`
and `1.5B-RL`; select the 7B arms explicitly. The study defaults remain zero-shot,
seed `0`, dtype `bfloat16`, batch size `1024`, maximum model length `2048`, maximum
batched tokens `2048`, GPU memory utilization `0.25` and eager execution. `--limit`
accepts the existing fraction or scenario-count setting.

Each model runs in a separate process using the installed module and current
interpreter. Workers reuse the prepared JSONL, disable dataset caching and
record per-scenario outputs in `OUTPUT/ARM/samples.jsonl` with metrics and run
metadata in `results.json`. Repeated runs replace those files. After the workers
finish, the existing summary function writes `OUTPUT/summary.csv`, including
accuracy, yes-rate, label-specific accuracies and paired McNemar comparisons.
It retains other model arms already present in that output directory. Scoring,
sample formats, result metadata and worker cleanup are preserved.

The legacy `python scripts/ethics/run.py` command delegates to the installed CLI
and retains its flags and default `results/ethics_results` directory anchored to
the checkout root. The installed command requires `--out` and `--subset`. Importing
the runner or showing help does not load the runtime or change environment
variables. Core-package checks exercise task preparation from the installed wheel;
worker tests simulate model scoring and pass the outputs through the analysis
code. Actual GPU inference remains unvalidated.

## Other ETHICS subsets

```bash
es-capacity-evaluate ethics --subset rest --config /data/models.toml --arms 7B-base 7B-ES 7B-RL --data-dir /data/ethics --out /data/ethics-run
```

`rest` evaluates deontology, justice, virtue and utilitarianism together. The
dataset root must contain `deontology/deontology_test.csv`,
`justice/justice_test.csv`, `virtue/virtue_test.csv` and
`utilitarianism/util_test.csv`. `--data-dir` overrides `ETHICS_DATA_DIR`, then the
existing `/tmp/ethics/ethics` fallback. Missing inputs fail before writing output.
The default model order remains the three 7B arms followed by the three 1.5B arms.

Preparation writes `rest_data/` JSONLs and `rest_tasks/` YAMLs with quoted absolute
data paths. Contrast groups, reply choices and the two orderings of each utility
pair are preserved. Each worker writes `ARM/rest_samples.jsonl` and
`ARM/rest_results.json`; `rest_summary.csv` is refreshed after successful workers.
Commonsense files remain separate. The legacy `scripts/ethics/run_rest.py`
launcher retains its flags and checkout-relative output default. Scoring is
always zero-shot for these four subsets; `--csv` is for commonsense only.

## Model and data paths

The registry in `src/es_capacity/models.py` defines the same six model arms used
in saved outputs; `RL` denotes the GRPO checkpoints. The README links to all six
checkpoint repositories. Users download checkpoint files themselves and provide
local paths. MMLU, GPQA, ETHICS, maths evaluation and checkpoint analysis share the
same resolver.

Copy `config.example.toml` to `config.toml` and edit the `[models]` table. The six
paths are independent; directories can have any names. Absolute paths and `~`
are supported. Relative paths are anchored to the configuration file's directory,
not the shell's current directory. Shell variables such as `$HOME` are not
expanded inside TOML strings. `config.toml` is ignored by Git.

```toml
[models]
"7B-base" = "/data/checkpoints/base"
"7B-ES" = "/another/disk/es-model"
"7B-RL" = "../downloaded-checkpoints/grpo"
```

You may configure a subset and select it with `--arms`. Commands without an arms
selection retain their existing defaults, which can require additional entries.
Check paths before starting an evaluation:

```bash
es-capacity-models check --config config.toml --arms 7B-base 7B-ES 7B-RL
es-capacity-models path 7B-ES --config config.toml
es-capacity-evaluate mmlu --config config.toml --arms 7B-base 7B-ES 7B-RL --out /data/mmlu-run
```

`check` reports every selected path as `OK` or `MISSING`. Missing checkpoint files
return status 1; invalid configuration returns status 2. `path` prints one
validated absolute checkpoint path, making the same resolution available to the
maths shell wrapper. Neither command loads weights or GPU libraries. A valid
local directory must contain `config.json`; these checks do not establish that
all weight shards/tokenizer files are complete or usable by the model runtime.

Configuration selection is explicit:

1. `--config FILE` takes precedence over `ES_CAPACITY_CONFIG`.
2. `ES_CAPACITY_CONFIG` can select a file for every workflow. Use an absolute path
   when running commands from different directories.
3. If neither is set, the previous local-mirror convention under `MODELS_DIR`
   (default `~/models`) remains available. Directory basenames follow the model
   IDs in the registry.

When a file is selected, a missing arm entry is an error. No local path falls
back to a Hugging Face model ID. Parent evaluation processes check all selected
models before dataset preparation and pass the selected config file to workers.
Existing scripts that relied on automatic model downloads must now download the
models themselves and configure their paths.

Maths evaluation uses the same config and requires this package installed in its
own evaluation environment:

```bash
export MATH_EVAL_DIR=/path/to/limit-of-RLVR/math/examples/math_eval
bash scripts/passk/run_eval.sh --config config.toml 7B-base base my-run
```

The optional `config.sh` still supplies shell settings to the maths wrapper.
`config.example.sh` now illustrates the external harness path and config-file
selection; per-model paths belong in TOML. Existing exported model-path variables
and direct local paths remain accepted by the maths wrapper. Its argument is
resolved and checked before it changes directory into the external harness.
Python scripts do not source shell configuration files.

`ETHICS_DATA_DIR` selects the unpacked dataset containing `deontology/`, `justice/`,
`virtue/` and `utilitarianism/`. `scripts/ethics/run_rest.py --data-dir PATH` takes
precedence; the `/tmp/ethics/ethics` default is retained. Output-directory defaults
are unchanged. Analysis of saved outputs does not require model configuration.

The write-up's grader-audit workflow is currently excluded from the published
repository. Making that workflow reproducible is a separate follow-up; the packaging work does not add the ignored local scripts or raw benchmark text.

## Review and validation

Keep environment changes, code movement and scoring changes in separate commits.
Tests use standard-library `unittest` with the analysis dependencies and require
no GPU libraries. They cover configuration, synthetic summary outputs, and the
boundary between analysis and model execution. Evaluation regressions also cover
question-ID and sample-budget checks, complete ETHICS groups and paired IDs,
matching GPQA result/sample files, fresh dataset preparation and stored histogram
grids. Run the full suite with the analysis group enabled:

```bash
uv lock --check
uv sync --locked --group analysis
uv run --locked --group analysis python -m unittest discover -s tests -v
uv run --locked --group analysis es-capacity-ethics --help
uv run --locked --group analysis es-capacity-models --help
uv run --locked --group analysis es-capacity-mmlu --help
uv run --locked --group analysis es-capacity-gpqa --help
uv run --locked --group analysis es-capacity-weights --help
uv run --locked --group analysis es-capacity-evaluate mmlu --help
uv run --locked --group analysis es-capacity-evaluate gpqa --help
uv run --locked --group analysis es-capacity-evaluate ethics --help
uv run --locked --group analysis python scripts/passk/analyze.py --help
uv run --locked --group analysis python scripts/mmlu/tables.py --help
uv run --locked --group analysis python scripts/gpqa/analyze.py --help
uv run --locked --group analysis python scripts/gpqa/tables.py --help
uv run --locked --group analysis python scripts/weights/plot.py --help
git diff --check
```

Also regenerate the pass@k figure in a disposable copy of the checkout to check
that the locked plotting environment can consume the committed summaries. These
are installation smoke checks, not a full validation of statistical correctness.
For the ETHICS extraction, also compare both summary CSVs byte for byte against
the previous implementation using the same saved samples in disposable directories.
When changing evaluation analysis, rerun the saved-data checks in disposable
directories: compare complete-input pass@k and ETHICS summaries before and after
the change, validate GPQA run/sample matching, and check histogram coordinates
against the serialized bin edges. Raw experiment files are not included in Git;
CI covers these contracts with synthetic regression inputs.

## Continuous integration

The [CI workflow](.github/workflows/ci.yml) runs when a pull request targeting
`main` is opened, reopened or receives new commits, and on every push to `main`,
including merges. Draft pull requests are included. A push to another branch
without an open pull request does not trigger it. GitHub reports the result on
the pull request; requiring that check before merging is a separate repository
setting ([GitHub workflow events](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#pull_request)).

CI uses Ubuntu 24.04 and uv 0.10.3, with separate jobs for Python 3.10 (the declared
minimum) and Python 3.12 (the default selected by `.python-version`). Each job
installs the locked analysis environment, runs the full test suite and checks
analysis command help. It then builds the source distribution and wheel, installs
the wheel into a fresh temporary environment, and runs the tests from the source
distribution against that installation. Before adding the analysis extra, the
package check runs the installed model-config command on a synthetic checkpoint
directory, including the conditional TOML parser on Python 3.10. These checks
need no raw benchmark outputs or downloaded model checkpoints.

The final check named `CI` runs even if a matrix job fails or is skipped and
succeeds only when every Python job succeeds. Use this stable check name in
`main` branch protection, with the branch required to be up to date. Repository
settings are managed separately from the workflow file. The matrix continues
both Python jobs after a failure so their individual results remain visible.

Run the commands in **Review and validation** above, followed by the same package
check used in CI:

```bash
uv run --locked --group analysis python scripts/check_package.py
```

To reproduce the minimum-version job without replacing this checkout's normal
`.venv`, select a separate environment directory:

```bash
UV_PYTHON=3.10 UV_PROJECT_ENVIRONMENT=/tmp/es-capacity-py310 uv sync --locked --group analysis
UV_PYTHON=3.10 UV_PROJECT_ENVIRONMENT=/tmp/es-capacity-py310 uv run --locked --group analysis python -m unittest discover -s tests -v
UV_PYTHON=3.10 UV_PROJECT_ENVIRONMENT=/tmp/es-capacity-py310 uv run --locked --group analysis python scripts/check_package.py
```

The package check uses temporary build and installation directories and removes
them on exit. Use `--out-dir NEW_DIRECTORY` to retain the exact validated wheel
and source archive with a `SHA256SUMS` file after all checks pass; an existing
directory is rejected. Its analysis dependencies are constrained to `uv.lock`. A failure
returns a nonzero exit code and prints the failing command. CI checks the current
software contracts; it does not rerun model evaluation or establish the study's
scientific conclusions.

Workflow actions are pinned to commit IDs, the job has read-only repository
permissions, and a newer commit cancels an older run for the same PR or branch.
Update action pins and the uv version deliberately and validate the workflow when
changing them.

## Package builds

`uv build` produces a wheel and a source distribution in `dist/`, using the pinned
Hatchling build backend. The wheel contains the shared package, installed analysis
and model-configuration commands, the synthetic pass@k example and dependency metadata;
model weights, evaluation outputs and the write-up are excluded. The source
distribution also includes the script entry points, tests and setup documentation.
It can rebuild the wheel without a Git checkout.

```bash
uv build
```

For an installation outside the checkout, install the local wheel with its
analysis extra into a separate environment:

```bash
python -m pip install '/path/to/es_capacity-0.1.0-py3-none-any.whl[analysis]'
es-capacity-ethics --results-dir /path/to/ethics_results --subset commonsense
es-capacity-ethics --results-dir /path/to/ethics_results --subset rest
```

The extra makes the analysis dependencies available to standard installers
([uv dependency documentation](https://docs.astral.sh/uv/concepts/projects/dependencies/#optional-dependencies)).
Installing a wheel this way uses its dependency bounds, not `uv.lock`; use the
locked checkout environment above to reproduce the validated dependency versions.

Validate the wheel by installing it into a fresh environment and importing
`es_capacity` from a directory outside this checkout. Editable-install tests alone
do not establish that the built distribution includes the required modules.

For versioned downloads, see the [release notes](CHANGELOG.md) and
[GitHub release procedure](RELEASING.md). Releases are prepared in a PR, then
tagged and published from the validated merged commit.
