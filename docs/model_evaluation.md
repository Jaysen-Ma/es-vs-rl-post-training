# Models and evaluation

Use this guide to evaluate checkpoints you download yourself, or to compare their
weights. No training step is required. Model execution needs a separate GPU
evaluation environment; checkpoint comparison runs on the CPU with PyTorch and
SafeTensors. Neither runtime is installed by the saved-result analysis extra.

For existing predictions, use [saved-result analysis](reproduction.md). The
[contributor guide](development.md) explains the source layout and tests.

## Download checkpoints

Download the full checkpoint directories, including weights, model configuration
and tokenizer files, from the six study sources:

| Model arm | Download |
|---|---|
| `1.5B-base` | [Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B) |
| `1.5B-ES` | [Qwen2.5-1.5B-ES-math](https://huggingface.co/zocrate/Qwen2.5-1.5B-ES-math) |
| `1.5B-RL` | [Qwen-2.5-1.5B-SimpleRL-Zoo](https://huggingface.co/hkust-nlp/Qwen-2.5-1.5B-SimpleRL-Zoo) |
| `7B-base` | [Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B) |
| `7B-ES` | [Qwen2.5-7B-ES-math](https://huggingface.co/zocrate/Qwen2.5-7B-ES-math) |
| `7B-RL` | [Qwen-2.5-7B-SimpleRL-Zoo](https://huggingface.co/hkust-nlp/Qwen-2.5-7B-SimpleRL-Zoo) |

`RL` denotes the GRPO checkpoints. Download only the arms you intend to select;
configure independent local paths using [config.example.toml](../config.example.toml).
The following configuration mechanism is shared by every model workflow.

## Model and data paths

The registry in `src/es_capacity/models.py` defines the same six model arms used
in saved outputs; `RL` denotes the GRPO checkpoints. The table above links to all six
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
`virtue/` and `utilitarianism/`. `es-capacity-evaluate ethics --subset rest --data-dir PATH` takes
precedence; the `/tmp/ethics/ethics` default is retained. Output-directory defaults
are unchanged. Analysis of saved outputs does not require model configuration.

The write-up's grader-audit workflow is currently excluded from the published
repository. Making that workflow reproducible is a separate follow-up; the packaging work does not add the ignored local scripts or raw benchmark text.

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
its interpreter for installed commands or `python scripts/...`. Users evaluate downloaded
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
