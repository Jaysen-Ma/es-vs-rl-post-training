# Development and reproduction

The `es_capacity` package provides the shared model registry, filesystem
configuration and saved-result ETHICS analysis. Other analysis commands and
model runners remain in `scripts/`. Model execution requires a separate GPU
environment.

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
uv run --locked --group analysis python scripts/passk/analyze.py --help
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
configuration and runner CLI wiring have also been checked on CPython 3.10.19,
including the `tomli` fallback. Analysis on other Python versions and other
platforms has not yet been validated.

To deliberately update dependencies, use `uv lock --upgrade-package NAME`, repeat
the relevant validation, and commit the lockfile change with its reason. Routine
setup should use `--locked`.

## What can be reproduced from a clone

| Task | Command | Required inputs |
|---|---|---|
| Paper's pass@k figure | `uv run --locked --group analysis python scripts/plot_hero.py` | The committed three-way summary JSON files |
| pass@k summaries | `uv run --locked --group analysis python scripts/passk/analyze.py --run-tag RUN --label base --label trained --label rl --baseline base --plot` | Raw per-question evaluation JSONL files under `results/RUN/` |
| MMLU tables | `uv run --locked --group analysis python scripts/mmlu/tables.py --sweep-root PATH` | Saved MMLU results and sample files |
| ETHICS commonsense summary | `uv run --locked --group analysis es-capacity-ethics --results-dir PATH --subset commonsense` | `PATH/ARM/samples.jsonl` for each model arm |
| Other four ETHICS summaries | `uv run --locked --group analysis es-capacity-ethics --results-dir PATH --subset rest` | `PATH/ARM/rest_samples.jsonl` for each model arm |
| GPQA summaries | `uv run --locked --group analysis python scripts/gpqa/analyze.py --sweep-root PATH --out OUTPUT` | Saved GPQA seed-sweep outputs |
| GPQA tables | `uv run --locked --group analysis python scripts/gpqa/tables.py --dir PATH` | Reduced GPQA CSV files |
| Weight-change plots | `uv run --locked --group analysis python scripts/weights/plot.py --scale 7B --dir PATH` | Weight-statistics and histogram CSV files |

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
python scripts/mmlu/run.py --config config.toml --arms 7B-base 7B-ES 7B-RL
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

CI uses Ubuntu 24.04, Python 3.12 from `.python-version`, and uv 0.10.3. It installs
the locked analysis environment, runs the test suite and checks analysis command
help. It then builds the source distribution and wheel, installs the wheel into
a fresh temporary environment, and runs the tests from the source distribution
against that installation. The package check also verifies that core command help
works without analysis dependencies and that the analysis extra installs without
GPU libraries. These checks use synthetic predictions and need no raw benchmark
outputs or model checkpoints.

Run the commands in **Review and validation** above, followed by the same package
check used in CI:

```bash
uv run --locked --group analysis python scripts/check_package.py
```

The package check uses temporary build and installation directories and removes
them on exit. Its analysis dependencies are constrained to `uv.lock`. A failure
returns a nonzero exit code and prints the failing command. CI checks the current
software contracts; it does not rerun model evaluation or establish the study's
scientific conclusions.

Workflow actions are pinned to commit IDs, the job has read-only repository
permissions, and a newer commit cancels an older run for the same PR or branch.
Update action pins and the uv version deliberately and validate the workflow when
changing them.

## Package builds

`uv build` produces a wheel and a source distribution in `dist/`, using the pinned
Hatchling build backend. The wheel contains the shared package, ETHICS command
and dependency metadata;
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
