# Development and reproduction

The `es_capacity` package provides the shared model registry, filesystem
configuration and saved-result ETHICS analysis. Other analysis commands and
model runners remain in `scripts/`. Model execution and ES training require
separate environments.

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
minimum is 3.10 because the scripts use Python 3.10 syntax. Other Python versions
and platforms have not yet been validated.

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

## Model execution and external repositories

`external-repositories.toml` records the exact local checkout revisions inspected
on 2026-11-05. These are setup reference points, not evidence that the historical
experiments ran at precisely those commits. The experiment records do not yet
establish that provenance. Branch names are hints; use the recorded commit IDs
when reconstructing these checkouts.

The external repositories currently declare incompatible dependency versions:

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

Use separate environments for training and maths evaluation. The local MMLU,
GPQA and ETHICS runners additionally depend on the evaluation harness (`lm_eval`),
Hugging Face Datasets and vLLM. They are not covered by `uv.lock` yet. Setup for
those workflows must be validated on suitable hardware before it is described
as reproducible.

Install the shared package into an existing GPU environment with:

```bash
python -m pip install --no-deps -e .
```

The core installation has no mandatory third-party dependencies. This command
adds shared configuration and analysis code without installing or upgrading the
GPU runtime. ETHICS statistical summaries still require SciPy in that environment.
Keep using its interpreter for `python scripts/...` commands.

## Model and data paths

The registry in `src/es_capacity/models.py` defines the same six model arms used
in saved outputs. MMLU, GPQA, ETHICS and checkpoint analysis share these definitions.
The existing `RL` labels are retained for compatibility.

- `MODELS_DIR` selects the local checkpoint mirror; the default is `~/models`.
  Evaluation runners use a checkpoint under that directory if its `config.json`
  exists, and otherwise use its Hugging Face model ID. The previous basename
  matching rule is retained for custom IDs and paths supplied through `--arms`.
- Checkpoint analysis requires local files and reports a missing checkpoint
  rather than downloading one.
- `ETHICS_DATA_DIR` selects the unpacked directory containing `deontology/`,
  `justice/`, `virtue/` and `utilitarianism/`. The old `/tmp/ethics/ethics` default
  is retained. `scripts/ethics/run_rest.py --data-dir PATH` takes precedence and
  is forwarded to worker processes.

These environment variables are read when paths are requested. For example:

```bash
export MODELS_DIR=/path/to/models
export ETHICS_DATA_DIR=/path/to/ethics/ethics
python scripts/ethics/run_rest.py --arms 7B-base --data-dir /another/ethics/ethics
```

The Python scripts do not automatically source `config.sh`. Export variables in
the shell or explicitly source the file. The mathematical harness remains a shell
wrapper: copy `config.example.sh` to `config.sh` and set `MATH_EVAL_DIR` and its
per-model variables as before. Output-directory defaults are unchanged in this
step; the shared package does not infer a repository root from its installation.

The write-up's grader-audit workflow is currently excluded from the published
repository. Making that workflow reproducible is a separate follow-up; the packaging work does not add the ignored local scripts or raw benchmark text.

## Review and validation

Keep environment changes, code movement and scoring changes in separate commits.
Tests use standard-library `unittest` and require no GPU libraries. The ETHICS
tests check summary output on small synthetic predictions, command errors, and
the boundary between analysis and model execution. Summary-output tests require
the analysis dependencies; run the full suite with that group enabled:

```bash
uv lock --check
uv sync --locked --group analysis
uv run --locked --group analysis python -m unittest discover -s tests -v
uv run --locked --group analysis es-capacity-ethics --help
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
Previously withheld scoring and sample-matching fixes remain a separate review.

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
