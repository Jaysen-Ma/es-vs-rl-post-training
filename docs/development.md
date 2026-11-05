# Contributor guide

The package separates saved-result analysis, model execution and command-line
handling. Start with [reproduction](reproduction.md) to understand data contracts,
or [model evaluation](model_evaluation.md) for checkpoint/runtime setup.

## Repository map

| Location | Purpose |
|---|---|
| `src/es_capacity/config.py`, `models.py` | Shared path configuration and the six model arms |
| `src/es_capacity/analysis/` | Saved-output statistics, plotting and checkpoint comparison |
| `src/es_capacity/evaluation/` | Model workers and benchmark task preparation |
| `src/es_capacity/cli/` | Installed commands, arguments and user-facing errors |
| `src/es_capacity/evaluation/tasks/` | Task resources shipped in the wheel |
| `tests/` | Synthetic regression inputs and tests; outside the installed package |
| `scripts/` | Compatibility launchers, external maths wrapper and maintenance utilities |
| `results/`, `writeup/` | Committed study summaries, figures and paper |
| `.github/workflows/` | The same analysis/package checks required for merging |

## Making changes

Keep numerical functions in `analysis/`, argument handling in `cli/`, and model
runtime imports inside evaluation workers. Use the shared model resolver rather
than constructing checkpoint paths independently. Model runners must check selected
paths before preparing data or starting workers. Preserve compatibility launchers
when moving an existing command.

To add a model arm, update the registry, configuration example and download table,
then add a configuration test. To add an analysis or benchmark, document its input
files, units, output schema and reproduction boundaries. Cover meaningful numerical
or workflow contracts with synthetic fixtures and independent references; do not
commit benchmark text, model weights or generated answers. Add required non-code
resources to the wheel and exercise them in `scripts/check_package.py`.

Run the relevant tests while editing, then the full suite and fresh-wheel check
before pushing. Keep logically separate migrations in separate commits. Optional
CPU checkpoint tests require PyTorch/SafeTensors and are described in
[checkpoint comparison](model_evaluation.md#checkpoint-comparison). GPU inference
validation remains outside the lightweight CI suite.

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
uv run --locked --group analysis es-capacity-compare-weights --help
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

The [CI workflow](../.github/workflows/ci.yml) runs when a pull request targeting
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
Hatchling build backend. The wheel contains the shared package, installed analysis,
evaluation and model-configuration commands, task resources, the synthetic pass@k
example and dependency metadata;
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

For versioned downloads, see the [release notes](../CHANGELOG.md) and
[GitHub release procedure](../RELEASING.md). Releases are prepared in a PR, then
tagged and published from the validated merged commit.
