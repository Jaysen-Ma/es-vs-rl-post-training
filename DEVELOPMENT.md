# Development and reproduction

The first packaging step defines an environment for the existing analysis
scripts. The code is still run from this checkout; it is not yet an installable
Python library. Model execution and ES training require separate environments.

## Analysis environment

Analysis means processing saved model outputs: computing pass@k, making paired
statistical comparisons, and producing tables and plots. It does not generate
new model responses. The `analysis` dependency group contains NumPy, SciPy and
Matplotlib and does not install PyTorch, vLLM or model checkpoints.

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and use Python
3.12, as selected by `.python-version`. From the repository root:

```bash
uv sync --locked --group analysis
uv run --locked --group analysis python scripts/passk/analyze.py --help
```

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
| GPQA summaries | `uv run --locked --group analysis python scripts/gpqa/analyze.py --sweep-root PATH --out OUTPUT` | Saved GPQA seed-sweep outputs |
| GPQA tables | `uv run --locked --group analysis python scripts/gpqa/tables.py --dir PATH` | Reduced GPQA CSV files |
| Weight-change plots | `uv run --locked --group analysis python scripts/weights/plot.py --scale 7B --dir PATH` | Weight-statistics and histogram CSV files |

The figure command overwrites `writeup/figures/hero_passk.png` and
`writeup/figures/hero_passk.pdf`. Its numerical inputs are committed, but rendered
files can differ across plotting environments. The other commands require data
that is excluded from Git; a fresh clone cannot regenerate every paper result
without obtaining or regenerating those inputs. The source raw evaluation data
should be retained separately.

ETHICS summary functions also operate on saved predictions, but currently live
inside the model-running scripts. This dependency group does not make those
scripts GPU-free entry points; separating them is part of the later refactor.
Checkpoint comparison (`scripts/weights/compare.py`) requires PyTorch and
safetensors and is also outside this analysis environment.

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

For the mathematical harness, copy `config.example.sh` to `config.sh` and set
`MATH_EVAL_DIR` and the model locations. The other runners currently use
`MODELS_DIR` or Hugging Face model IDs. ETHICS' four additional subsets expect an
unpacked dataset under `/tmp/ethics/ethics`; the shared-path refactor will make
that explicit and configurable.

The write-up's grader-audit workflow is currently excluded from the published
repository. Making that workflow reproducible is a separate follow-up; this
commit does not add the ignored local scripts or raw benchmark text.

## Review and validation

Keep environment changes, code movement and scoring changes in separate commits.
Before committing an environment update:

```bash
uv lock --check
uv sync --locked --group analysis
uv run --locked --group analysis python scripts/passk/analyze.py --help
uv run --locked --group analysis python scripts/mmlu/tables.py --help
uv run --locked --group analysis python scripts/gpqa/analyze.py --help
uv run --locked --group analysis python scripts/gpqa/tables.py --help
uv run --locked --group analysis python scripts/weights/plot.py --help
git diff --check
```

Also regenerate the pass@k figure in a disposable copy of the checkout to check
that the locked plotting environment can consume the committed summaries. These
are installation smoke checks, not validation of statistical correctness. Tests
for the scoring and sample-matching rules will be reviewed separately from this
environment change.
