# Making a GitHub release

**TLDR:** For maintainers publishing a new version on GitHub. Choose a version,
build and check the package, then upload the tested files. You do not need these
steps to run the project.

The release process publishes downloads on GitHub, with no PyPI publication.

## What the terms mean

| Term | Meaning here |
|---|---|
| Version | A label such as `0.2.0`, recorded in `pyproject.toml` |
| Tag | A Git name such as `v0.2.0` pointing to the exact released commit |
| GitHub release | A page attached to that tag, with notes and downloadable files |
| Wheel (`.whl`) | The installable Python package and commands |
| Source archive (`.tar.gz`) | Package source, tests, configuration example and guides |
| `SHA256SUMS` | File fingerprints used to check that downloaded files are unchanged |

Checkpoints, benchmark data, saved model answers, the paper and study results
are outside these package downloads. A clone includes the paper and committed
evaluation summaries. Model execution also needs its separate runtime environment.

## 1. Prepare a version

Choose a version/tag that has not been used. The commands below use `0.2.0` as an
example; replace it with the version you are preparing.

In a pull request, update `project.version` in `pyproject.toml`, run `uv lock`,
and update version numbers in installation examples. Describe the changes and
validation limits in the PR. A small bug fix usually increments the last number;
new functionality increments the middle number. Document any breaking changes.

Merge the reviewed PR into `main`, then wait for GitHub's automated `CI` check on
that merged commit to pass.

## 2. Build and check the downloads

Use a new checkout so the release comes from committed code. Use Python 3.12 and
uv 0.10.3, matching CI:

```bash
git clone --branch main https://github.com/Jaysen-Ma/es-vs-rl-post-training.git es-capacity-release
cd es-capacity-release
git status --short
git rev-parse HEAD
uv sync --locked --group analysis
uv run --locked --group analysis python scripts/check_package.py --out-dir dist
```

Before building, `git status --short` should be empty and `HEAD` should be the
merged commit whose CI passed. The package checker builds the downloads, installs
the wheel in a clean environment, runs tests and exercises installed commands.
It saves files in `dist/` only after those checks pass. `dist/` must not already
exist; use a fresh checkout rather than mixing files from different builds.

For version `0.2.0`, the three files are:

```text
es_capacity-0.2.0-py3-none-any.whl
es_capacity-0.2.0.tar.gz
SHA256SUMS
```

Upload these exact checked files. Do not rebuild between checking and uploading.

## 3. Name the commit and publish

From that same checkout, create and push the tag:

```bash
git tag -a v0.2.0 -m "Release v0.2.0"
git push origin refs/tags/v0.2.0
```

Create a draft release on GitHub using that tag. Write short notes explaining
what changed and what has been validated, then attach the three files from
`dist/`. Review the tag, notes and attachments before publishing the draft.

Keep released tags and files unchanged. Corrections get a new version.

For user installation instructions, see
[installing a downloaded package](reproduction.md#install-a-downloaded-package).
