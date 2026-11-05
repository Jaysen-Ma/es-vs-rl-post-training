# GitHub releases

The first release is `v0.1.0`, matching `project.version = "0.1.0"` in
`pyproject.toml`. Releases provide a wheel, source archive and `SHA256SUMS` on
GitHub. There is no PyPI publication step.

## Prepare the release in a pull request

Update the version in `pyproject.toml`, refresh `uv.lock`, update installation
examples containing the version and add release notes to `CHANGELOG.md`. The first
release already uses `0.1.0`. Subsequent bug fixes increment the patch version;
new functionality increments the minor version. Before 1.0, document any breaking
changes explicitly and increment the minor version for them.

Review the release notes, package contents and supported environments. Require
the PR's `CI` check to pass. The repository owner merges the PR into `main`.
Release preparation on a branch does not create a release or tag.

## Build from the merged commit

Use a clean checkout of `main`. Confirm that the `CI` run for that exact commit
has passed on GitHub; a passing PR run alone is not the release check.

```bash
git clone --branch main https://github.com/Jaysen-Ma/es-vs-rl-post-training.git es-capacity-release
cd es-capacity-release
git status --short
git rev-parse HEAD
uv sync --locked --group analysis
uv run --locked --group analysis python -m unittest discover -s tests -v
uv run --locked --group analysis python scripts/check_package.py --out-dir dist
```

Use uv 0.10.3 and Python 3.12, as in CI. The package check builds the wheel from
the source archive, installs the wheel into a separate environment, runs the
shipped tests against it, and exercises the installed commands. Only after all
checks pass does it copy those exact artifacts into `dist/` and write SHA-256
checksums. An existing output directory is rejected to avoid mixing candidates.
Do not run another build between validation and upload.

For `0.1.0`, the directory must contain exactly:

```text
es_capacity-0.1.0-py3-none-any.whl
es_capacity-0.1.0.tar.gz
SHA256SUMS
```

Verify the checksums from that directory with `sha256sum --check SHA256SUMS`.
The CI package checks exercise the same export and checksum verification on both
supported Python versions. Dependency versions are constrained to `uv.lock`
during validation; ordinary wheel installations use the declared dependency
bounds unless the user also supplies constraints.

## Tag and review a draft

After confirming the merged commit and successful checks, create an annotated
tag at the checked-out commit and push only that tag:

```bash
git tag -a v0.1.0 -m "Release v0.1.0"
git push origin refs/tags/v0.1.0
gh release create v0.1.0 --verify-tag --draft --title "v0.1.0" --notes-file CHANGELOG.md dist/es_capacity-0.1.0-py3-none-any.whl dist/es_capacity-0.1.0.tar.gz dist/SHA256SUMS
```

For later releases, supply only the new version's notes in `--notes-file` rather
than uploading the entire cumulative changelog. The
[GitHub CLI release command](https://cli.github.com/manual/gh_release_create)
supports draft creation and requires the existing tag with `--verify-tag`.

Review the draft's tag, notes and three assets on GitHub. Download the assets
into an empty directory and verify `SHA256SUMS` there. Confirm that the tag's
commit matches the validated checkout. Publish the reviewed draft on GitHub.
Never move a published tag or replace its assets; correct problems in a new
version.

## Install a release

Download all three assets from the chosen GitHub release into one directory and
verify them before installing the wheel into a separate virtual environment:

```bash
sha256sum --check SHA256SUMS
python3 -m venv .venv
.venv/bin/python -m pip install './es_capacity-0.1.0-py3-none-any.whl[analysis]'
.venv/bin/es-capacity-passk --example --out example-output --plot
```

These commands target Linux, the validated platform. Users who need the locked
environment or checkout-specific scripts can extract the source archive and
follow `DEVELOPMENT.md`. Model execution still requires a separate GPU
environment, which has not been validated for this release.
