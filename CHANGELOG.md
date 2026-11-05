# Changelog

Release availability and dates are recorded on the
[GitHub Releases page](https://github.com/Jaysen-Ma/es-vs-rl-post-training/releases).
An entry here can describe a release still being prepared.

## 0.1.0

The first packaged release provides tools for analysing saved evaluation outputs
from the ES and GRPO post-training study.

- Installed `es-capacity-passk` and `es-capacity-ethics` commands compute summaries
  from saved scores and predictions. The pass@k command includes a synthetic
  example that requires no checkpoints or benchmark downloads.
- `es-capacity-models` checks user-supplied local checkpoint paths. The evaluation
  scripts share a TOML configuration for the six model arms. Users download
  checkpoints themselves; the README links to their sources.
- Comparison checks reject mismatched question IDs, incomplete ETHICS groups,
  and missing or inconsistent GPQA result/sample files. Weight plots use the
  histogram bin edges stored with the counts.
- The analysis dependency extra, locked checkout environment, wheel and source
  archive support installations outside the repository. CI tests Python 3.10
  and 3.12, including fresh wheel installations and the bundled example.

Existing script entry points remain available. Valid, complete saved pass@k and
ETHICS inputs retain their numerical outputs. The package does not change the
external maths generation or grading harness.

### Scope

The wheel contains the shared package and installed analysis commands. The source
archive also contains scripts, tests, configuration examples, the dependency lock
and development/release instructions. Neither includes checkpoints, raw benchmark
data, saved generations or the paper. Obtain the full repository for the write-up
and committed study summaries.

GPU evaluation environment validation remains deferred. External repository
revisions are setup references; they do not establish the exact environments of
the historical experiments. This release does not claim full experiment
reproduction or publish a package to PyPI.
