# Evaluation results

This directory contains compact evaluation summaries for the six study
checkpoints: Qwen2.5-1.5B and Qwen2.5-7B, each with base, ES and GRPO arms.
`RL` in filenames/tables denotes GRPO. Training curves, trainer logs and
in-training evaluations stay local and are not part of this public results tree.

| Location | Contents |
|---|---|
| [passk/1.5B](passk/1.5B/) and [passk/7B](passk/7B/) | Four maths benchmarks per size, each with a three-model summary JSON and plot |
| [mmlu/scores.csv](mmlu/scores.csv) | Six model scores, sample counts and paired comparisons |
| [ethics/summary.csv](ethics/summary.csv) | Commonsense results for all six models |
| [ethics/rest_summary.csv](ethics/rest_summary.csv) | Deontology, justice, virtue and utilitarianism results for all six models |
| [weights/figures](weights/figures/) | Supplementary checkpoint-weight comparison figures |

These are existing study results. The pass@k files were moved without numerical
changes. The MMLU and ETHICS CSVs were regenerated from the saved study outputs
with the packaged analysis commands and checked against the existing summaries.
No new model inference was run for this reorganisation.

The pass@k JSONs retain the study's three-way schema: `pass_at_k_base`,
`pass_at_k_es`, `pass_at_k_rl`, their sampling budgets and k grids, and coverage
breakdowns. Pass@k values are fractions; breakdown fields ending in `_pct` are
percentages. `scripts/plot_hero.py` reads these files to regenerate the combined
paper figure. New analyses with `es-capacity-passk` use its generic labelled
summary schema, described in the [reproduction guide](../docs/reproduction.md).

MMLU/ETHICS accuracy columns are fractions from 0 to 1. For deontology, justice
and virtue, `set_acc` measures whole-group correctness. For utilitarianism it
compares preference scores across both scenario orders; `example_acc` uses the
canonical order. ETHICS p-values retain their
existing display precision; a value printed as zero is rounded. Interpretation
and multiple-comparison correction are described in the paper.

Benchmark text, individual model responses and raw result directories are not
included. The aggregate files support inspection of the reported results; they
cannot reproduce paired statistics without the original per-question outputs.
Use the [evaluation guide](../docs/model_evaluation.md) to evaluate downloaded
checkpoints and the [reproduction guide](../docs/reproduction.md) to analyse saved
outputs. Select a separate output directory for new runs.

Git ignores results by default and allows only the summary/figure paths above.
Review the allowlist in `.gitignore` when publishing another reduced artifact.
