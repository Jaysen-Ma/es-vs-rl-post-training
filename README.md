<div align="center">
<h1>What Post-Training Changes: Reasoning Gains and Safety Side Effects of ES vs RL</h1>

[Jaysen Ma](https://github.com/Jaysen-Ma)

<a href="writeup/writeup.pdf"><img src='https://img.shields.io/badge/Write--up-PDF-red' alt='Write-up PDF'></a> <a href="https://huggingface.co/zocrate/Qwen2.5-7B-ES-math"><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-checkpoints-blue' alt='Checkpoints'></a>
</div>

## Overview

Post-training with group relative policy optimisation (GRPO) can make a language model more likely to solve a maths problem on its first attempt, while narrowing the range of problems it can solve under repeated sampling ([Yue et al.](https://arxiv.org/abs/2504.13837)). Is this narrowing specific to GRPO, or does it also occur with alternative post-training methods?

The narrowing observed after maths-only RL post-training with GRPO raises a broader question: how much do different post-training methods change model behaviour beyond mathematical performance? In particular, do they also affect moral judgement, and what might those effects imply for safety and alignment as training is scaled up?

[Evolution strategies (ES)](https://arxiv.org/abs/2509.24372) offer a different approach to RL post-training, using reward-guided parameter perturbations without backpropagation. Comparing ES with GRPO allows us to investigate whether these methods differ in both their reasoning coverage and their effects on moral judgement.

We compare Qwen2.5-1.5B and Qwen2.5-7B base models with ES and GRPO checkpoints trained on the same set of 8,523 maths problems. We evaluate mathematical performance across sampling budgets, general knowledge on MMLU, and moral judgement on ETHICS.

### Reasoning capacity (pass@k)

We compare each post-trained model with its base using pass@k: the probability of obtaining at least one correct answer in k attempts.

We evaluate four maths benchmarks, with up to 512 attempts per question on AIME24 and 128 on MATH500, Minerva Math and OlympiadBench.

![pass@k, Qwen2.5-1.5B (top) and 7B (bottom)](writeup/figures/hero_passk.png)

> *Figure: pass@k of base, ES and GRPO (SimpleRL-Zoo) on AIME24, MATH500, Minerva Math and OlympiadBench; top row Qwen2.5-1.5B, bottom row Qwen2.5-7B.*

**At 1.5B, both methods improve performance across the measured sampling budgets.** ES and GRPO outperform the base model at every evaluated k on all four benchmarks. We find no evidence of narrowing within this range.

**At 7B, stronger first-attempt performance does not always translate into broader coverage.** GRPO achieves the highest pass@1 on all four benchmarks, but falls below the base model on each at the largest evaluated k. ES matches or exceeds the base at the largest k on **3 out of 4 tasks**, falling below it on AIME24.

Narrowing is therefore not exclusive to GRPO in these experiments, although it appears more consistently across benchmarks for GRPO than for ES. The difference between the 1.5B and 7B results also raises a question about when narrowing emerges. With one training run per method at each size, unequal training budgets and different reward graders, we cannot isolate the effects of model size or training method.

### Moral judgement (ETHICS)

Maths-only post-training can also affect behaviour outside maths. To examine this, we evaluate all six models on the five [ETHICS](https://arxiv.org/abs/2008.02275) subsets: commonsense, deontology, justice, virtue and utilitarianism. These tasks assess agreement with labelled judgements about moral scenarios; none was used for post-training.

**ES reduces accuracy on some ETHICS subsets at both 1.5B and 7B.** At 1.5B, ES lowers scores on deontology, justice and virtue. At 7B, it lowers commonsense accuracy. All four declines are statistically significant after correcting for multiple comparisons.

GRPO shows no statistically significant declines on ETHICS at either model size.

These changes occur despite stable general knowledge performance: all post-trained models remain within 0.5 percentage points of their base on MMLU, with no statistically significant differences.

| Subset | Chance | 1.5B Base | 1.5B ES | 1.5B GRPO | 7B Base | 7B ES | 7B GRPO |
|---|---|---|---|---|---|---|---|
| Commonsense | 50.0* | 52.79 | 53.23 | 52.23 | 63.42 | **57.68** | 63.40 |
| Deontology | 6.25 | 9.8 | **4.1** | **12.3** | 24.5 | 23.9 | 24.0 |
| Justice | 6.25 | 16.0 | **7.0** | 16.3 | 29.1 | 31.7 | 30.5 |
| Virtue | 3.1 | 46.8 | **42.9** | 47.0 | 56.9 | 56.5 | 57.0 |
| Utilitarianism | 50.0 | 51.1 | 52.5 | 52.4 | 73.5 | 72.5 | 73.7 |

Scores are zero-shot accuracies (%), based on the relative likelihoods of two candidate replies. **Bold** entries indicate a statistically significant change from the corresponding base model under an exact McNemar test with Bonferroni correction across 20 comparisons (p < 0.0025).

Deontology, justice and virtue are scored in groups of four, four and five related items; a group counts as correct only when every item is answered correctly. Utilitarianism compares preferences across both scenario orders. \*For commonsense, always answering “not wrong” achieves 53.3%. Full p-values are reported in the [write-up](writeup/writeup.pdf).

These results show that maths-only evolutionary post-training can improve mathematical performance and preserve general knowledge scores while reducing accuracy on moral judgement tasks. The declines observed with ES motivate investigating how these effects develop as evolutionary training scales up, and what they might imply for safety and alignment. The present experiments do not establish that trajectory, and ETHICS scores capture only a narrow aspect of behaviour relevant to these questions.

## Limitations

- One training run per method.
- Not compute-matched: GRPO saw 8x the data and 2x the generations; ES ran 1/10 of the recommended iterations.
- Grader confound: GRPO was trained with the same lenient answer extraction used for eval; ES was rewarded for `\boxed{}` only. Re-scoring with eight answer graders gives the same direction at 7B.

## Evaluation

For analysis of saved outputs, install [uv](https://docs.astral.sh/uv/getting-started/installation/) and use the locked Python environment:

```bash
uv sync --locked --group analysis
uv run --locked --group analysis python scripts/plot_hero.py
```

This regenerates the pass@k figure from committed summaries. Model evaluation and training require separate GPU environments. See [Development and reproduction](DEVELOPMENT.md) for input requirements, environment boundaries and the recorded external repository revisions.

```bash
cp config.example.sh config.sh   # fill in MATH_EVAL_DIR and model paths
scripts/passk/run_eval.sh QWEN25_7B_BASE base my-run   # pass@k (needs Jaysen-Ma/limit-of-RLVR, branch fix/math-equal-timeout-bypass)
python scripts/ethics/run.py --arms <arms>        # ETHICS commonsense
python scripts/ethics/run_rest.py --arms <arms>   # other four ETHICS subsets
```

ES training uses [Jaysen-Ma/es-at-scale](https://github.com/Jaysen-Ma/es-at-scale) (branch `fix/multi-engine-colocation`). MMLU, GPQA and weight-change scripts are in `scripts/mmlu`, `scripts/gpqa`, `scripts/weights`.

**Checkpoints:** [zocrate/Qwen2.5-1.5B-ES-math](https://huggingface.co/zocrate/Qwen2.5-1.5B-ES-math) · [zocrate/Qwen2.5-7B-ES-math](https://huggingface.co/zocrate/Qwen2.5-7B-ES-math)

## Acknowledgements

Built on [VsonicV/es-at-scale](https://github.com/VsonicV/es-at-scale) (ES training) and [LeapLabTHU/limit-of-RLVR](https://github.com/LeapLabTHU/limit-of-RLVR) (pass@k evaluation). GRPO checkpoints from [SimpleRL-Zoo](https://github.com/hkust-nlp/simpleRL-reason).

## Citation

```bibtex
@misc{ma2026esvsrl,
  title={What Post-Training Changes: Reasoning Gains and Safety Side Effects of ES vs RL},
  author={Ma, Jaysen},
  year={2026},
  howpublished={\url{https://github.com/Jaysen-Ma/es-vs-rl-post-training}}
}
```

