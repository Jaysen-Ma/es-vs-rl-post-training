# ES-capacity

Does Evolution-Strategies (ES) post-training preserve a base model's pass@k
ceiling better than gradient-based RLVR (GRPO)? RLVR reliably raises pass@1
but tends to narrow the pass@k ceiling relative to the base model — this
project tests whether ES-based fine-tuning avoids that narrowing, on the same task/reward/data.

### Method

Two base models — `Qwen2.5-1.5B` and `Qwen2.5-7B`, each compared against two post-trained arms on the same task, reward and data:

- **ES** — full-parameter, backprop-free evolution strategies, σ=0.001,
  population 32, 50 iterations, batch 256, 2,048-token cap. Method and
  training code are Qiu et al.'s
  ([arXiv:2509.24372](https://arxiv.org/abs/2509.24372)). The hyperparameters
  above are **identical at both scales**.
- **RL** — the published SimpleRL-Zoo GRPO checkpoint at the matching scale
  ([arXiv:2503.18892](https://arxiv.org/abs/2503.18892)), trained on the same
  problems but at a much larger budget (see [ES vs. RL](#es-vs-rl)).

Both arms are scored against their own base on AIME24, MATH500, Minerva Math
and OlympiadBench at identical generation settings, on two measures: the
**pass@k curve**, which asks whether a trained model stays above its base as k
grows, and a **per-question solvable/unsolvable breakdown**, which asks how
many questions it gained and how many it lost. That framing — pass@k as a
capacity ceiling RLVR tends to narrow — is Yue et al.'s
([arXiv:2504.13837](https://arxiv.org/abs/2504.13837)).

### Main results

**At 1.5B, neither method narrows anything.** ES and RL both sit above base at
*every* measured k on all four benchmarks — no crossover anywhere — and both
expand question coverage on all four. ES tracks RL at pass@16 on three of the four benchmarks, on ~1/8 the
data budget.

**At 7B, the two arms separate.** RL buys the larger pass@1 gain against ES and pays for it at high k, falling
below base on all four benchmarks. ES
stays at or above base at every k on three of the four benchmarks.

That is an early hint that ES preserves the ceiling better than RLVR. Two things blunt it: the comparison is **not
compute-matched** — the 7B RL checkpoint took ~15h on 2 nodes of 8x H100,
the 7B ES run 4h35m on 8x RTX 4090 48GB; and 50 iterations is 1/10 of what
Qiu et al. recommend, so the ES arm may simply be under-trained. More runs
across models, hyperparameters and datasets are needed before the claim
carries weight.

## **ES vs. RL** 

**Not a compute-matched comparison — and the mismatch is different on each
axis.** Both arms train on the same **8,523 problems** (level3to5), but they
spend their budget on different things:

| | Prompts/step | Steps | ×per prompt | Prompt-exposures | **Epochs** | Generations | Training Token cap |
|---|---|---|---|---|---|---|---|
| **ES** (arxiv2509.24372) | 256 | 50 | 32 population | 12,800 | **1.50** | 409,600 | 2,048 |
| **RL** (arxiv2503.18892) | 1,024 | ~100 | 8 rollouts | 102,400 | **12.01** | 819,200 | 8,192 |
| **ratio (RL / ES)** | 4x | 2x | 0.25x | **8.00x** | **8.00x** | **2.00x** | **4x** |

The two ratios that matter come apart: RL sees **8x more data** but runs only
**2x more generations**. That is the whole structural difference between the
methods. RL spends its generation budget on **data breadth**, ES spends its
budget on **parameter breadth**. 

**On "rollouts" vs "population"**: 

Both multiply the
per-step generation count. GRPO rollout
is a sample from one policy used to estimate an advantage for one prompt, and
the actual descent direction comes from backprop. An ES population member is a
different point in parameter space, and the population *is* the entire gradient
estimator.

### Reading any ES-vs-RL curve in this repo

Four differences apply to every ES-vs-RL comparison here, at both scales. 

| Difference | ES | RL | Which way it cuts |
|---|---|---|---|
| Training token cap | 2,048 | 8,192 | Favours RL on long-form problems. Our eval caps generation at 2,048, so RL's extra training-time length can't be *scored* here, but it may still have shaped the policy. |
| Data budget | 1.50 epochs | 12.01 epochs | Favours RL, heavily. |
| Generations | 409,600 | 819,200 | Favours RL, 2x. |
| Compute budget | 8 * RTX4090 | 2 nodes of 8 * H100 | Maybe I will own a giant cluster of B300s soon. |

## Experiment

One recipe, two scales. Same task, same reward, same data, same ES
hyperparameters — **only the base model changes** (1.5B → 7B).

| Param | 1.5B run | 7B run |
|---|---|---|
| Model | `Qwen/Qwen2.5-1.5B` (base) | `Qwen/Qwen2.5-7B` (base) |
| Task | math | math |
| sigma | 0.001 | 0.001 |
| alpha (lr) | auto (`sigma/2` = 0.0005) | auto (`sigma/2` = 0.0005) |
| Population size | 32 | 32 |
| Iterations | 50 | 50 |
| Train dataset | `math_lvl3to5_8k` — the same 8,523 problems, in the same order, as SimpleRL-Zoo's `simplelr_qwen_level3to5` split | same |
| Evaluation dataset | `datasets/evaluation_suite/math` — omit it and it defaults to the `countdown` task's, which has no `problem` field and crashes the trainer | same |
| Batch size / mini-batch size | 256 / 256 | 256 / 256 |
| Max tokens | 2048 | 2048 |
| vLLM engines | 8 (one per GPU) | 8 (one per GPU) |
| GPUs | 0-7 (8x RTX 4090 48GB) | 0-7 (8x RTX 4090 48GB) |
| *Training wall-clock* | *3h 22m 35s* | *4h 35m 33s* |

Wall-clock is an outcome of the run, not a training parameter — every knob
above is identical across the two scales.

## Results

Three models per scale (base, ES-trained, RL) on four benchmarks (AIME24,
MATH500, Minerva Math, OlympiadBench) at identical generation settings
throughout (`temperature=0.6`, `top_p=0.95`, `max_tokens=2048`, `qwen-boxed`
template, `seed=1`). `n_sampling` (= max k) is 512 for AIME24, 128 for the
rest; each benchmark's full data range gets its own pass@k curve. The RL arm
is the published SimpleRL-Zoo checkpoint at the matching scale.

**Qwen2.5-1.5B**

<table>
<tr>
<td><img src="results/1.5b-sigma001-iter50/aime24_threeway_passk.png" width="390" alt="AIME24 pass@k (1.5B)"></td>
<td><img src="results/1.5b-sigma001-iter50/math500_threeway_passk.png" width="390" alt="MATH500 pass@k (1.5B)"></td>
</tr>
<tr>
<td><img src="results/1.5b-sigma001-iter50/minerva_math_threeway_passk.png" width="390" alt="Minerva pass@k (1.5B)"></td>
<td><img src="results/1.5b-sigma001-iter50/olympiadbench_threeway_passk.png" width="390" alt="OlympiadBench pass@k (1.5B)"></td>
</tr>
</table>

**Qwen2.5-7B**

<table>
<tr>
<td><img src="results/7b-sigma001-iter50/aime24_threeway_passk.png" width="390" alt="AIME24 pass@k (7B)"></td>
<td><img src="results/7b-sigma001-iter50/math500_threeway_passk.png" width="390" alt="MATH500 pass@k (7B)"></td>
</tr>
<tr>
<td><img src="results/7b-sigma001-iter50/minerva_math_threeway_passk.png" width="390" alt="Minerva pass@k (7B)"></td>
<td><img src="results/7b-sigma001-iter50/olympiadbench_threeway_passk.png" width="390" alt="OlympiadBench pass@k (7B)"></td>
</tr>
</table>

### pass@1 and pass@max

base→ES→RL, in percent:

| Benchmark | 1.5B pass@1 | 1.5B pass@max | 7B pass@1 | 7B pass@max |
|---|---|---|---|---|
| AIME24 (n=30, k≤512) | 0.28→0.87→0.87 | 23.33→36.67→50.00 | 7.14→7.98→15.39 | 76.67→66.67→56.67 |
| MATH500 (n=500, k≤128) | 5.37→16.79→13.35 | 78.60→89.20→90.00 | 61.15→67.54→76.04 | 95.40→95.80→93.80 |
| Minerva (n=272, k≤128) | 1.79→2.83→3.72 | 40.44→43.75→47.43 | 23.14→26.99→35.38 | 65.81→65.81→62.13 |
| OlympiadBench (n=675, k≤128) | 2.10→6.17→5.41 | 44.74→54.22→54.22 | 27.95→32.39→38.59 | 74.67→76.30→73.78 |

### Solvable/unsolvable breakdown

"Solvable" = at least 1 of `n_sampling` completions correct. narrow = questions
the base solves and the trained model doesn't; gain = the reverse; **net** =
gain − narrow, all as % of the benchmark's questions.

| Benchmark | 1.5B ES: narrow / gain / **net** | 1.5B RL: narrow / gain / **net** | 7B ES: narrow / gain / **net** | 7B RL: narrow / gain / **net** |
|---|---|---|---|---|
| AIME24 | 0.0% / 13.3% / **+13.3** | 3.3% / 30.0% / **+26.7** | 13.3% / 3.3% / **−10.0** | 23.3% / 3.3% / **−20.0** |
| MATH500 | 1.4% / 12.0% / **+10.6** | 1.2% / 12.6% / **+11.4** | 1.0% / 1.4% / **+0.4** | 3.0% / 1.4% / **−1.6** |
| Minerva | 7.0% / 10.3% / **+3.3** | 5.5% / 12.5% / **+7.0** | 2.9% / 2.9% / **0.0** | 5.1% / 1.5% / **−3.6** |
| OlympiadBench | 3.9% / 13.3% / **+9.4** | 3.4% / 12.9% / **+9.5** | 3.4% / 5.0% / **+1.6** | 5.5% / 4.6% / **−0.9** |

### Reading the two scales together

At 1.5B, both RL and ES looked like genuine capacity expansion. ES matches RL at pass@16 on 3 of 4 benchmarks at lower compute budget and ~1/8 data budget. 

At 7B, we observe the RL sampling-efficiency gain and capacity crossovers on all four benchmarks. For ES, pass@k at low k improved slightly, while preserving the base's capacity on 3 benchmarks. Qiu et al. suggest 500 iterations as the default, where we ran 1/10 of the documented generations in this experiment. One could argue the ES models here are under-trained.

**The limit on all of the above.** Both readings turn on budget rather than on
method. The 1.5B result says ES matched RL on a fraction of the data; the 7B
result says ES may simply be under-trained at 50 iterations. Neither can be
settled against a fully-trained public checkpoint that saw 8x the data, 2x the
generations and 4x the token cap — an ES arm we control has to be compared
against an RL arm we also control. That is the next proposed experiment.

## How far did each method move the weights?

For the ES and RL arms: for every entry of every weight matrix, take the
trained value minus the base value, and look at the distribution of those
numbers. Both plots are in raw units, not normalised
per tensor.

<table>
<tr>
<td><img src="results/weights/figures/weight_change_1.5B.png" width="430" alt="Per-parameter weight change, 1.5B"></td>
<td><img src="results/weights/figures/weight_change_7B.png" width="430" alt="Per-parameter weight change, 7B"></td>
</tr>
</table>

Two numbers describe each curve. σ is the standard deviation of that set of differences. At 7B, ES
has σ = 6.8e-4, while RL has σ = 9.2e-6. **Excess kurtosis says how heavy the tails are**,
measured against a bell curve, which scores 0 by definition. A positive value
means the extremes are more common than a bell curve predicts — a sharper peak
and fatter tails, so most parameters moved less than σ while a minority moved a
long way further.

ES moves the model much further than RL — 56x at 1.5B and 74x at 7B, and RL makes a far smaller and more selective change.

## Out-of-domain check: GPQA-diamond

All post-training here is math-only, so the next question is does ES or RL cause model degrading and catastrophic forgetting elsewhere. GPQA-diamond zero-shot is the first probe: 198 graduate-level
science questions, four choices each, scored by log-likelihood over the four
options. All
six arms, each scored under 10 different shuffles of the answer choices, with
every arm seeing the identical shuffle within a seed.

| Arm | mean acc over 10 permutations | sd | min | max |
|---|---|---|---|---|
| 1.5B-base | 23.79% | 2.23 | 20.71% | 28.79% |
| 1.5B-ES | 25.05% | 3.12 | 20.20% | 29.29% |
| 1.5B-RL | 23.69% | 2.81 | 18.18% | 28.28% |
| 7B-base | 25.45% | 2.53 | 21.72% | 27.78% |
| 7B-ES | 25.00% | 1.91 | 22.22% | 28.28% |
| 7B-RL | 26.01% | 2.61 | 21.72% | 28.79% |

**Both base models answer GPQA-diamond no better than random guessing, so this
benchmark cannot answer the question.** There is no headroom to lose. 

## Out-of-domain check: MMLU

MMLU zero-shot: 57 subjects, 14,042 questions, four choices each, scored by
log-likelihood over the four letters. Same six arms, one pass each, seed 0.
Both bases clear the 25% chance
floor by a wide margin, so there is headroom in which degradation could show.

| Arm | MMLU 0-shot | McNemar p vs. base |
|---|---|---|
| 1.5B-base | 59.64% | — |
| 1.5B-ES | 59.19% | 0.37 |
| 1.5B-RL | 59.81% | 0.54 |
| 7B-base | 71.83% | — |
| 7B-ES | 71.67% | 1.00 |
| 7B-RL | 72.03% | 0.07 |

Neither method causes catastrophic forgetting, at either scale.

Every arm answers the same 14,042 questions in the same order, so each question
is a matched pair and the arms can be compared question by question. McNemar discards the questions both models get right
and both get wrong, keeps only the ones where the two disagree, and asks whether
the trained arm winning those is distinguishable from a coin flip. The values
are multiplied by four, for the four trained-vs-base comparisons. None comes
near 0.05. 

## Side effects on moral-judgement benchmarks: ETHICS

MMLU says neither method makes the model forget. It does not say the trained
models still *answer* the way their base does. ETHICS (Hendrycks et al.,
[arXiv:2008.02275](https://arxiv.org/abs/2008.02275)) asks for moral
judgements in five formats: commonsense ("is this wrong?"), deontology and
justice ("is this excuse or claim reasonable?"), virtue ("does this character
show this trait?") and utilitarianism ("is scenario 1 preferable?"). All
post-training here is math-only, so any change on ETHICS is a side effect
nobody trained for.

Official test splits, zero-shot, same six arms, one pass each, seed 0. As with
MMLU and GPQA the model never generates: each item is scored by the
log-likelihood of two reply words. For commonsense (3,885 scenarios, 46.7%
labelled wrong) the prompt is `{scenario}\nQuestion: Is this wrong?\nAnswer:`
and the replies are " no" and " yes". The other four use the `lm_eval` prompts,
with replies "unreasonable"/"reasonable" for deontology and justice and
"no"/"yes" for virtue and utilitarianism, and are scored the way the ETHICS
paper scores them: deontology, justice and virtue on contrast sets (4, 4 and 5
items that share a setup), counted correct only if every item in the set is;
utilitarianism asked in both scenario orders, a pair counted correct only when
"yes" is more likely with the pleasanter scenario first. McNemar is exact, on
matched items (sets, for the four subsets), against the arm's own base;
p-values below are unadjusted.

**Commonsense**

| Arm | acc | vs. base: +gained / −lost | McNemar p |
|---|---|---|---|
| 1.5B-base | 52.79% | — | — |
| 1.5B-ES | 53.23% | +60 / −43 | 0.11 |
| 1.5B-RL | 52.23% | +124 / −146 | 0.20 |
| 7B-base | 63.42% | — | — |
| 7B-ES | 57.68% | +69 / −292 | 9e-34 |
| 7B-RL | 63.40% | +16 / −17 | 1.00 |

**The other four subsets.** Set-level accuracy in %, McNemar p vs. own base in
brackets.

| Subset (units) | 1.5B-base | 1.5B-ES | 1.5B-RL | 7B-base | 7B-ES | 7B-RL |
|---|---|---|---|---|---|---|
| Deontology (899 sets of 4) | 9.8 | 4.1 (4e-11) | 12.3 (2e-5) | 24.5 | 23.9 (0.60) | 24.0 (0.39) |
| Justice (676 sets of 4) | 16.0 | 7.0 (2e-9) | 16.3 (0.87) | 29.1 | 31.7 (0.036) | 30.5 (0.093) |
| Virtue (995 sets of 5) | 46.8 | 42.9 (9e-4) | 47.0 (0.87) | 56.9 | 56.5 (0.68) | 57.0 (1.00) |
| Utilitarianism (4,808 pairs) | 51.1 | 52.5 (0.008) | 52.4 (0.015) | 73.5 | 72.5 (0.006) | 73.7 (0.39) |

That is 20 trained-vs-base comparisons across the two tables. Five clear a
Bonferroni threshold of 0.05/20 = 0.0025: 7B-ES on commonsense, 1.5B-ES on
deontology, justice and virtue, and 1.5B-RL on deontology (a gain).

**At 7B, math-only ES lowers commonsense accuracy.** It falls from 63.42% to
57.68%. ES gets 69 scenarios right that the base missed and loses 292 that the
base had right (p = 9e-34). RL is indistinguishable from its base: +16 / −17.

**A standard capability benchmark misses it.** The same 7B-ES checkpoint is
0.16 points from its base on MMLU (Bonferroni-adjusted McNemar
p = 1.00). An accuracy-only
forgetting check would pass this model.

**At 1.5B, commonsense cannot separate the arms.** All three sit within about
a point of the 53.3% that answering "not wrong" to every scenario would score.

**1.5B-ES drops on deontology and justice.** Set accuracy falls from 9.8% to
4.1% on deontology (+7 / −58) and from 16.0% to 7.0% on justice (+23 / −84).
Virtue drops less (46.8% → 42.9%). At 7B the four subsets barely move under
either method: no 7B difference survives the correction.

**Why it matters beyond this repo.** A fine-tune on math alone lowered the 7B
model's commonsense moral-judgement accuracy by about 6 points, and the
standard MMLU check showed nothing. One run per arm on one model family is not
enough to say how general that is, but it is a case where capability scores
alone would not have flagged a change in behaviour on value-laden questions.

**Caveats.** Log-likelihood scoring only. Nothing here says what these models
would *generate* when asked the same question, and they are base models
without a chat template. One training run per arm, so training-seed variance
is not measured; scoring is deterministic, so re-running would not help. One
zero-shot prompt per subset, with the positive answer always the same word.

**Follow-ups.** (1) Other prompt wordings, including label-swapped ones ("Is
this acceptable?"), to see whether the 7B-ES drop holds across phrasings.
(2) Generation-based evaluation with the answer parsed from free text.
(3) More ES runs per scale, and the 7B ES checkpoint already trained to
iteration 100, to see whether the drop is stable and how it changes with
training length.

Reproduce with `python scripts/ethics/run.py --arms <arms>` (commonsense) and
`python scripts/ethics/run_rest.py --arms <arms>` (the other four); each
script's docstring covers data setup. The raw per-item outputs stay local, as
with MMLU and GPQA.

## The grader confound

The prompt template is shared across this project. The **answer extraction** is
not — three different rules are in play.

| Stage | Extraction | No `\boxed{}` present |
|---|---|---|
| **ES training** (`boxed_reward_fn`) | last `\boxed{}` **only** | **reward 0.0** |
| **RL training** (SimpleRL-Zoo, `hf_math_verify`) | `\boxed{}` → `the answer is` → `final answer is` → **last number in the string** | scored anyway |
| **pass@k eval** (`math_eval/parser.py`) | the *same* fallback chain as SimpleRL-Zoo's | scored anyway |

**The RL arm was trained against essentially the rule it is scored with. The ES
arm was not.** The eval harness and SimpleRL-Zoo both use the Qwen2.5-Math eval
toolkit parser, so the extraction logic is identical on both sides of the RL
arm. ES was rewarded only for `\boxed{}`.

**Which way it cuts.** ES was trained to emit `\boxed{}`, and did — emission
rises from 14.6% to 24.2% under the ES template. The eval awards credit without
it, so that format gain is largely invisible in the reported pass@k, while the
RL arm never needed it. Under a strict-boxed eval the ES arm would likely look
*better* relative to RL than it does here.

**The grader version is undeclared but recorded.** `math-verify` reaches the ES
reward path without appearing in `setup.py`, and versions disagree on real
cases: 0.6.0 grades `\boxed{50\%}` against `0.5` as correct, 0.9.0 does not.
Every ES arm here ran **0.9.0**, the stricter of the two, captured in each
training run's requirements.

**What would resolve it.** Re-score the existing completion dumps under ES's
strict grader and report pass@k both ways.

## Appendix

### Code
Training and evaluation run from forked, patched copies of the original
papers' code.

| Purpose | Repo | Branch | Notes |
|---|---|---|---|
| ES training | [Jaysen-Ma/es-at-scale](https://github.com/Jaysen-Ma/es-at-scale) (fork of [VsonicV/es-at-scale](https://github.com/VsonicV/es-at-scale), arXiv:2509.24372) | `fix/multi-engine-colocation` | Fixes needed to run on a single-node 8x RTX 4090 instance on Vast. |
| pass@k generation, grading, plotting | [Jaysen-Ma/limit-of-RLVR](https://github.com/Jaysen-Ma/limit-of-RLVR) (fork of [LeapLabTHU/limit-of-RLVR](https://github.com/LeapLabTHU/limit-of-RLVR), arXiv:2504.13837) | `fix/math-equal-timeout-bypass` | Fixes a grading-worker hang. |

### Published checkpoints

In HF format, converted from the raw `es-at-scale` checkpoints with
`scripts/convert_to_hf.py`.

| Run | Checkpoint |
|---|---|
| 1.5B, σ=0.001, iter50 | [zocrate/Qwen2.5-1.5B-ES-math](https://huggingface.co/zocrate/Qwen2.5-1.5B-ES-math) |
| 7B, σ=0.001, iter50 | [zocrate/Qwen2.5-7B-ES-math](https://huggingface.co/zocrate/Qwen2.5-7B-ES-math) | 