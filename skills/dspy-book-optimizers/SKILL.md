---
name: dspy-book-optimizers
description: >-
  Measured optimizer results on one shared task, from chapter 6 of Context
  Engineering with DSPy — twelve optimizers on the same detector, same split,
  same metric, with real accuracy, cost and wall-time for each. Two optimizers
  score below the unoptimized baseline, the two free ones beat both, MIPROv2
  shows a ten-point validation-to-test overfit gap, and GEPA wins at
  twenty-six points for about sixty cents. Use it to set expectations before
  spending a compile budget.
when_to_use: >-
  User asks "which optimizer actually wins", "is GEPA worth it", "how much does
  optimizing cost", "how long does compiling take", "will optimizing help at
  all"; before committing to an optimizer; when an optimizer made results
  worse and that seems impossible.
---

# Optimizer Results, Measured (book chapter 6)

From the O'Reilly companion repo (MIT). `dspy-optimizer-selection` tells you
which optimizer to choose. This skill tells you what happened when someone ran
twelve of them on one task and wrote down the numbers.

**Read the caveat first.** The task is binary AI-text detection on 300 rows,
and the test split is *deliberately baseline-adversarial* — built from
predictions the unoptimized model got wrong. The authors say plainly it
teaches "optimizer tradeoffs, not a general AI-detector leaderboard." The
numbers below are real and reproducible; they are not your numbers.

## The table

Baseline: 53.75% on the locked 80-row test set. Split is pair-grouped,
160 train / 60 validation / 80 test, hash-verified on load.

| Optimizer | Test | Uplift | Cost | Time |
|---|---:|---:|---:|---:|
| GEPA | **80.00** | **+26.25** | $0.58 | 617s |
| KNNFewShot | 72.50 | +18.75 | **$0.00** | **0s** |
| BootstrapFinetune | 70.00 | +18.75 | $0.87 | 1026s |
| Ensemble | 70.00 | +16.25 | $0.89 | 1151s |
| LabeledFewShot | 67.50 | +13.75 | **$0.00** | **0s** |
| BootstrapFewShot | 67.50 | +13.75 | $0.003 | 4.5s |
| MIPROv2 | 66.25 | +12.50 | ≥$0.31 | 271s |
| BootstrapRS | 65.00 | +11.25 | $0.88 | 1119s |
| BetterTogether | 65.00 | +13.75 | $0.84 | 1740s |
| *(unoptimized)* | 53.75 | — | $0.00 | 0s |
| COPRO | 50.00 | **−3.75** | ≥$0.07 | 861s |
| SIMBA | **47.50** | **−6.25** | **$1.14** | 321s |

The weight optimizers ran against a different baseline (51.25%, a local Qwen
student), so their uplift is measured from there.

## Four things this table teaches

**Optimizing can make it worse.** COPRO lost 3.75 points and SIMBA lost 6.25 —
and SIMBA was the most expensive run in the table at $1.14. If you do not
measure a baseline and a held-out test set, you cannot detect this, and you
will ship a regression believing you shipped an improvement.

**The free optimizers are not the weak ones.** `LabeledFewShot` and
`KNNFewShot` cost nothing, take no measurable time, and beat six paid
optimizers. `BootstrapFewShot` cost a third of a cent and 4.5 seconds for the
same +13.75 as LabeledFewShot. Start here, always.

**Validation gain is not test gain.** MIPROv2 reached 76.67 on validation and
66.25 on test — a 10.4-point overfit gap. GEPA was the only optimizer where
validation and test agreed exactly, both 80.00. Report the held-out number.

**Cost does not predict quality.** Ranking by spend gives SIMBA ($1.14, worst
result) above GEPA ($0.58, best result). Budget buys search breadth, not
insight — as the notebooks put it, extra compile spend "buys candidate
selection, not a new instruction."

## Invocations worth copying

```python
dspy.LabeledFewShot(k=4).compile(detector, trainset=trainset, sample=True)

dspy.BootstrapFewShot(metric=exact_match, max_bootstrapped_demos=2,
                      max_labeled_demos=2, max_rounds=1, max_errors=1
                      ).compile(detector, trainset=trainset)

dspy.KNNFewShot(k=4, trainset=trainset, vectorizer=vectorizer, metric=exact_match,
                max_bootstrapped_demos=0, max_labeled_demos=4).compile(detector)

dspy.GEPA(metric=feedback_metric, auto='light', reflection_minibatch_size=3,
          reflection_lm=reflection_lm, num_threads=4,
          candidate_selection_strategy='pareto', use_merge=False,
          track_best_outputs=True, track_stats=True, seed=42
          ).compile(detector, trainset=trainset, valset=valset)
```

Two plumbing traps the notebooks call out:

- **Do not pass `num_threads` to `KNNFewShot`.** DSPy forwards unknown constructor arguments to `BootstrapFewShot`, which does not accept it.
- **`KNNFewShot` takes `trainset` and `vectorizer` in the constructor**, and only the student in `compile`.

The KNN run used a local hashed n-gram embedder, which is why its cost is
genuinely zero rather than merely small.

## Why GEPA won here

Its metric returned a diagnosis, not just a score, and the instruction it was
given told the model *what to judge*: writing style rather than topic or
technical sophistication. A companion experiment measured the same run
properly — uplift +26.25 points, McNemar p = 0.000104, bootstrap 95% confidence
interval [16.25, 36.25] points, 684 metric calls. That is the chapter's one
statistically defended result, and its stated conclusion includes the limit:
*"This is a pedagogical optimizer stress test, not evidence that AI-text
detection works reliably in the wild."*

GEPA earns its cost when failures can be described in words. With a scalar-only
metric it has nothing to reflect on and you pay the reflection model anyway.

## Ensemble is not an optimizer

`dspy.Ensemble.compile()` takes a **list of programs**, no trainset and no
metric. It combines candidates you already built. Its 70.00 came with an
evaluation cost 3.7 times the baseline and mean latency of 4.8 seconds against
1.8 — because every component runs on every example, forever. That is an
inference-time tax, not a one-off compile cost.

## Anti-patterns

- Quoting these numbers as general optimizer rankings. The test split was built to be adversarial to the baseline.
- Skipping the baseline, then being unable to notice a COPRO- or SIMBA-shaped regression.
- Reporting the validation score. MIPROv2 looks 10 points better there than it is.
- Reaching for an expensive optimizer before running the two free ones.
- Using GEPA with a scalar metric.
- Treating `Ensemble`'s accuracy gain as free; it multiplies every future inference.
- Copying the notebooks' model identifiers, which are book-internal aliases that do not resolve.

## Where to go next

- Choosing an optimizer for your task shape → `dspy-optimizer-selection`
- GEPA's reflection model, budgets and feedback metrics → `dspy-gepa-optimizer`
- Writing the metric first → `dspy-book-metrics`, `dspy-evaluation-harness`
- Building the trainset these all consume → `dspy-book-datasets`
- Full reference (every configuration, the experiment's statistics) → [reference.md](reference.md)
- Runnable, dry-run capable → [example_optimizer_results.py](example_optimizer_results.py)
