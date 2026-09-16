---
name: dspy-optimizer-selection
description: >-
  Choose the smallest DSPy optimizer that fits the data, the budget and the
  artifact being tuned, instead of reaching for GEPA every time. Covers the
  whole family — LabeledFewShot, BootstrapFewShot, BootstrapFewShotWithRandomSearch,
  KNNFewShot, COPRO, MIPROv2, SIMBA, GEPA, BootstrapFinetune, Ensemble and
  BetterTogether — with the trainset size, metric shape and cost that each one
  actually needs, a baseline-first discipline, and the escalation order to
  follow when a cheaper optimizer plateaus.
when_to_use: >-
  User says "optimize this", "compile the program", "which optimizer", "GEPA or
  MIPRO", "few-shot", "fine-tune the prompts", "it is not getting better";
  before any compile run; when an optimizer is too slow or too expensive; when
  a scalar-only metric makes reflective optimization pointless; when prompt
  optimization has plateaued and weights are the next lever.
---

# DSPy Optimizer Selection (3.2.x)

Consolidated from the optimizer skills of `OmidZamani/dspy-skills` (MIT). The
rule this skill exists to enforce: **measure the uncompiled program first, then
pick the cheapest optimizer whose requirements you actually meet.** An
optimizer is not a quality setting. Each one needs a different trainset size
and a different metric shape, and the expensive ones buy nothing when the
cheap one has not been tried.

GEPA itself is not re-taught here — `dspy-gepa-optimizer` owns it. This skill
decides *whether* GEPA is the right call.

## Step 0, not optional: the baseline

```python
baseline = dspy.Evaluate(devset=devset, metric=metric, num_threads=8)(program)
```

Without that number, no optimizer result means anything. Metric contract and
devset construction: `dspy-evaluation-harness`.

## The selection matrix

Read top to bottom and stop at the first row you can satisfy.

| Your situation | Optimizer | Needs | Cost |
|---|---|---|---|
| A handful of labeled examples, want a floor | `dspy.LabeledFewShot` | any trainset, no metric | ~0 LM calls |
| ~10+ examples, no demos yet | `dspy.BootstrapFewShot` | metric returning a score; teacher LM | tens of calls |
| 50+ examples, demos help but you want the best set | `dspy.BootstrapFewShotWithRandomSearch` (alias `dspy.BootstrapRS`) | as above, plus budget for repeated search | hundreds |
| Examples vary a lot; each input wants its own demos | `dspy.KNNFewShot` | trainset plus a `dspy.Embedder` vectorizer | bootstrap cost plus embeddings |
| Instructions are the problem, demos are fine | `dspy.COPRO` | scalar metric | moderate |
| Want instructions **and** demos searched properly | `dspy.MIPROv2` | 100+ examples ideally; scalar metric; `dspy[optuna]` | high |
| Failures can be described in words, not just scored | `dspy.GEPA` → `dspy-gepa-optimizer` | 5-arg metric returning `dspy.Prediction(score, feedback)` | high, reflection LM |
| Want a reflective pass cheaper than GEPA | `dspy.SIMBA` | scalar metric, mini-batches | medium |
| Prompts have plateaued, a fine-tunable LM is available | `dspy.BootstrapFinetune` | fine-tunable LM, `set_lm()`, training infra | highest |
| Several good candidate programs already exist | `dspy.Ensemble` | the candidates | inference cost, every call |
| Want prompt **then** weight optimization in one run | `dspy.BetterTogether` | both of the above | highest |
| The artifact is text but not a DSPy program | GEPA over a custom adapter | a metric on that artifact | varies |

## Escalation order

```
baseline → LabeledFewShot → BootstrapFewShot → BootstrapRS
                                   ↓ (plateau)
                   scalar metric ──┴── rich feedback available
                          ↓                     ↓
                   MIPROv2 / SIMBA            GEPA
                          └──────── plateau ───┘
                                     ↓
                      BootstrapFinetune / BetterTogether
```

Escalate only on a measured plateau. Two optimizers in a row that fail to beat
the baseline usually mean the metric is wrong, not that the optimizer is weak.

## Constructor shapes worth remembering

Verified against the live wheel; full parameter lists in [reference.md](reference.md).

```python
dspy.BootstrapFewShot(metric=metric, max_bootstrapped_demos=4, max_labeled_demos=16)

dspy.MIPROv2(metric=metric, auto="light", num_threads=8, seed=9)   # auto: light|medium|heavy

dspy.SIMBA(metric=metric, bsize=32, num_candidates=6, max_steps=8) # keyword-only

dspy.KNNFewShot(k=3, trainset=trainset, vectorizer=dspy.Embedder("openai/text-embedding-3-small"))

dspy.BetterTogether(metric=metric,                 # named optimizers, any number
                    bootstrap=dspy.BootstrapFewShotWithRandomSearch(metric=metric),
                    gepa=dspy.GEPA(metric=metric, auto="light", reflection_lm=reflection_lm))

compiled = optimizer.compile(program, trainset=trainset)           # most optimizers
```

Three signatures break that last pattern, and getting them wrong is the usual
first error:

| Optimizer | Its actual `compile` call |
|---|---|
| `dspy.Ensemble` | `compile(programs)` — a list of programs, no student, no trainset |
| `dspy.KNNFewShot` | `compile(student, teacher=...)` — the trainset was passed to the constructor |
| `dspy.SIMBA` | `compile(student, trainset, seed=...)` — the seed lives here, not in the constructor |

`dspy.MIPROv2` validates its LMs **in the constructor**: with no
`dspy.configure(lm=...)` and no `prompt_model`/`task_model` pair it raises
`ValueError` before you ever call `compile`. `dspy.GEPA` does the same with
`reflection_lm`. Any dry-run or test path that constructs them therefore needs
a stub `dspy.LM(...)`, which costs nothing because constructing an LM makes no
network call.

`dspy.SIMBA` and `dspy.Ensemble` take keyword arguments only; a positional
metric raises `TypeError`. `dspy.LabeledFewShot` takes only `k` and no metric
at all. `dspy.BetterTogether` accepts arbitrary **named** optimizers and runs
them in the order given; there is no fixed prompt/weight argument pair.

## Metric shape decides half the matrix

| Metric returns | Optimizers that can use it |
|---|---|
| `float` / `bool` | everything except GEPA's reflection |
| `dspy.Prediction(score=..., feedback=...)` (5-arg signature) | GEPA, and everything else (the score is read) |
| a plain dict | nothing — `dspy.Evaluate` aggregates with `sum()` and raises `TypeError` |

Writing the 5-argument form from the start costs nothing and keeps GEPA open
as an option. See `dspy-evaluation-harness`.

## Budget discipline

- Split optimization cost from inference cost. `Ensemble` is cheap to build and
  expensive forever after; `MIPROv2` is the reverse.
- Set `num_threads` to what your rate limit tolerates, not to your core count.
- Pass a `seed` where the optimizer accepts one, and record it with the result.
- Save every candidate worth comparing: `compiled.save("cand-3.json")`.
- Track usage with `dspy.configure(track_usage=True)` during compile runs —
  an optimizer that silently costs ten times the alternative is a finding.

## Anti-patterns

- Compiling before evaluating. There is nothing to compare against.
- Picking GEPA for a scalar metric — the reflection LM has nothing to read, and you pay for it anyway.
- `MIPROv2` on 20 examples: the Bayesian search overfits the devset and reports a gain that does not survive a held-out set.
- Reusing the trainset as the devset, then believing the number.
- Escalating to `BootstrapFinetune` before checking that a prompt optimizer plateaued.
- Treating `auto="heavy"` as "better" — it is "more calls"; verify the gain.
- Chaining optimizers in `BetterTogether` that need metric shapes you do not have.

## Where to go next

- GEPA in depth (reflection LM, feedback metrics, budgets) → `dspy-gepa-optimizer`
- Metrics, devsets, `dspy.Evaluate` → `dspy-evaluation-harness`
- Saving, loading and shipping the compiled artifact → `dspy-production`
- The end-to-end build this fits into → `dspy-advanced-workflow`
- Full reference (every constructor, parameter meanings, trainset guidance) → [reference.md](reference.md)
- Runnable, dry-run capable → [example_optimizer_selection.py](example_optimizer_selection.py)
