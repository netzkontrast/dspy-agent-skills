# DSPy Optimizer Selection — Reference

Source: the optimizer skills of `OmidZamani/dspy-skills` (MIT) —
`dspy-optimizer-selection`, `dspy-bootstrap-fewshot`, `dspy-miprov2-optimizer`,
`dspy-simba-optimizer`, `dspy-finetune-bootstrap`, `dspy-better-together` —
consolidated into one routing skill. Signatures below were read off the
installed wheel with `inspect.signature`, not copied from prose docs.

## Verified constructor signatures

```python
dspy.MIPROv2(metric, prompt_model=None, task_model=None, teacher_settings=None,
             max_bootstrapped_demos=4, max_labeled_demos=4,
             auto="light",                      # "light" | "medium" | "heavy" | None
             num_candidates=None, num_threads=None, max_errors=None,
             seed=9, init_temperature=..., ...)

dspy.SIMBA(*, metric, bsize=32, num_candidates=6, max_steps=8, max_demos=4,
           prompt_model=None, teacher_settings=None,
           demo_input_field_maxlen=100_000, num_threads=None,
           temperature_for_sampling=0.2, temperature_for_candidates=0.2)

dspy.BootstrapFinetune(metric=None, multitask=True, train_kwargs=None,
                       adapter=None, exclude_demos=False, num_threads=None)

dspy.BetterTogether(metric, **optimizers)      # named optimizers, run in order

dspy.Ensemble(*, reduce_fn=None, size=None, deterministic=False)

dspy.KNNFewShot(k, trainset, vectorizer, **few_shot_bootstrap_args)
```

```python
dspy.LabeledFewShot(k=16)                      # no metric parameter at all

dspy.BootstrapFewShot(metric=None, metric_threshold=None, teacher_settings=None,
                      max_bootstrapped_demos=4, max_labeled_demos=16,
                      max_rounds=1, max_errors=None)

dspy.COPRO(prompt_model=None, metric=None, breadth=10, depth=3,
           init_temperature=1.4, track_stats=False)
```

## `compile` is not uniform

Most optimizers are `compile(student, trainset=...)`, several accept
`teacher=` and `valset=`, and three differ enough to trip you:

| Optimizer | `compile` parameters |
|---|---|
| `MIPROv2` | `student, trainset, teacher, valset, num_trials, ...` |
| `GEPA` | `student, trainset, teacher, valset` |
| `BetterTogether` | `student, trainset, teacher, valset, num_threads` |
| `BootstrapFewShot` | `student, teacher, trainset` |
| `BootstrapFinetune` | `student, trainset, teacher` |
| `COPRO` | `student, trainset, eval_kwargs` |
| `LabeledFewShot` | `student, trainset, sample` |
| `SIMBA` | `student, trainset, seed` — the seed is here, not in the constructor |
| `KNNFewShot` | `student, teacher` — the trainset went to the constructor |
| `Ensemble` | `programs` — a list; there is no student and no trainset |

`dspy.SIMBA` and `dspy.Ensemble` are keyword-only constructors. `dspy.Ensemble`
takes no metric: it combines programs you already have.

## Choosing by trainset size

| Examples | Realistic options |
|---|---|
| < 10 | `LabeledFewShot`, or fix the data first |
| 10–50 | `BootstrapFewShot`, `SIMBA` |
| 50–100 | `BootstrapFewShotWithRandomSearch`, `SIMBA`, `GEPA` |
| 100–500 | `MIPROv2`, `GEPA`, `BootstrapRS` |
| 500+ | `MIPROv2`, `BootstrapFinetune`, `BetterTogether` |

GEPA is the exception to "more data is better": it learns from the *feedback
text* of a few dozen informative failures, so a train-heavy split of 20–50
well-chosen examples often beats 500 bland ones. Details in
`dspy-gepa-optimizer`.

## Parameter meanings that actually matter

| Parameter | Where | What it controls |
|---|---|---|
| `max_bootstrapped_demos` | Bootstrap*, MIPROv2 | how many teacher-generated demos land in the prompt; raises token cost per call forever |
| `max_labeled_demos` | Bootstrap*, MIPROv2 | how many raw trainset examples may be used directly |
| `auto` | MIPROv2, GEPA | preset budget. `light` first, always |
| `num_candidates` | MIPROv2, SIMBA | breadth of the search |
| `bsize` | SIMBA | mini-batch size per step; the introspection happens per batch |
| `max_steps` | SIMBA | optimization steps; the main cost lever |
| `multitask` | BootstrapFinetune | one model for all predictors, or one per predictor |
| `train_kwargs` | BootstrapFinetune | passed to the provider's fine-tuning job; per-LM dict is allowed |
| `reduce_fn` | Ensemble | how candidate outputs are combined, e.g. `dspy.majority` |
| `size` | Ensemble | how many candidates to sample per call |
| `seed` | MIPROv2 | reproducibility; record it with the result |

## Teacher and prompt models

`BootstrapFewShot` and friends generate demos with a *teacher*. By default the
teacher is the student's own LM, which limits the ceiling. A stronger teacher
is set through `teacher_settings`:

```python
optimizer = dspy.BootstrapFewShot(
    metric=metric,
    teacher_settings=dict(lm=dspy.LM("openai/gpt-5")),
)
```

`MIPROv2` separates `prompt_model` (writes candidate instructions) from
`task_model` (runs the program). Give the prompt model the stronger LM; it is
called far less often.

## Fine-tuning path

`dspy.BootstrapFinetune` distills prompt-level behaviour into weights:

1. Run a prompt optimizer first and confirm it plateaued.
2. Point the student at a fine-tunable LM (`program.set_lm(...)`).
3. `optimizer.compile(program, trainset=trainset)` bootstraps traces and submits the job.
4. Evaluate the fine-tuned program against the *prompt-optimized* baseline, not the raw one.

The win is usually latency and cost per call, not accuracy. If accuracy is the
goal and the prompt optimizer plateaued, suspect the metric or the data.

## BetterTogether

```python
optimizer = dspy.BetterTogether(
    metric=metric,
    bootstrap=dspy.BootstrapFewShotWithRandomSearch(metric=metric),
    gepa=dspy.GEPA(metric=metric, auto="light", reflection_lm=reflection_lm),
)
compiled = optimizer.compile(program, trainset=trainset)
```

The keyword *names* are yours; the order is the run order. Any number of stages
is allowed. This replaced an older two-slot API — code or docs that pass a
prompt-optimizer and weight-optimizer pair are targeting a version before
3.2.0 and will not run.

## Reporting a compile run

Record all of it, or the run is not reproducible:

| Field | Why |
|---|---|
| optimizer class and every non-default argument | the run cannot be repeated without it |
| DSPy version | optimizer internals change between minors |
| trainset / devset sizes and the split rule | guards against train–dev leakage |
| baseline score and compiled score, same devset | the only number that means anything |
| seed | reproducibility |
| LM usage during compile, and per call after | separates one-off from recurring cost |

## Deliberately not ported here

`dspy-gepa-reflective` and `dspy-better-together` from the source pack overlap
`dspy-gepa-optimizer`, which already teaches GEPA's reflection LM, feedback
metrics, budget presets and `BetterTogether` chaining in more depth. Porting
them would have created two sources of truth for the same API.
