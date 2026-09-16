---
name: dspy-book-eight-steps
description: >-
  The eight-step method that chapters 1 to 3 of Context Engineering with DSPy
  use to build every program — signatures, modules, a few explored examples, a
  dataset, metrics, a baseline, optimization, then test and iterate. Carries
  the chapter's distinctive move, optimizing the judge before the task so the
  metric is trustworthy before anything is measured against it, and the
  dual-mode metric that serves both plain evaluation and an optimizer.
when_to_use: >-
  User is starting a DSPy program and asks "where do I begin", "what order",
  "how do I know it improved"; a build has skipped straight to optimizing; a
  metric and a baseline are missing.
---

# The Eight Steps (book chapters 1 to 3)

From the O'Reilly companion repo (MIT). `dspy-fundamentals` owns the API.
This skill owns the **order**, which is where most builds go wrong.

| # | Step | Produces |
|---|---|---|
| 1 | Specify your signatures | the task and, if you need one, its judge |
| 2 | Build your modules | `Predict` or `ChainOfThought` over each signature |
| 3 | Explore a few examples | a hand-run loop, ending in `inspect_history` |
| 4 | Collect your dataset | `dspy.Example` objects with a seeded split |
| 5 | Define your metrics | one per thing being measured |
| 6 | Establish a baseline | a number to beat, before any compile |
| 7 | Optimize | the judge first, then the task |
| 8 | Test and iterate | best-of-N, and a run on a different provider |

Steps 3 and 6 are the ones people skip. Step 3 costs a few calls and tells you
whether the signature is even close. Step 6 is what makes step 7 falsifiable.

## The distinctive move: optimize the judge first

When the metric is itself a model, it is a program, and an unoptimized program
is a bad metric. So chapter 3 compiles the judge against labelled data, saves
it, reloads it, and only then optimizes the task against that judge.

The ordering matters because optimizing a task against an unvalidated judge
moves the program toward the judge's errors, and you cannot tell from the score
that it happened.

## The dual-mode metric

One function serves plain evaluation and an optimizer:

```python
def metric(example, response, trace=None, pred_name=None, pred_trace=None):
    score = 1 if example.is_ai == response.is_ai else 0
    if pred_name:
        return dspy.Prediction(score=score, feedback=example.notes)
    return score
```

When the optimizer is asking, it returns feedback. When `dspy.Evaluate` is
asking, it returns a number. Note where the feedback comes from: a `notes`
column written by a human during error analysis. The dataset carries the
diagnosis.

Write the five-argument form from the start. It costs nothing and keeps
reflective optimization available.

## Step 8 is not optional

Two checks the chapter runs and most builds skip:

- Wrap the optimized program in best-of-N selection and see whether sampling more helps. If it does, the program is under-determined.
- Re-run the evaluation with `set_lm` pointed at a **different provider**. A program that only works on the model it was optimized against is overfitted to that model, and you would rather learn that now.

One calling-convention trap: the reward function for best-of-N receives
`(kwargs_dict, pred)`, not the metric's argument list. They are different
contracts and the mismatch fails at runtime.

## A deterministic metric beats a judge when one exists

The quickstart notebook replaces the LLM judge with a table of roughly eighteen
compiled regex patterns for known tells, scoring zero on any hit and naming the
matched patterns in its feedback string. It costs nothing, never drifts, and
the feedback is more specific than a judge's.

Reach for a judge when the quality you want cannot be written as a pattern —
not before.

## Warnings the chapters do not give

Chapter 3 evaluates on the **full dataset including the training rows**, and
its sibling notebook optimizes against a slice of its own test set. The books's
numbers are illustrative, not clean. Keep your splits disjoint, and do not
inherit this.

Model identifiers throughout these notebooks are forward-dated placeholders
that do not resolve against any provider. Substitute real ones.

## Anti-patterns

- Optimizing before a baseline exists. There is nothing to compare to.
- Optimizing a task against a judge that was never checked against human labels.
- A three-argument metric that must be rewritten before GEPA can use it.
- Skipping step 3, then debugging an optimizer run when the signature was wrong.
- Evaluating on rows the optimizer trained on.
- Assuming the optimized program transfers to another model without testing it.

## Where to go next

- The API behind each step → `dspy-fundamentals`
- Step 4 in depth → `dspy-book-datasets`
- Step 5 in depth → `dspy-book-metrics`
- Steps 6 and 7 with measured outcomes → `dspy-book-optimizers`, `dspy-optimizer-selection`
- Full reference (each step's code, the judge loop) → [reference.md](reference.md)
- Runnable, dry-run capable → [example_eight_steps.py](example_eight_steps.py)
