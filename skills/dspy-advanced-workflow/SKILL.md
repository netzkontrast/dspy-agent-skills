---
name: dspy-advanced-workflow
description: Build DSPy 3.2.x programs through spec, program, metric and baseline; extend to optimization and export when requested and justified by task budget. Orchestrates the other nine DSPy skills (dspy-fundamentals, dspy-evaluation-harness, dspy-gepa-optimizer, dspy-rlm-module, dspy-rlm-workflow, dspy-deep-refine, dspy-reflect-loop, dspy-clarify, dspy-tetraframe) in the correct order. Use for greenfield DSPy builds; prototypes may stop at a validated baseline.
when_to_use: User wants to build, optimize, and ship a new DSPy pipeline; says "full workflow" / "end to end" / "from scratch"; or needs the standard loop applied to a greenfield task.
---

# DSPy Advanced Workflow (2026)

This skill runs the seven-step loop that turns a natural-language task description into an optimized, saved, deployable DSPy program. Use the relevant steps in order. Stop at a validated baseline for a prototype; optimizer runs require an appropriate authorized budget and evidence of need. Exporting a local artifact does not authorize deployment.

## The seven steps

### 1. Spec

Rephrase the user's task in one sentence. Identify inputs, outputs, the quality axis that matters, and any constraints (latency, cost, tool access, context size). Pick predictor shape:

| Task shape | Predictor |
|---|---|
| Single-step structured I/O | `dspy.Predict` / `dspy.ChainOfThought` |
| Tool use / multi-step | `dspy.ReAct` |
| Code execution | `dspy.ProgramOfThought` |
| Long context / codebase | `dspy.RLM` → `dspy-rlm-module` |
| Context-heavy, multi-step, must be verified | decompose/solve/synthesize/verify → `dspy-rlm-workflow` |
| Retrieval base keeps failing the question | refine the base → `dspy-deep-refine` |
| Users keep correcting the program | corrections → gold + feedback → `dspy-reflect-loop` |
| Content moves into an authoritative store, or a statement is vague | clarify gate → `dspy-clarify` |
| A contested or hard-to-reverse decision (merge/supersede a page, resolve a conflict, pick a design) | four-corner assessment → `dspy-tetraframe` |

### 2. Program

Write the typed `dspy.Signature` + `dspy.Module` subclass per `dspy-fundamentals`. No hard-coded prompts. Keep predictors named so GEPA can target them.

### 3. Data

Build `trainset` and **separate** `valset` as `dspy.Example(...).with_inputs(...)`. For GEPA, maximize trainset size and keep validation just large enough to represent downstream behavior; held-out `testset` is reported on at the end only. See `dspy-evaluation-harness`.

### 4. Rich metric

Write `rich_metric(gold, pred, trace=None, pred_name=None, pred_trace=None)` returning `dspy.Prediction(score=0..1, feedback="natural-language critique")`. The feedback is load-bearing — it's what GEPA's reflection LM learns from. A dict with the same fields crashes `dspy.Evaluate`; only `dspy.Prediction` aggregates correctly. See `dspy-evaluation-harness`.

### 5. Baseline

```python
evaluator = dspy.Evaluate(devset=valset, metric=rich_metric,
                          num_threads=8, display_progress=True,
                          provide_traceback=True,
                          save_as_json="runs/baseline.json")
baseline = evaluator(program)
print("Baseline:", baseline.score)
```

### 6. GEPA optimize

```python
reflection_lm = dspy.LM("openai/gpt-5", temperature=1.0, max_tokens=32000)
optimizer = dspy.GEPA(
    metric=rich_metric,
    auto="medium",
    reflection_lm=reflection_lm,
    candidate_selection_strategy="pareto",
    track_stats=True,
    track_best_outputs=True,
    log_dir="./gepa_logs",
    num_threads=8,
    seed=0,
)
optimized = optimizer.compile(student=program, trainset=trainset, valset=valset)
print("Optimized:", evaluator(optimized).score)
```

Run `auto="light"` first as a sanity check; move to `auto="medium"`/`"heavy"` for the final run. See `dspy-gepa-optimizer`.

If you need a deliberate multi-stage compile loop, DSPy 3.2.x also exposes `dspy.BetterTogether(metric=..., bootstrap=..., gepa=...)` for chaining named optimizers after you have a clean baseline GEPA setup.

### 7. Export & deploy

```python
optimized.save("artifacts/program.json", save_program=False)     # state, portable
# or for full deployment artifact:
optimized.save("artifacts/program_dir/", save_program=True)
```

Deploy:
- Load with `dspy.load("artifacts/program_dir/")` or reconstruct + `.load("program.json")`.
- Wrap in FastAPI/CLI.
- Enable `track_usage=True` for cost/latency observability.
- Log with MLflow (`mlflow.dspy.autolog()`) or W&B in CI.
- Keep an offline regression test that runs the `evaluator` against the saved program and fails CI below a threshold.

## Full orchestration template

```python
"""DSPy end-to-end pipeline — spec → optimize → deploy."""

import dspy
from pathlib import Path

# ----- 1–2. Spec & program (dspy-fundamentals) -----
class MyTask(dspy.Signature):
    """<one-line instruction from the spec>."""
    input_field: str = dspy.InputField()
    output_field: str = dspy.OutputField()

class MyProgram(dspy.Module):
    def __init__(self):
        super().__init__()
        self.step = dspy.ChainOfThought(MyTask)
    def forward(self, **kw):
        return self.step(**kw)

# ----- 3. Data (dspy-evaluation-harness) -----
trainset = [...]   # list[dspy.Example(...).with_inputs(...)]
valset   = [...]

# ----- 4. Rich metric (dspy-evaluation-harness) -----
def rich_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    score = ...          # compute 0..1
    feedback = ...       # detailed critique
    return dspy.Prediction(score=score, feedback=feedback)  # NOT a dict

# ----- 5. Baseline -----
dspy.configure(lm=dspy.LM("openai/gpt-4o"), track_usage=True)
evaluator = dspy.Evaluate(devset=valset, metric=rich_metric, num_threads=8,
                          display_progress=True, provide_traceback=True,
                          save_as_json="runs/baseline.json")
program = MyProgram()
print("Baseline:", evaluator(program).score)

# ----- 6. GEPA optimize (dspy-gepa-optimizer) -----
optimizer = dspy.GEPA(
    metric=rich_metric,
    auto="medium",
    reflection_lm=dspy.LM("openai/gpt-5", temperature=1.0, max_tokens=32000),
    candidate_selection_strategy="pareto",
    track_stats=True, track_best_outputs=True,
    log_dir="./gepa_logs", num_threads=8, seed=0,
)
optimized = optimizer.compile(student=program, trainset=trainset, valset=valset)
print("Optimized:", evaluator(optimized).score)

# ----- 7. Export (dspy-fundamentals) -----
Path("artifacts").mkdir(exist_ok=True)
optimized.save("artifacts/program.json", save_program=False)
```

## Guardrails

- Define the metric in step 4 before optimization; use informative feedback appropriate to the task.
- Always baseline before optimizing — no baseline, no claim.
- Save both pre- and post-optimization metrics to JSON for auditability.
- If held-out test score drops, preserve that result and diagnose using training/validation evidence rather than assuming a cause. After test-informed changes, use a new untouched final holdout or label subsequent results exploratory; do not repeatedly tune against the original test set.
- Freeze optimized program with `module._compiled = True` before multi-stage re-compilation.

## The self-optimizing loop (beyond one GEPA run)

Once a program has a metric, three skills keep improving it and what it works with:

| Loop | Skill | What improves | Signal |
|---|---|---|---|
| runtime iteration | `dspy-rlm-workflow` (`dspy.Refine` around the module) | this call's output | the verification cascade |
| compile-time optimization | `dspy-gepa-optimizer` | the program's instructions/demos | the metric's feedback |
| knowledge-base refinement | `dspy-deep-refine` | the base the program retrieves from | unanswerable queries → reviewed edits |
| human feedback | `dspy-reflect-loop` | trainset + metric feedback | corrections/approvals from sessions |
| precision gate | `dspy-clarify` | what a claim, task or query actually asserts | explicit scope, bound terms, open questions |
| decision assessment | `dspy-tetraframe` | the frame a decision is made in | four isolated corners, contradiction map, verified P* |

Order per cycle: reflect (new gold from corrections) → GEPA (re-optimize) → deep-refine (fix the base for queries that still fail) → rlm-workflow (verified execution). Clarify runs at every boundary where content changes authority: before decomposition, before refining the base for a query, before promotion. TetraFrame runs before a contested decision is recorded — a deep-refine proposal that merges, supersedes or deletes, a promotion that contradicts the store, a design choice with two camps — and its run is cited by the decision. Every loop is dry-run-first and keeps a human approval on writes.

## Runnable scaffold → [example_pipeline.py](example_pipeline.py)
