---
name: dspy-rlm-workflow
description: Solve context-heavy, multi-step problems as a DSPy 3.2.x program using the Recursive Language Model workflow — distill the context, decompose into a dependency-ordered set of sub-problems, solve each (recursing when needed), synthesize with an explicit consistency check, then verify through a three-tier cascade whose result doubles as the GEPA metric. Iterate at runtime with dspy.Refine and at compile time with GEPA. Use when the input is large, the task is multi-step, or quality must be verified before delivery.
when_to_use: >-
  User says "rlm workflow", "decompose this", "distill the context", "too much
  context", "multi-step", "verify before delivery"; the task spans many files,
  documents or sub-tasks; a single predictor keeps failing on a large input;
  or the user wants a verified, iterated result instead of a one-shot answer.
---

# DSPy RLM Workflow (3.2.x)

Port of the `rlm-workflow` skill suite (distill → decompose → solve → synthesize
→ verify → iterate, after the Recursive Language Models paper, arXiv:2512.24601)
into DSPy. The difference from the prose version: every phase is a typed
Signature or a deterministic function, the verification cascade **is** the
metric, and iteration is `dspy.Refine` at runtime and `dspy.GEPA` at compile time.
Nothing in the pipeline is a hand-written prompt.

## Phase → DSPy construct

| Phase | Prose skill | DSPy construct | Deterministic part |
|---|---|---|---|
| 1 Initialize | complexity + depth + success criteria | `Initialize` signature → `WorkflowPlan` | depth clamp, budget knobs |
| 2 Distill | pattern filter, relevance 0–3, keep 10–20 % | `dspy.RLM` over the raw context **or** `Distill` signature per chunk | compression ratio computed from token counts, never reported by the LM |
| 3 Decompose | sub-problems + dependency graph | `Decompose` signature → `list[SubProblem]` | DAG check + topological order (`validate_dag`) |
| 4 Solve | per sub-problem, may recurse | `Solve` signature, executed in topological order with dependency results injected; recursion when `complexity == "high"` and depth remains | scheduler |
| 5 Synthesize | agreements / contradictions / gaps | `Synthesize` signature → `Synthesis` (unified output + explicit conflicts + gaps + confidence) | every sub-result must be referenced (`coverage`) |
| 6 Verify | Tier 1 syntactic, 2 semantic, 3 pragmatic | `verify_cascade()` returning `dspy.Prediction(score, feedback)` | Tier 1 fully deterministic; Tier 2 tests/gold; Tier 3 LM judge with a rubric |
| 7 Iterate | re-decompose with feedback | `dspy.Refine(module, N, reward_fn, threshold)` at runtime; `dspy.GEPA(metric=verify_cascade)` at compile time | fail-fast, bounded N |

## Canonical program

```python
import dspy
from pydantic import BaseModel, Field
from typing import Literal

class SubProblem(BaseModel):
    id: int
    description: str
    dependencies: list[int] = Field(default_factory=list)
    complexity: Literal["low", "medium", "high"] = "medium"
    success_criteria: str

class Decompose(dspy.Signature):
    """Break the problem into independent, testable sub-problems with explicit
    dependencies. Prefer more small sub-problems over few large ones. Every
    dependency must name an existing sub-problem id; no cycles."""
    problem: str = dspy.InputField()
    distilled_context: str = dspy.InputField()
    strategy: Literal["by-domain", "by-dependency", "by-size"] = dspy.OutputField()
    sub_problems: list[SubProblem] = dspy.OutputField()

class Solve(dspy.Signature):
    """Solve one sub-problem using only the distilled context and the results of
    its dependencies. State the approach, the result, and a confidence in [0, 1]."""
    sub_problem: SubProblem = dspy.InputField()
    distilled_context: str = dspy.InputField()
    dependency_results: str = dspy.InputField()
    approach: str = dspy.OutputField()
    result: str = dspy.OutputField()
    confidence: float = dspy.OutputField()

class Synthesize(dspy.Signature):
    """Combine sub-results into one answer. List agreements, contradictions (with
    how each was resolved and why), and gaps. Never resolve a contradiction silently."""
    problem: str = dspy.InputField()
    sub_results: str = dspy.InputField()
    agreements: list[str] = dspy.OutputField()
    contradictions: list[str] = dspy.OutputField()
    gaps: list[str] = dspy.OutputField()
    answer: str = dspy.OutputField()
    confidence: float = dspy.OutputField()

class RLMWorkflow(dspy.Module):
    def __init__(self, max_depth: int = 2):
        super().__init__()
        self.decompose = dspy.ChainOfThought(Decompose)
        self.solve = dspy.ChainOfThought(Solve)
        self.synthesize = dspy.ChainOfThought(Synthesize)
        self.max_depth = max_depth

    def forward(self, problem: str, distilled_context: str, depth: int = 0) -> dspy.Prediction:
        plan = self.decompose(problem=problem, distilled_context=distilled_context)
        order = validate_dag(plan.sub_problems)          # raises on cycles / unknown ids
        results: dict[int, str] = {}
        for sp in order:
            deps = "\n".join(f"[{d}] {results[d]}" for d in sp.dependencies)
            if sp.complexity == "high" and depth < self.max_depth:
                sub = self.forward(sp.description, distilled_context, depth + 1)   # recurse
                results[sp.id] = sub.answer
                continue
            results[sp.id] = self.solve(sub_problem=sp, distilled_context=distilled_context,
                                        dependency_results=deps).result
        merged = self.synthesize(problem=problem,
                                 sub_results="\n".join(f"[{k}] {v}" for k, v in results.items()))
        return dspy.Prediction(plan=plan, sub_results=results, **merged)
```

`validate_dag` (see [reference.md](reference.md)) is plain Python: it rejects
unknown dependency ids and cycles and returns the topological order. Do not ask
the LM to order the work — it will get it wrong on the day it matters.

## Distill before you decompose

Two options, pick by size:

- **≤ ~100k tokens**: a `Distill` signature per heading-chunk with a relevance
  score `0–3`; keep 3s, summarize 1s, drop 0s. Report the compression ratio from
  `len()` of the kept text, not from the model.
- **> ~100k tokens**: `dspy.RLM("context, query -> distilled")` with a cheap
  `sub_lm` (see `dspy-rlm-module`). The RLM slices and greps the context itself.

Either way the distilled text is an *input* to `RLMWorkflow`, so the workflow is
testable with a hand-written distillation.

## The verification cascade is the metric

```python
def verify_cascade(gold, pred, trace=None, pred_name=None, pred_trace=None):
    t1 = tier1_syntactic(pred)               # parse/schema/format: deterministic, 0..1
    if t1 < 1.0:
        return dspy.Prediction(score=0.3 * t1, feedback=f"Tier 1 failed: {tier1_report(pred)}")
    t2 = tier2_semantic(gold, pred)          # tests / gold checks / cited evidence, 0..1
    if t2 < 0.8:
        return dspy.Prediction(score=0.3 + 0.4 * t2, feedback=f"Tier 2 failed: {tier2_report(gold, pred)}")
    t3 = tier3_pragmatic(gold, pred)         # LM judge with a written rubric, 0..1
    return dspy.Prediction(score=0.3 + 0.4 * t2 + 0.3 * t3,
                           feedback=tier3_report(gold, pred) or "Verified on all three tiers.")
```

Fast-fail ordering keeps cost down and makes the feedback specific — which is
exactly what GEPA needs. Tier 3's judge runs on a cheaper model than the task LM.

## Iterate: runtime and compile time

```python
def reward(args: dict, pred: dspy.Prediction) -> float:
    return float(verify_cascade(args_to_gold(args), pred).score)

refined = dspy.Refine(module=RLMWorkflow(), N=3, reward_fn=reward, threshold=0.9)
```

`dspy.Refine` re-runs the module with a new `rollout_id` at `temperature=1.0`
and, after a below-threshold attempt, asks an LM (`OfferFeedback`) for
per-module advice that is injected as a `hint_` input on the next try — the
prose skill's "iteration log with feedback applied", automated. Use
`dspy.BestOfN` when you only want sampling without advice.

Compile-time iteration is GEPA on the same metric:

```python
optimizer = dspy.GEPA(metric=verify_cascade, auto="light",
                      reflection_lm=dspy.LM("openai/gpt-5", temperature=1.0, max_tokens=32000))
optimized = optimizer.compile(student=RLMWorkflow(), trainset=trainset, valset=valset)
```

Because the cascade returns per-tier feedback, GEPA can assign blame to
`decompose`, `solve` or `synthesize` separately (`pred_name`).

## Workflow depth

| Depth | Use for | What changes |
|---|---|---|
| 1 | moderate problems | single decomposition, no recursion |
| 2 | complex problems | `high`-complexity sub-problems recurse once |
| 3+ | very complex | recursion until `max_depth`; `Refine` with `N=3` around the whole module |

Bound every recursion with `max_depth` and every retry with `N`; log
`track_usage=True` — a depth-3 run with recursion can issue dozens of calls.

## Anti-patterns

- Letting the LM emit the execution order or claim the compression ratio — compute both.
- A verifier that returns only a float — GEPA has nothing to learn from.
- Skipping Tier 1 because "the model is good" — a schema error at Tier 1 is free to catch, expensive at Tier 3.
- Resolving contradictions inside `Synthesize` without listing them — the `contradictions` field is required output, and Tier 2 checks it is non-empty when sub-results disagree.
- Recursing without a depth bound; `Refine` without `fail_count`.
- Re-decomposing on every iteration — `Refine` keeps the plan and re-runs the module with advice; only re-decompose when Tier 2 blames the decomposition.

## Where to go next

- Long-context distillation → `dspy-rlm-module`
- Metric details and the five-argument contract → `dspy-evaluation-harness`
- Optimizing the workflow → `dspy-gepa-optimizer`
- Refining the knowledge base the workflow retrieves from → `dspy-deep-refine`
- Full reference → [reference.md](reference.md)
- Runnable example → [example_rlm_workflow.py](example_rlm_workflow.py)
