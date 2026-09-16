# DSPy RLM Workflow — Reference

Source: `rlm-workflow` skill suite (rlm-distill, rlm-decompose, rlm-synthesize,
rlm-verify, rlm-workflow; after Zhang, Kraska, Khattab, *Recursive Language
Models*, arXiv:2512.24601) mapped onto the DSPy 3.3.1 surface
(`dspy.Refine`, `dspy.BestOfN`, `dspy.RLM`, `dspy.GEPA`).

## Pydantic models

```python
from pydantic import BaseModel, Field
from typing import Literal

class WorkflowPlan(BaseModel):
    complexity: Literal["constant", "linear", "quadratic"]
    depth: int = Field(ge=1, le=3)
    success_criteria: list[str]

class SubProblem(BaseModel):
    id: int
    description: str
    dependencies: list[int] = Field(default_factory=list)
    complexity: Literal["low", "medium", "high"] = "medium"
    success_criteria: str

class SubResult(BaseModel):
    id: int
    approach: str
    result: str
    confidence: float = Field(ge=0.0, le=1.0)

class Synthesis(BaseModel):
    agreements: list[str]
    contradictions: list[str]      # "topic — A says / B says — resolution — why"
    gaps: list[str]
    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
```

Complexity classes (from rlm-decompose): *constant* — one step, decomposition
buys nothing; *linear* — N independent sub-tasks; *quadratic* — N tasks with N
interactions. Decompose only linear and quadratic problems.

## `validate_dag`

```python
def validate_dag(sub_problems: list[SubProblem]) -> list[SubProblem]:
    """Return sub-problems in dependency order; raise ValueError on bad graphs."""
    by_id = {sp.id: sp for sp in sub_problems}
    if len(by_id) != len(sub_problems):
        raise ValueError("duplicate sub-problem ids")
    unknown = {d for sp in sub_problems for d in sp.dependencies if d not in by_id}
    if unknown:
        raise ValueError(f"dependencies on unknown ids: {sorted(unknown)}")
    order, state = [], {}                       # state: 1 = visiting, 2 = done
    def visit(sid: int, stack: tuple[int, ...]) -> None:
        if state.get(sid) == 2:
            return
        if state.get(sid) == 1:
            raise ValueError(f"cycle: {' -> '.join(map(str, stack + (sid,)))}")
        state[sid] = 1
        for dep in by_id[sid].dependencies:
            visit(dep, stack + (sid,))
        state[sid] = 2
        order.append(by_id[sid])
    for sp in sub_problems:
        visit(sp.id, ())
    return order
```

Recombination strategy follows from the graph: a chain → *sequential*; no
edges → *parallel* (run `Solve` calls with `dspy.Parallel` or threads); mixed →
*hierarchical* (synthesize groups first).

## Distillation

| Input size | Construct | Notes |
|---|---|---|
| ≤ ~100k tokens | `Distill` signature per chunk: `chunk, query -> relevance: Literal[0,1,2,3], excerpt` | chunk by heading or ~2k tokens; keep 3, summarize 1–2, drop 0 |
| > ~100k tokens | `dspy.RLM("context, query -> distilled", sub_lm=cheap, max_llm_calls=30)` | Deno required; see `dspy-rlm-module` |

Report `compression = len(distilled) / len(original)`; target 10–20 %. Record
what was excluded (the prose skill's "Summary of Excluded Content") as a list
output so the verifier can check that nothing named in `success_criteria` was
dropped.

Distillation patterns from rlm-distill still apply as *retriever strategies*
inside the RLM's tools: funnel filter (all → dirs → matches → sections), anchor
expansion (known file → imports → callers → tests), cross-reference (intersect
pattern hits).

## The three tiers, precisely

| Tier | Question | Implementation in the metric | Weight |
|---|---|---|---|
| 1 Syntactic | Is it structurally correct? | Pydantic validation of outputs, required fields present, format/lint/compile commands for code outputs, `contradictions` list well-formed | 0.3 |
| 2 Semantic | Does it mean what it should? | unit tests / gold comparison / every `success_criteria` item addressed (string or judge check), sub-results all referenced by the synthesis, contradictions listed when sub-results disagree | 0.4 |
| 3 Pragmatic | Does it work in the real world? | LM judge (cheaper model) with a written rubric: integrates, performance within stated bounds, maintainable, follows conventions | 0.3 |

Fast-fail: a Tier 1 failure returns `0.3 * t1` and a feedback string naming the
failed check; Tier 2 below 0.8 stops before the judge. The prose skill's
"Overall Confidence 0–100 %" is `score * 100`.

Feedback strings must say *which sub-problem or module* failed and *what good
looks like*; GEPA reads them with `pred_name` set to the predictor under
reflection, so per-module blame in the text is what makes credit assignment work.

## `dspy.Refine` / `dspy.BestOfN` (DSPy 3.3.1)

```python
dspy.Refine(module: dspy.Module, N: int,
            reward_fn: Callable[[dict, dspy.Prediction], float],
            threshold: float, fail_count: int | None = None)
dspy.BestOfN(module, N, reward_fn, threshold, fail_count=None)
```

- Both run `module` up to `N` times with distinct `rollout_id`s at `temperature=1.0`
  and return the first prediction with `reward >= threshold`, else the best seen.
- `Refine` additionally calls an internal `OfferFeedback` signature after a
  below-threshold attempt and injects per-module advice as a `hint_` input field
  on the next attempt. `BestOfN` samples without advice.
- `reward_fn(args, pred)` receives the call kwargs and the prediction and returns a
  float — wrap the verification cascade: `float(verify_cascade(gold_from(args), pred).score)`.
- `fail_count` (default `N`): how many attempts may raise before the error propagates.
- `Refine` needs `inspect.getsource(module.__class__)` and of `reward_fn` — define
  both in a file, not in a notebook cell or lambda.

## Budget knobs

| Knob | Where | Guidance |
|---|---|---|
| `max_depth` | `RLMWorkflow` | 1 for prototypes; 2 default; 3 only with `track_usage=True` |
| `N`, `threshold` | `dspy.Refine` | `N=3`, threshold = the score you would ship at (0.85–0.9) |
| `max_llm_calls` | `dspy.RLM` distiller | 20–50 |
| `auto` | `dspy.GEPA` | `"light"` first — the cascade is expensive per example |
| judge model | Tier 3 | cheaper than the task LM; deterministic tiers first |

## Failure modes

| Symptom | Cause | Fix |
|---|---|---|
| `ValueError: cycle` from `validate_dag` | LM produced circular dependencies | let it fail; Tier 1 feedback tells `decompose` to remove the cycle; GEPA learns it |
| synthesis ignores a sub-result | coverage not checked | Tier 2 `coverage` check: every `[id]` appears in `answer` or `gaps` |
| `Refine` never improves | reward is float-only, advice generic | make the cascade's feedback specific; Refine's `OfferFeedback` reads the trajectory, not your text, so also keep sub-results in the prediction |
| cost explodes at depth 3 | recursion × Refine × judge | cap `max_depth=2`, `N=2`, judge on a small model |
| Tier 3 always passes | rubric vague | write the rubric as pass/fail bullets with the stated bounds (latency, size) |

## Output template (for the agent's report)

Keep the prose skill's report shape so humans can read a run:

```markdown
## RLM Workflow (DSPy) — run report
**Problem** · **Complexity** · **Depth** · **Success criteria**
**Distillation**: original N tokens → M tokens (x %); excluded: …
**Decomposition** (strategy): table id | sub-problem | deps | status
**Sub-results**: per id — approach / result / confidence
**Synthesis**: agreements / contradictions (resolved how) / gaps
**Verification**: Tier 1 · Tier 2 · Tier 3 → score, blocking issues
**Iterations**: attempt | reward | advice applied
**Lessons learned**
```
