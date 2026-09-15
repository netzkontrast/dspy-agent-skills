---
name: dspy-autodialectics
description: >-
  Wrap a DSPy program in the Autodialectics anti-slop harness — compile the
  task into an immutable contract (sha256, domain defaults, forbidden
  shortcuts), force a typed thesis → antithesis → synthesis plan with an
  objection ledger, keep execution and verification in separate predictors,
  score the output on 12 deterministic slop dimensions that return
  dspy.Prediction(score, feedback) so the same function is the GEPA metric,
  gate accept/revise/reject on fixed thresholds, and promote a GEPA challenger
  only when score rises, slop does not, and canary cases pass. Tiered, so the
  gate runs with zero LM calls.
when_to_use: >-
  User says "anti-slop", "keep it honest", "dialectic", "thesis antithesis",
  "objections", "did it really finish", "fake completion", "requirement drift",
  "champion challenger", "canary"; an agent or program drifts from the task,
  claims done without evidence, or self-certifies; when the verification step
  of a pipeline must be independent of the executor; when a GEPA metric needs
  a reusable slop penalty.
---

# DSPy Autodialectics (3.2.x)

Port of `autodialectics` (an agentic harness: immutable contracts, evidence,
dialectical planning, domain execution, independent verification, slop
scoring, gate, champion/challenger evolution) into DSPy constructs. What is
ported is the **control loop**, not the product: the CLI, REST API, MCP server,
CLI gateways, SQLite store and code sandbox stay in the original. The shift:
every stage that was a prompt string plus a regex parser becomes a typed
Signature, and every stage that was a heuristic stays a **deterministic
function that returns `dspy.Prediction(score, feedback)`** — so the slop
scorer is both the runtime gate and the GEPA metric.

## Load only the tier you need

| Tier | What | LM calls | Load |
|---|---|---|---|
| 0 | `compile_contract` → `slop_score` → `gate` on an output you already have | none | this file + the example's functions |
| 1 | `Dialectic` (thesis/antithesis/synthesis) + `Verify` around your executor | 4 | the canonical program below |
| 2 | champion/challenger: GEPA on the planner with `slop_score` folded into the metric, canary gate on promotion | GEPA budget | [reference.md](reference.md) §Evolution |

Tier 0 is free and already catches fake completion, requirement drift and
ignored objections. Do not load Tier 2 for a single run.

## Autodialectics stage → DSPy construct

| Stage (original) | DSPy construct | Deterministic |
|---|---|---|
| `ContractCompiler` (domain keywords, defaults, sha256) | `compile_contract(task) -> Contract` (pydantic) | yes |
| `ContextExplorer` (lexical / RLM) | `dspy.RLM` above ~8k chars → `dspy-rlm-module`; below, pass assets inline | — |
| `DialecticalPlanner` thesis / antithesis / synthesis | three Signatures; `Objection` and `Disposition` are typed outputs — no regex | — |
| domain `ExecutionAdapter` | your program (any `dspy.Module`); the contract's `domain` selects it | — |
| `RunEvaluator.verify()` (fresh context, keyword overlap) | `Verify` signature that sees **only** contract + output, plus deterministic per-criterion overlap | mixed |
| `SlopScorer` (12 metrics, weighted composite) | `slop_score(...) -> dspy.Prediction` | yes |
| `AdvanceGate` accept / revise / reject | `gate(verdict, confidence, score, slop)` | yes |
| `ChampionChallengerManager` + GEPA | `dspy.GEPA` on `Dialectic.thesis`; `promote()` rule; canary `dspy.Example`s | yes (rule) |

## Canonical program

```python
import dspy
from pydantic import BaseModel, Field
from typing import Literal

Domain = Literal["code", "research", "writing", "experiment", "analysis", "generic"]

class Contract(BaseModel):                       # immutable once compiled
    source_hash: str                             # sha256 of the normalized task
    title: str
    domain: Domain
    objectives: list[str]
    constraints: list[str] = Field(default_factory=list)
    acceptance_criteria: list[str]
    forbidden_shortcuts: list[str]

class Objection(BaseModel):
    claim: str                                   # the step being challenged
    objection: str
    severity: float = Field(ge=0.0, le=1.0)

class Disposition(BaseModel):
    objection_index: int
    accepted: bool
    how: str                                     # how the plan changed, or why rejected

class Thesis(dspy.Signature):
    """Plan the task as numbered, concrete steps that satisfy every objective under
    every constraint. Include verification steps; never plan around a forbidden shortcut."""
    contract: Contract = dspy.InputField()
    evidence_summary: str = dspy.InputField()
    steps: list[str] = dspy.OutputField()

class Antithesis(dspy.Signature):
    """Find the real flaws, risks and gaps in the plan. Be adversarial but concrete;
    do not invent problems. Severity: 1.0 = the plan fails the contract, 0.3 = nit."""
    contract: Contract = dspy.InputField()
    steps: list[str] = dspy.InputField()
    evidence_summary: str = dspy.InputField()
    objections: list[Objection] = dspy.OutputField()

class Synthesis(dspy.Signature):
    """Revise the plan. For EVERY objection state accepted or rejected and how the
    revised steps reflect it. Unaddressed objections with severity > 0.8 are failures."""
    contract: Contract = dspy.InputField()
    steps: list[str] = dspy.InputField()
    objections: list[Objection] = dspy.InputField()
    dispositions: list[Disposition] = dspy.OutputField()
    revised_steps: list[str] = dspy.OutputField()
    assumptions: list[str] = dspy.OutputField()

class Verify(dspy.Signature):
    """Independent verification. You see only the contract and the output — not the
    plan. For each acceptance criterion decide pass/fail with a one-line reason."""
    contract: Contract = dspy.InputField()
    output: str = dspy.InputField()
    checks: list[tuple[str, bool, str]] = dspy.OutputField(desc="(criterion, passed, reason)")
    independent_findings: list[str] = dspy.OutputField()

class Dialectic(dspy.Module):
    def __init__(self, executor: dspy.Module):
        super().__init__()
        self.thesis, self.antithesis = dspy.ChainOfThought(Thesis), dspy.ChainOfThought(Antithesis)
        self.synthesis, self.verify = dspy.ChainOfThought(Synthesis), dspy.Predict(Verify)
        self.executor = executor

    def forward(self, contract: Contract, evidence_summary: str = "") -> dspy.Prediction:
        t = self.thesis(contract=contract, evidence_summary=evidence_summary)
        a = self.antithesis(contract=contract, steps=t.steps, evidence_summary=evidence_summary)
        s = self.synthesis(contract=contract, steps=t.steps, objections=a.objections)
        out = self.executor(contract=contract, plan=s.revised_steps)      # your program
        v = self.verify(contract=contract, output=out.output)             # never sees the plan
        return dspy.Prediction(plan=s.revised_steps, objections=a.objections,
                               dispositions=s.dispositions, output=out.output,
                               declared_uncertainties=getattr(out, "uncertainties", []),
                               checks=v.checks, findings=v.independent_findings)
```

## The slop metric and the gate (deterministic)

`slop_score(contract, output, declared_uncertainties, objections, dispositions,
evidence_excerpts)` computes the twelve dimensions of the original with its
weights (unsupported claims 0.15, fake completion 0.15, verbosity 0.12,
repetition 0.10, requirement drift 0.10, self-verification bias 0.08, five
more at 0.05) and returns
`dspy.Prediction(score=1 - composite, feedback="<worst dimensions, named>")`.
The feedback is load-bearing: GEPA's reflection LM reads "fake completion:
'done' claimed with no test results or files" and rewrites the instruction.

```python
def gate(verdict_pass: bool, confidence: float, score: float, slop: float) -> str:
    if not verdict_pass and confidence < 0.3: return "reject"     # verification failed badly
    if slop > 0.7:                             return "reject"     # excessive slop
    if verdict_pass and score >= 0.6 and slop < 0.4: return "accept"
    return "revise"
```

`score` is the rubric-weighted run score (task success 0.30, groundedness
0.20, objection coverage 0.10, requirement fidelity 0.10, …; the domain
shifts the weights — see reference). Wrap it with `dspy.Refine(module,
N=3, reward_fn=..., threshold=0.6)` when "revise" should retry in-process.

## Evolution (Tier 2, only with a benchmark)

- Trainset = past runs as `dspy.Example(contract=..., evidence_summary=...,
  failure_focus="<what the slop feedback said>")`; the metric is
  `0.4 * plan_covers_failure_focus + 0.6 * slop_score(...).score`.
- `dspy.GEPA(metric=..., auto="light", reflection_lm=...)` on the `Dialectic`
  → the **challenger**; the current compiled program is the **champion**.
- `promote(champion, challenger)` only if challenger score > champion score
  **and** challenger slop ≤ champion slop **and** every canary passes. Canary
  = an example whose text is deliberately contradictory; it passes only when
  the output contains hedges (`must_include`) and no certainty words
  (`must_not_include`). A challenger that "wins" by sounding confident fails
  the canary and is not promoted. Keep the previous champion for rollback.

## Anti-patterns

- Parsing "Claim / Objection / Severity" out of free text — the original needed four regex styles; typed `list[Objection]` output removes the parser and the parser-gap penalty.
- Letting `Verify` see the thesis or synthesis — that is self-verification; it sees the contract and the output only.
- Reading `slop_score` as a number — its feedback is the GEPA signal; log it.
- Re-compiling the contract mid-run — compare `source_hash`; a changed task is a new run.
- Loading Tier 2 for one-off work; promotion needs a benchmark and canaries, not one run.
- Treating "no objections" as coverage 1.0 without checking that the antithesis actually ran (the original's parser-gap bug).

## Where to go next

- Evidence for long inputs → `dspy-rlm-module`; verified multi-step execution → `dspy-rlm-workflow`
- Metric conventions and gold sets → `dspy-evaluation-harness`; compile the challenger → `dspy-gepa-optimizer`
- Corrections from the user become gold for the same metric → `dspy-reflect-loop`
- Full reference (contract defaults, all 12 heuristics, rubric weights, promotion rule, canary format) → [reference.md](reference.md)
- Runnable, dry-run capable → [example_autodialectics.py](example_autodialectics.py)
