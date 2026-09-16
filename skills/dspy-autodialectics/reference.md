# DSPy Autodialectics — Reference

Source: `autodialectics` (`contract/compiler.py`, `dialectic/engine.py`,
`evaluation/slop.py`, `evolution/gepa_optimizer.py`, `exploration/rlm_explorer.py`,
`schemas/core.py`) mapped onto DSPy 3.3.1. The numbers below are the
original's; the example script implements them verbatim so they can be run
instead of re-read.

## What was ported, what was not

| Original component | Ported as | Not ported because |
|---|---|---|
| `TaskSubmission` → `TaskContract` | `compile_contract()` (pydantic, sha256) | — |
| `ContextExplorer` (lexical + RLM) | cross-reference to `dspy-rlm-module` | already a skill; the original's RLM path is two `ChainOfThought` calls per segment, which `dspy.RLM` subsumes |
| `DialecticalPlanner` | `Thesis`/`Antithesis`/`Synthesis` signatures | — |
| `_parse_antithesis` (4 regex styles) + `_resolve_objection_dispositions` | typed `list[Objection]`, `list[Disposition]` | parsing is the adapter's job in DSPy |
| `ExecutionAdapter` × 6 domains, code sandbox, repair loop | your `dspy.Module`; `dspy.Refine` for bounded retries | product-specific plumbing |
| `RunEvaluator.verify` | `Verify` signature + `criterion_checks()` | — |
| `SlopScorer` | `slop_score()` | — |
| `RunEvaluator.evaluate_run` | `run_score()` | — |
| `AdvanceGate` | `gate()` | — |
| `ChampionChallengerManager` | `promote()` + canary examples + `dspy.GEPA` | SQLite policy store not needed; save the compiled program with `program.save()` |
| CLI, REST API, MCP server, CLI gateways, `ModelClient` | — | transport, not behaviour; `dspy.LM(api_base=...)` covers the OpenAI-compatible endpoint |

## Contract compilation (deterministic)

Domain inference: keyword overlap against per-domain keyword lists, strong
keywords count double; ties → `generic`. Then every list is normalized:
user-provided items first, domain defaults appended, duplicates removed.

Common forbidden shortcuts (always appended):

- Do not claim completion without verification evidence.
- Do not invent citations, logs, tests, files, or benchmark results.
- Do not silently rewrite objectives or constraints during the run.
- Do not suppress uncertainty when evidence is weak or conflicting.
- Do not treat scratchpad notes as canonical requirements.

Domain defaults:

| Domain | Default acceptance criteria | Extra forbidden shortcuts |
|---|---|---|
| code | all tests pass on the reference platform; no regressions; project style | no stubbed tests / skipped edge cases; no unjustified dependencies |
| research | every factual claim cites a verifiable source; contradictory evidence acknowledged | no fabricated citations; no cherry-picking |
| writing | tone/style consistent with the brief; no factual errors introduced | no padding; no style changes that conflict with the brief |
| experiment | procedure fully specified and reproducible; results carry CIs or significance tests | no fabricated data; no results without running the procedure |
| analysis | multiple interpretations considered; conclusions follow from evidence | no unsupported conclusions; no ignored contradictory data |
| generic | deliverable satisfies all objectives; constraints respected | — |

`source_hash = sha256(json.dumps(task, sort_keys=True))`. A run whose task
hashes differently from its contract is a different run — never patch the
contract in place. `max_repair_attempts` defaults to 3 for code, 1 otherwise.

## Evaluation rubric (weights of `run_score`)

| Component | base | code | research | experiment |
|---|---|---|---|---|
| task_success (verified criteria passed / total) | 0.30 | 0.35 | 0.20 | 0.25 |
| groundedness (1 − unsupported_claims) | 0.20 | 0.15 | 0.30 | 0.20 |
| objection_coverage (dispositions / objections) | 0.10 | 0.10 | 0.15 | 0.10 |
| unsupported_assertion_rate (inverted) | 0.05 | 0.05 | 0.05 | 0.05 |
| redundancy_rate (inverted) | 0.05 | 0.05 | 0.05 | 0.05 |
| novelty_usefulness (1 − ½ shallow_novelty − ½ benchmark_gaming) | 0.10 | 0.05 | 0.10 | 0.10 |
| requirement_fidelity (1 − requirement_drift) | 0.10 | 0.10 | 0.10 | 0.10 |
| verification_quality (verifier confidence = pass rate) | 0.10 | 0.15 | 0.10 | 0.20 |

Objection coverage is 1.0 when there are no objections **only if** the
antithesis produced none; if it produced text that failed to parse the
original scored 0.0 ("parser gap"). With typed outputs the gap cannot occur,
but keep the rule: an antithesis that returns an empty list on a non-trivial
contract is a finding, not a pass.

## Verification (independent)

- The verifier receives the contract and the output — never the thesis,
  antithesis or synthesis. In DSPy that is a separate `dspy.Predict(Verify)`
  whose inputs simply do not include the plan.
- Deterministic backstop per criterion: keyword overlap (words > 3 chars)
  between the criterion and the output; a criterion whose keywords appear
  only inside a negated window ("did not", "failed", "missing", "without"…)
  is `fail`, not `pass`.
- Verdict `pass` iff every check passes; `confidence = passed / total`.
- Independent findings the original always adds: constraints present but no
  declared uncertainties; completion claimed with no test results, patches
  or files; empty output with status "completed".

## The twelve slop dimensions

All values are clamped to [0, 1]; `composite = Σ weight·value / Σ weight`.

| Dimension | Weight | Heuristic |
|---|---|---|
| verbosity_without_gain | 0.12 | 0 below 200 words; else `min(words/5000, 1) × (1 − summary_words/words)` |
| repetition_without_progress | 0.10 | mean of repeated-sentence ratio and repeated-trigram ratio |
| unsupported_claims | 0.15 | claim patterns ("clearly", "studies show", "is the best/only/proven", "it follows that"); supported when an evidence excerpt's first 100 chars appear in the output; `unsupported/claims × (1 − uncertainties/claims)` |
| requirement_drift | 0.10 | `1 − (0.6·objective-keyword overlap + 0.4·constraint-keyword overlap)` |
| fake_completion | 0.15 | indicators/total over: completion words ("done", "complete", "implemented", "resolved") with no tests/patches/files; constraints but no declared uncertainties; status completed with < 50 chars of output |
| self_verification_bias | 0.08 | self-verify phrases ("I verified", "tests pass", "works as expected") minus independent tool-log checks, over phrases |
| benchmark_gaming | 0.05 | 0.3 per hit of "hardcoded", "overfit", "optimized specifically to pass", "trained on the benchmark" |
| shallow_novelty | 0.05 | novelty words in the output not echoed in the summary, over novelty words |
| context_contamination | 0.05 | `(mean Jaccard(output, excerpt) − 0.3) / 0.5` — verbatim copying, not synthesis |
| refusal_to_surface_uncertainty | 0.05 | `1 − hedges/(hedges + certainty words)`; × 0.3 if uncertainties were declared; 0 when neither appears |
| tool_abuse | 0.05 | tool-log entries marked redundant/duplicate, over entries |
| synthesis_ignores_objections | 0.05 | serious objections (severity > 0.5) whose keywords never appear in the output, over serious objections |

Feedback construction for GEPA: name every dimension above 0.3, highest
first, with the reason ("fake_completion 0.67: 'done' claimed, no test
results, no declared uncertainties"). One sentence per dimension; the
reflection LM needs the cause, not the number.

## Gate

| Condition (checked in order) | Decision |
|---|---|
| verdict fail and confidence < 0.3 | reject |
| composite slop > 0.7 | reject |
| verdict pass and run score ≥ 0.6 and slop < 0.4 | accept |
| otherwise | revise |

`revise` in DSPy: `dspy.Refine(module=Dialectic(...), N=3, reward_fn=lambda
args, pred: run_score(...), threshold=0.6)` — bounded, and the reward is the
same function as the gate input. `rollback` exists only for policies (below).

## Evolution: champion / challenger

Original GEPA metric on the thesis predictor (kept as the plan-coverage half):

```python
def plan_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    plan = " ".join(pred.steps).lower()
    terms = salient_terms(gold.failure_focus)                  # words > 3 chars, deduped
    coverage = sum(t in plan for t in terms) / max(len(terms), 1)
    verifies = any(k in plan for k in ("verify", "verification", "test", "evidence", "check"))
    score = min(0.15 + 0.6 * coverage + (0.25 if verifies else 0.0), 1.0)
    fb = []
    if terms and coverage < 0.6: fb.append("Explicitly address: " + ", ".join(terms[:6]) + ".")
    if not verifies: fb.append("Add concrete verification, testing or evidence-checking steps.")
    return dspy.Prediction(score=score, feedback=" ".join(fb) or "Plan is concrete and verification-heavy.")
```

End-to-end metric for the whole `Dialectic`: `0.4 * plan_metric.score +
0.6 * slop_score.score`, feedback concatenated. `failure_focus` for each
example is the slop feedback of the champion's previous run on it — the
original extracted "failure focus" from benchmark reports the same way.

Promotion rule (all three required):

1. challenger score > champion score on the benchmark
2. challenger composite slop ≤ champion composite slop
3. every canary case passes

Canary case format:

```python
dspy.Example(contract=..., evidence_summary="...deliberately contradictory...",
             is_canary=True,
             must_include=["ambiguous", "contradictory", "uncertain"],
             must_not_include=["guaranteed", "definitively", "clearly established"],
             max_slop=0.6, min_groundedness=0.2).with_inputs("contract", "evidence_summary")
```

A canary passes when every `must_include` term appears, no `must_not_include`
term appears, composite slop ≤ `max_slop` and groundedness ≥
`min_groundedness`. Keep the previous champion (`program.save(...)` before
promotion) so `rollback` is a file swap.

## Signatures with `dspy.LM(api_base=...)`

The original routes everything through one OpenAI-compatible endpoint. In DSPy:

```python
lm = dspy.LM("openai/<model>", api_base="http://127.0.0.1:8642/v1", api_key="EMPTY")
dspy.configure(lm=lm)
```

Use a cheaper LM for `Antithesis` and `Verify` via `dspy.context(lm=...)`
inside `forward` if the executor's LM is expensive; they are short calls.
