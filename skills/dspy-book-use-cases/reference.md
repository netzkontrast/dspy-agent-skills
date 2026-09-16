# Seven Applications — Reference

Source: chapter 9 of `context-engineering-dspy-book` (MIT). Repo pins
`dspy==3.3.0`. All seven use one task model and require an LLM key.

## The seven

| Example | Composition | Metric | Optimized |
|---|---|---|---|
| sentiment classifier | one `ChainOfThought`, inline signature | accuracy | yes, few-shot random search |
| invoice extraction | `ChainOfThought`, 8 string outputs | partial credit (presence + correctness)/8 | four approaches compared |
| blog writer | two bare signatures, no module | embedding distance to reference passages | **no** — metric defined, never used |
| news researcher | decompose → `Parallel` × `ReAct` → synthesize | **none** | no |
| video generator | nested modules inside `Refine(N=3, threshold=0.8)` | vision judge, score/10 | no optimizer; inference-time retry |
| customer service | guard → `ReAct` → guard | **none** | no |
| financial analyst | `ProgramOfThought` plus `RLM`, judged separately | deterministic parse, then trajectory judge | no |

Two are architecture demos with no metric. Two more define quality signals they
never optimize against. Treat the chapter as a pattern library, not a set of
finished systems.

## Invoice extraction: the benchmark template

Data: five hand-written invoices, each carrying text plus eight gold fields,
split three train and two holdout. One case has genuinely empty gold fields, to
test that the model emits empty strings rather than inventing values.

Two metrics, deliberately separated:

- a **partial-credit** metric, `((0.5 * present) + (0.5 * correct)) / 8`, with the five-argument GEPA-compatible signature — this is what the optimizer sees;
- a **strict** exact-match fraction used for reporting only, never given to the optimizer.

Four approaches compared: baseline, labelled few-shot, per-field majority
voting over three completions, and GEPA.

Harness: token accounting by snapshotting LM history before and after each
phase, evaluation repeated three times over the holdout, everything serialized
into one result document with generated-at, a status, model names, train and
holdout identifiers, per-approach usage, the learned instructions, and an
explicit limitations list.

Recorded results (2026-08-09, six predictions per approach):

| Approach | Exact | Partial | Cost |
|---|---:|---:|---:|
| baseline | 81.25% | 90.62% | $0.0012 |
| few-shot | **87.50%** | 90.62% | $0.0015 |
| majority voting | 81.25% | 90.62% | $0.0027 |
| GEPA | 81.25% | 90.62% | $0.0013 |

GEPA optimization used 393 task requests and **zero** reflection requests, and
the learned instruction came back identical to the original. The notebook
explains why: the task model scored perfectly on all three training examples,
so perfect-score skipping suppressed reflective mutation. The run is recorded
as it happened rather than tuned until it matched the book's illustration.

The test file is plain `unittest`, static, and never calls an API:

1. every code cell parses, execution counts are cleared, no outputs committed;
2. the holdout is exactly as declared, train and holdout are disjoint, and each approach has exactly six predictions drawn only from holdout identifiers;
3. the four recorded scores still match, reflection requests are still zero, and the learned instruction still equals the original.

Copy the pattern: commit the result artifact, then assert against the artifact.

## Financial analyst: the graded metric

```python
# stage 1, deterministic, free
value = parse_financial_number(pred.answer)      # normalizes $1.25B, -2.8%, €42 million
if not math.isclose(value, gold, rel_tol=0.05):
    return 0.0                                   # judge never called

# stage 2, only for correct answers
verdict = judge(question=..., documents=..., trajectory=pred.trajectory)
return 0.3 if verdict.poisoning_detected else 0.7 if verdict.drift_detected else 1.0
```

`parse_financial_number` **raises** on unparseable input, caught by the metric
as a failure. The judge grades the trajectory for context poisoning and drift,
so a correct answer reached badly scores 0.3 or 0.7 rather than 1.0.

## Customer service: the sandwich

Input guard is a two-layer OR — compiled regex patterns first, then an LLM
classifier with `is_injection: bool` and a reason. On detection, a fixed
refusal returns immediately and the tools are never reached. Output guard runs
deterministic PII redaction.

Tools are deliberately narrow: document search and order lookup, not a general
tool belt.

## Video generator: gate before the expense

`Refine(module=ImageGenerator(), N=3, reward_fn=image_reward, threshold=0.8)`
where the reward is a vision judge scoring the generated image. Only an image
that clears the bar proceeds to the video call. Both external APIs are stubbed
in the notebook, so it runs free.

Note the calling convention: `reward_fn(args_dict, pred)`, not the metric
signature.

## Gaps

- **No Pydantic models anywhere.** Typed outputs are primitives and generics.
- **No human-in-the-loop** in any example. The only human guidance is an instruction to hand-label trajectories and confirm the judge agrees.
- Two notebooks use the legacy unannotated field style; five use the typed form.
- One declares a numeric score as a string and parses it with `float()`.
- One declares an explicit `rationale` output while also using `ChainOfThought`, duplicating the injected reasoning field.
