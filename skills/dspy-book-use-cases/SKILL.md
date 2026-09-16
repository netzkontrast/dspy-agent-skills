---
name: dspy-book-use-cases
description: >-
  Seven complete DSPy applications from chapter 9 of Context Engineering with
  DSPy, as a pattern library routed by task shape — classification, structured
  extraction with a committed regression benchmark, tool-using support agent
  with sandwich guardrails, parallel research fan-out, computational analysis
  graded on its reasoning trajectory, and two creative generators that score
  output with embeddings or a vision judge instead of a gold string.
when_to_use: >-
  User is starting a new DSPy application and asks "how should I structure
  this", "is there an example like mine", "how do I test a DSPy app", "how do
  I guard an agent against injection", "how do I score creative output"; a
  reference architecture is wanted before writing from scratch.
---

# Seven Applications (book chapter 9)

From the O'Reilly companion repo (MIT). Find the entry closest to your task
shape, copy its architecture, replace its domain.

## Route by shape

| Your task | Example | The move worth copying |
|---|---|---|
| one label per input, fuzzy boundary | sentiment classifier | inline signature plus few-shot search; the **demos** carry the label definition, not the prompt |
| many fields from unstructured text | invoice extraction | partial-credit metric, per-field majority voting, a committed benchmark artifact |
| answer from private data with tools | customer service | narrow tools inside a sandwich guardrail |
| one question needing many lookups | news researcher | decompose, fan out in parallel, synthesize while preserving conflicts |
| numbers over long documents | financial analyst | code execution plus a metric that grades the reasoning trajectory |
| text with no gold answer | blog writer | embedding distance to reference passages |
| an expensive irreversible call | video generator | a quality gate placed *before* the expensive step |

Routing rule: is there a gold answer? If yes, classification or extraction with
accuracy or partial credit plus an optimizer. If no, a judge or embedding
reward with `Refine`. Does it need lookup? Retrieval. Arithmetic? Code
execution with a deterministic parse in the metric.

## The one to copy first: a committed benchmark

The invoice extractor is the only example with a test, and its structure is the
transferable part. It runs four approaches over the same holdout, records
per-approach token counts and cost, and serializes everything into one result
document with a status, the model names, the train and holdout identifiers, and
an explicit limitations list.

Then the test asserts against **the committed artifact**, not against a live
run. Three checks:

1. Every notebook cell parses, and no outputs are committed.
2. The holdout is exactly the declared holdout, train and holdout are disjoint, and each approach has exactly the expected number of predictions drawn only from holdout identifiers. **That is a leakage guard.**
3. The recorded scores still match, and the diagnosis still holds.

The test is deterministic, offline, free and fast. This is how a DSPy
application gets a regression test without burning an API budget in CI.

**And it records an honest failure.** The run did not reproduce the book's
illustrative table. The task model scored perfectly on all three training
examples, so reflective mutation was suppressed, 393 task calls happened with
zero reflection calls, and the learned instruction came back identical to the
original. The only gain came from few-shot, from one repeat correctly emitting
empty strings for genuinely absent fields. Shipping that finding, in the
repository, is the practice to copy.

## Two metric patterns worth stealing

**Deterministic gate before the expensive judge.** The financial analyst parses
the numeric answer first, normalizing currency and percentages, and returns
zero on a wrong number **without calling the judge at all**. Only a correct
answer reaches the LLM judge, which then grades the trajectory: partial credit
for a right answer reached through poisoned or drifting reasoning. Cheap check
first, expensive check second, and partial credit that encodes *why*.

**Grade the trajectory, not just the answer.** A right answer from bad
reasoning is a latent failure. This is the chapter's most sophisticated metric
and it works in plain evaluation, where the optimizer-only trace argument is
absent.

## Guardrails as a sandwich

The support agent wraps the ReAct call on both sides: a cheap compiled-regex
check, then an LLM injection classifier, then — only if both pass — the agent;
then deterministic redaction of the output. A detected injection returns a
fixed refusal that **never reaches the tools**.

Two design choices carry the weight: the cheap check runs first so most attacks
cost nothing, and the tools are deliberately narrow rather than general.

## What the chapter does not give you

- **No Pydantic models anywhere.** Typed outputs are primitives and generics only. If you want validated structured output, you are extending the chapter, not copying it.
- **No human-in-the-loop step** in any of the seven. The closest is an instruction to hand-label a small set and confirm the judge agrees — judge calibration, not a runtime approval gate.
- **Two examples are incomplete**: the blog writer defines a metric it never optimizes with, and the news researcher has no metric at all.

## Anti-patterns

- Copying the two unannotated signatures; the other five use the typed form and should be your model.
- An explicit `rationale` output field alongside `ChainOfThought`, which already injects one.
- Declaring a numeric output as a string and parsing it with `float()`.
- Putting the quality gate after the expensive generation instead of before it.
- Calling a judge on every example when a deterministic check could reject most of them first.
- Treating the blog writer as a complete example; its metric is never used.

## Where to go next

- The metric shapes these use → `dspy-book-metrics`, `dspy-evaluation-harness`
- Agent construction and tool contracts → `dspy-book-agents`
- Module choice and multimodal inputs → `dspy-book-modules`
- Serving and observability → `dspy-book-production`, `dspy-production`
- Full reference (all seven architectures, benchmark numbers) → [reference.md](reference.md)
- Runnable, dry-run capable → [example_use_case_router.py](example_use_case_router.py)
