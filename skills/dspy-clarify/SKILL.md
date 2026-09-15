---
name: dspy-clarify
description: Add a clarify gate to any DSPy pipeline where accuracy matters — a Signature that rewrites a claim, requirement or question so its scope, terms and assumptions are explicit without adding or losing meaning, binds names to a known glossary, and turns every remaining ambiguity into a question for the human instead of picking a reading. A deterministic metric (no smuggled terms, no new quantifiers, grounded scope, resolved hedges, well-formed questions, language kept) makes the gate GEPA-optimizable. Use before promoting research into a canonical store, before decomposing a task, and before refining a knowledge base.
when_to_use: >-
  User says "clarify", "make this precise", "resolve the ambiguity", "what
  exactly does this claim say", "before this becomes canon"; a pipeline
  promotes content from a draft or research layer into an authoritative one;
  a task statement is vague before decomposition; or a query must be pinned
  down before a knowledge base is changed for it.
---

# DSPy Clarify (3.2.x)

Transposition of the `clarify` code skill (Hmbown/clarify, Apache 2.0: reveal
intent, make the implicit explicit, add nothing, remove noise, never change
behaviour) from code to **statements**. The gate takes a claim and its source
and returns the same claim with explicit scope, bound terms, named
assumptions, and a list of open questions. It does not decide anything the
source does not decide — that is the whole point: precision by *asking*, not
by guessing. In DSPy the "never change behaviour" rule becomes a
deterministic metric, so optimization can only make the gate clearer, not
bolder.

## Where it plugs in

| Stage | Before | Gate | After |
|---|---|---|---|
| research → canon promotion | a wiki claim with a citation | `ClarifyGate` → `verdict` | `clear` may be proposed for promotion; `needs-author` goes to the question list; `not-promotable` stays research |
| context-heavy work (`dspy-rlm-workflow` Initialize) | vague problem statement | clarify the statement and success criteria | decomposition starts from explicit scope and terms |
| knowledge-base refinement (`dspy-deep-refine`) | a failing query | clarify the query first | the loop refines the base for the question actually meant |
| learning from corrections (`dspy-reflect-loop`) | "use X instead of Y" | clarify old/new behaviour and scope | the gold example says exactly what changed |

## Canonical program

```python
import dspy
from pydantic import BaseModel, Field
from typing import Literal

UNSPECIFIED = "unspecified"

class Ambiguity(BaseModel):
    phrase: str                                   # the ambiguous span, verbatim
    readings: list[str] = Field(min_length=2)     # the distinct readings
    question: str                                 # what the human must decide; ends with "?"

class Binding(BaseModel):
    mention: str
    slug: str                                     # must be a known glossary slug

class Scope(BaseModel):
    world: str = UNSPECIFIED                      # domain-specific axes: world / act / character part …
    act: str = UNSPECIFIED
    part: str = UNSPECIFIED

class Clarification(BaseModel):
    clarified_text: str
    scope: Scope = Field(default_factory=Scope)
    assumptions: list[str] = Field(default_factory=list)
    ambiguities: list[Ambiguity] = Field(default_factory=list)
    bindings: list[Binding] = Field(default_factory=list)
    verdict: Literal["clear", "needs-author", "not-promotable"]

class ClarifyClaim(dspy.Signature):
    """Rewrite the claim so that a reader with the glossary understands exactly what
    it asserts, and nothing more: make the scope explicit only where the source
    states it, bind names to glossary slugs, state implicit assumptions as
    assumptions, and turn every remaining ambiguity into a question instead of
    choosing a reading. Never add, drop or generalise content; keep the source language."""
    claim_text: str = dspy.InputField()
    source_excerpt: str = dspy.InputField(desc="the cited lines of the source")
    entities: list[str] = dspy.InputField()
    glossary_terms: str = dspy.InputField(desc="comma-separated known slugs")
    canon_context: str = dspy.InputField(desc="authoritative passages about the same entities, or empty")
    clarification: Clarification = dspy.OutputField()

class ClarifyGate(dspy.Module):
    def __init__(self):
        super().__init__()
        self.clarify = dspy.ChainOfThought(ClarifyClaim)
    def forward(self, **inputs) -> dspy.Prediction:
        return dspy.Prediction(clarification=self.clarify(**inputs).clarification)
```

The scope axes are the domain's: for a novel, world / act / character part;
for a codebase, module / version / platform. Keep them as strings with an
`unspecified` default so the metric can check them.

## The metric is the "never change behaviour" rule

`clarify_metric` (see [example_clarify.py](example_clarify.py)) is fully
deterministic and returns `dspy.Prediction(score, feedback)`:

| Axis | Check | Weight |
|---|---|---|
| meaning kept | no glossary term in the output that is absent from claim + source + context; no entity dropped; no quantifier (*alle, immer, nie, all, never, only* …) the source does not state | 0.30 |
| scope grounded | every scope value other than `unspecified` occurs in claim, source or context | 0.15 |
| hedges resolved | hedge words (*meist, wohl, vielleicht, somehow, probably* …) do not increase, and any that remain are declared as ambiguities | 0.15 |
| bindings valid | every binding slug is a known glossary term | 0.15 |
| questions well-formed | each ambiguity has ≥ 2 readings and a question ending in `?`; `verdict == "clear"` iff no ambiguities | 0.15 |
| language kept | a German claim stays German | 0.10 |

Feedback names the offending terms, quantifiers, slugs or hedges, so GEPA gets
concrete blame. Because the axes are lexical, the gate cannot be optimized into
"sounding precise": a rewrite that adds a world or an *always* the source lacks
scores lower than the vague original.

## Rules the gate enforces on the pipeline

1. **`clear` is the only verdict that may propose promotion.** A `needs-author`
   clarification is not a failure; its questions are the deliverable.
2. **Questions are not answered by the model.** A second call with the same
   input must not resolve an ambiguity the first call raised; only the human
   (or a source passage) does.
3. **Bindings are to the glossary you pass in.** Unknown slug → LOW, same as
   `dspy-deep-refine`'s ambiguous node names.
4. **Assumptions are labelled.** Anything the clarified text needs that the
   source does not say goes to `assumptions`, never into `clarified_text`.
5. **Dry-run first.** The gate returns a `Clarification`; writing it anywhere
   (wiki page, promotion proposal, gold set) is the caller's separate step.

## Gold set and optimization

Twenty to forty claims with a hand-written clarification each are enough for
`dspy.GEPA(auto="light")`. Include claims where the right answer is
`needs-author`, so the optimizer learns that asking scores as high as
resolving. Fold `clarify_metric` into a promotion pipeline's metric with a
weight ≥ 0.3 so that clarity cannot be traded for throughput.

## Anti-patterns

- Letting the gate "fix" the claim from canon context — context is for binding
  names and detecting conflicts, not for rewriting what the source says.
- Domain-free scope: a scope model with no axes gives the metric nothing to ground.
- Treating hedges as noise to delete — a hedge the source uses is meaning; it is either grounded (source says *in Akt I*) or an ambiguity.
- Running the gate after promotion; it is a gate, not a linter.
- Skipping the glossary input — bindings become free text and the metric's binding axis is silently trivial.

## Where to go next

- The promotion path this gate guards → `dspy-deep-refine` (apply gate) and your project's promotion command
- Metric contract → `dspy-evaluation-harness`
- Optimizing the gate → `dspy-gepa-optimizer`
- Full reference (models, metric axes, feedback strings, extension points) → [reference.md](reference.md)
- Runnable example → [example_clarify.py](example_clarify.py)
