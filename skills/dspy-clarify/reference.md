# DSPy Clarify — Reference

Source of the principle: `Hmbown/clarify` (`code-clarifier` skill, Apache 2.0):
reveal intent through naming, restructure for readability, document only
where naming cannot, follow project standards, avoid over-clarification, and
above all *never change what it does*. This skill applies the same five rules
to statements and turns the last one into a metric.

## Rule mapping (code → statement)

| Code clarifier | Clarify gate |
|---|---|
| rename to reveal intent | bind mentions to glossary slugs (`bindings`) |
| magic values → named constants | implicit scope → explicit `scope` fields, only where the source states them |
| cryptic condition → predicate | hedge or vague span → `Ambiguity(phrase, readings, question)` |
| make assumptions explicit (types, guards) | `assumptions[]` |
| comments explain *why*, never *what* | `clarified_text` says what the source says, no more; rationale stays out |
| follow project standards | glossary + scope axes + language are inputs, not model choices |
| avoid over-clarification | metric penalises added terms, added quantifiers, ungrounded scope |
| never change behaviour | metric axis "meaning kept" (0.30) plus grounded scope |

## Models

```python
class Ambiguity(BaseModel):
    phrase: str
    readings: list[str] = Field(min_length=2)
    question: str

class Binding(BaseModel):
    mention: str
    slug: str

class Scope(BaseModel):            # domain axes; extend, keep "unspecified" defaults
    world: str = "unspecified"
    act: str = "unspecified"
    part: str = "unspecified"

class Clarification(BaseModel):
    clarified_text: str
    scope: Scope
    assumptions: list[str]
    ambiguities: list[Ambiguity]
    bindings: list[Binding]
    verdict: Literal["clear", "needs-author", "not-promotable"]
```

`not-promotable` is for inputs that are not claims (instructions, questions,
fragments) or that contradict their own citation; the metric does not score
it differently, the pipeline does (it never proposes promotion).

## Signature inputs

| Input | Purpose | Empty allowed |
|---|---|---|
| `claim_text` | the statement to clarify, source language | no |
| `source_excerpt` | the cited lines; the only ground for scope and quantifiers | no |
| `entities` | names the claim is about (from extraction) | yes |
| `glossary_terms` | comma-separated known slugs; binding targets | yes, but then the binding axis is trivial |
| `canon_context` | authoritative passages about the same entities; used for binding and conflict awareness, never for rewriting | yes |

## Metric axes in detail

Marker lists (extend per language/domain):

```python
HEDGES = ("irgendwie", "meist", "meistens", "wohl", "vielleicht", "ungefähr", "manchmal",
          "eigentlich", "somehow", "probably", "maybe", "roughly", "sometimes", "kind of", "sort of")
QUANTIFIERS = ("alle", "jede", "jeder", "jedes", "immer", "nie", "niemals", "kein", "keine", "nur",
               "all", "every", "always", "never", "only")
```

| Axis | Computation | Feedback string |
|---|---|---|
| meaning kept | `1 − min(1, 0.5·new_terms + 0.5·dropped_entities + 0.5·new_quantifiers)` where `new_terms` = glossary slugs in the output absent from claim ∪ excerpt ∪ context | "Introduced terms absent from the source: […]." / "Dropped entities: […]." / "Added quantifiers the source does not state: […]." |
| scope grounded | 1 if every non-`unspecified` scope value occurs in claim ∪ excerpt ∪ context, else 0 | "Scope values not found in the source: […]; use 'unspecified'." |
| hedges resolved | 1 if hedge count did not increase and every remaining hedge is contained in some `ambiguities[].phrase`, else 0 | "Hedges left unresolved and undeclared: […]." |
| bindings valid | 1 if every `bindings[].slug` ∈ glossary, else 0 | "Bindings to unknown glossary slugs: […]." |
| questions well-formed | 1 if every question ends with `?` and `(verdict != "clear") == bool(ambiguities)` | "Ambiguities without a question" / "Verdict inconsistent with the ambiguity list" |
| language kept | 0 if the claim is German (function words) and the output contains English function words | "The claim was translated; keep the source language." |

Weights 0.30 / 0.15 / 0.15 / 0.15 / 0.15 / 0.10. A clarification that
resolves nothing but declares its ambiguities correctly scores 1.0 — asking is
a complete answer.

## Extending the gate

- **Second-reader consistency**: run the gate twice with different
  `rollout_id`s and diff the `ambiguities` sets; a phrase flagged by one run
  only is itself a `needs-author` item (cheap uncertainty estimate).
- **Domain scope axes**: replace `Scope` fields; keep the `unspecified`
  default and the grounding rule.
- **Judge for paraphrase**: when lexical grounding is too strict for a domain,
  add an LM judge (cheaper model) as a seventh axis with weight ≤ 0.2; keep the
  lexical axes as hard floors.
- **Gold for GEPA**: `dspy.Example(claim_text, source_excerpt, entities,
  glossary_terms, canon_context)`; the metric needs no gold clarification,
  which makes gold sets cheap — but include hand-checked `needs-author` cases
  to verify the optimizer does not learn to over-resolve.

## Failure modes

| Symptom | Cause | Fix |
|---|---|---|
| everything is `clear` | instruction rewards resolution | gold with expected `needs-author`; the question axis penalises a `clear` verdict with hedges left |
| bindings to near-miss slugs | glossary passed as free text | pass exact slugs; the binding axis is exact-match |
| scope always `unspecified` | source excerpt too short | cite a larger window (the paragraph, not the line) |
| rewrite in English | model default | language axis; keep German markers in the marker list for your language |
| metric too strict on paraphrase | lexical grounding | add the judge axis above, never lower the meaning-kept weight |
