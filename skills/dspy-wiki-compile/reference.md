# DSPy Wiki Compile — Reference

Sources of the pattern and what each contributed:

| Repo | Contribution here |
|---|---|
| Karpathy LLM-wiki bootstrap | raw / wiki / schema layers; concept table statuses (`high-confidence · single-source · tentative · contradicted`); contradiction block with a `resolution` field; "open the page before citing it" |
| `llm-wiki-agent` (MIT) | typed page kinds; post-ingest validation (broken links, unindexed pages); health (free) vs lint (LLM) boundary |
| `llm-wiki-compiler` | two-phase compile (extract all, then merge); `^[file:L-L]` citations validated deterministically; freshness by source hash |
| `synthadoc` (AGPL, patterns only) | ingest decision rules RULE 1 / 1b / 2 / 2b / 3; active-page protection; `truncated` flag; staged candidates |
| `quicky-wiki` (MIT) | knowledge diff (`reinforced · challenged · new · gaps`) printed per ingest |

## Models

```python
class Citation(BaseModel):     file: str; start: int; end: int; quote: str            # 1-based, inclusive
class Claim(BaseModel):        text: str; citation: Citation; kind: Literal["fact","definition","rule","event","opinion"]; entities: list[str]
class Triage(BaseModel):       tier: Tier; category: str; language: str; truncated: bool
class Extraction(BaseModel):   source: str; triage: Triage; claims: list[Claim]
class PageState(BaseModel):    slug: str; status: PageStatus; body: str
class Disagreement(BaseModel): topic: str; sources: list[str]; positions: list[str]; resolution: Resolution
class ConceptDraft(BaseModel): slug: str; definition: str; sources: list[str]; citations: list[Citation]
                               disagreements: list[Disagreement]; status: ConceptStatus
class IngestDecision(BaseModel): slug: str; action: Action; rationale: str; conflicts: list[str]
class Diff(BaseModel):         reinforced: list[str]; challenged: list[str]; new: list[str]; gaps: list[str]
class Answer(BaseModel):       text: str; cited_pages: list[str]; confidence: Literal["high","medium","low"]; gaps: list[str]
class Compiled(BaseModel):     extractions; concepts; decisions; diffs: dict[str, Diff]
```

## Closed enums

| Enum | Values | Note |
|---|---|---|
| `Tier` | `primary · secondary · superseded · duplicate · out-of-scope` | replace with the domain's tiers; keep it closed |
| `Action` | `flag · update · create` | synthadoc's three outcomes; `create` is decided in code |
| `PageStatus` | `draft · reviewed · locked · contested · archived` | `reviewed` and `locked` are protected |
| `ConceptStatus` | `high-confidence · single-source · tentative · contradicted` | Karpathy concept-table statuses |
| `Resolution` | `pending · supersedes · both-valid` | `pending` forces `ConceptStatus.contradicted` |

## Decision rules (from synthadoc's `_DECISION_PROMPT`, encoded twice)

| Rule | In the signature docstring | In the metric |
|---|---|---|
| 1 — the source disputes the page → `flag` | yes | `challenged` items must map to `conflicts` |
| 1b — a `reviewed`/`locked` page is authoritative → `flag`, never `update` | yes | `update` + protected status + conflicts scores 0 and is named |
| 2 — additions without dispute → `update` | yes | `new` items must be absent from the page |
| 2b — a comprehensive profile of one entity → its own page | prompt only | — |
| 3 — no page → `create` | code, not LM | `create` on an existing page scores 0 |

## Metric

```python
compile_metric(gold, pred, trace=None, pred_name=None, pred_trace=None) -> dspy.Prediction(score, feedback)
```

`gold` fields: `sources: dict[name, text]`, `pages: dict[slug, PageState]`,
`known_entities: list[str]`, `language: str`. Weights: citations 0.30,
decisions 0.25, merge 0.20, diffs 0.15, links + language 0.10.

Feedback strings:

| Situation | Feedback |
|---|---|
| all axes clean | `cited, legally decided, merged across sources, consistent diff` |
| a citation range is outside the file or the quote is not in those lines | `citation does not resolve: <file>:<start>-<end> '<quote…>'` |
| protected update | `illegal decision: update on <slug> (status reviewed, conflicts [...])` |
| disagreement with one source | `<slug>: a disagreement needs two distinct sources` |
| status/disagreement mismatch | `<slug>: status <status> does not match its pending disagreements` |
| challenged without conflict | `<slug>: challenged item without a conflict: '<item>'` |
| new item already present | `<slug>: 'new' item already on the page: '<item>'` |
| entity not linked | `<slug>: known entities not linked: [...]` |
| language drift | `<slug>: definition is not in the source language (<lang>)` |

`language_kept` counts a handful of function words per language; extend
`LANGUAGE_MARKERS` for other languages or replace it with the domain's own
detector.

## What stays outside the program (the lint)

These are deterministic checks over files, not predictors. They run before
and after the program, for free:

- broken `[[links]]`, pages missing from the index, orphans, sparse pages
- required frontmatter fields and enum values; illegal lifecycle transitions
- citation ranges against the source files on disk (the same check as the metric, on the written page)
- stale sources by `sha256`; `truncated: true` warnings
- candidates older than N days; promotion hash mismatch

The program's metric proves the *draft* is right; the lint proves the
*written page* still is.

## Extension points

- **Retrieval before merge.** Pass BM25 hits over existing concept pages into
  `MergeConcepts` as extra `extractions` with `source="wiki:<slug>"`, so a new
  source merges into an existing concept instead of creating a twin.
- **Long sources.** Replace `self.extract` with a `dspy.RLM`-backed extractor
  for bodies above ~100k tokens; keep `Citation` line ranges by giving the RLM
  the numbered text.
- **Epistemic events.** Turn each `Diff` into log lines
  (`claim | <slug> | reinforced|challenged|new | by=<source>`) for a
  per-concept timeline; no confidence decay, supersession is explicit.
- **Batching.** Run merge and lint once per batch of ~25 sources, not per
  source; the two-phase order makes this natural.
