---
name: dspy-wiki-compile
description: >-
  Compile immutable sources into an LLM-maintained wiki with DSPy — triage,
  extract claims with line-range citations, merge concepts across a whole
  batch before any page is written (two-phase compile), decide per existing
  page between flag / update / create with active-page protection, print a
  knowledge diff (reinforced, challenged, new, gaps) and answer questions
  from pages only, naming gaps. A deterministic metric (citations resolve,
  decisions legal, disagreements two-sourced, diff consistent, entities
  linked, language kept) makes every stage GEPA-optimizable. Use when raw
  documents must become a maintained, cited knowledge layer.
when_to_use: >-
  User says "ingest", "compile the wiki", "build a knowledge base from these
  documents", "LLM wiki", "concept pages", "knowledge diff", "what changed
  with this source"; a folder of raw sources must become interlinked pages
  with citations; a page that a human reviewed must not be overwritten by
  a new source; the same concept appears in several sources and must merge
  into one page instead of duplicating.
---

# DSPy Wiki Compile (3.3.x)

The Karpathy LLM-wiki pattern as a DSPy program: **raw sources are immutable,
the LLM maintains the wiki, a schema is the contract.** Five repos taught the
details: `llm-wiki-agent` (post-ingest validation, health before lint),
`llm-wiki-compiler` (two-phase compile, line-range citations, freshness),
`synthadoc` (decision rules, active-page protection, truncation flag),
`quicky-wiki` (knowledge diff), and the Karpathy bootstrap skill (concept
table, contradiction block with a resolution field). The program returns
drafts; **writing them anywhere is the caller's separate, reviewed step.**

## Stages and what is deterministic

| Stage | Signature | LM | Deterministic guard |
|---|---|---|---|
| triage | `TriageSource` → `Triage{tier, category, language, truncated}` | yes | `truncated` recorded whenever a body was capped |
| extract | `ExtractClaims` → `list[Claim]`, each with `Citation{file, start, end, quote}` | yes | `citation_resolves`: range inside the file and quote a substring of those lines |
| merge (phase 2) | `MergeConcepts` over *all* extractions of the batch → `list[ConceptDraft]` | yes | runs only after every source is extracted; disagreements need two distinct sources |
| decide | `DecideIngest(page, concept)` → `IngestDecision{action: flag \| update \| create, conflicts}` | yes | `decision_legal`: `update` on a `reviewed`/`locked` page with conflicts is illegal; `create` is decided in code when no page exists |
| diff | `KnowledgeDiff(page, concept)` → `Diff{reinforced, challenged, new, gaps}` | yes | `diff_consistent`: challenged ⊆ conflicts, new ∩ page = ∅ |
| answer | `AnswerQuery(question, pages)` → `Answer{text, cited_pages, confidence, gaps}` | yes | gaps named when retrieval is thin (caller's rule) |
| health | — | no | broken links, unindexed pages, stale hashes, orphans: a lint, not a predictor |

## Canonical program

```python
class BatchCompile(dspy.Module):
    def __init__(self):
        super().__init__()
        self.triage = dspy.ChainOfThought(TriageSource)
        self.extract = dspy.ChainOfThought(ExtractClaims)
        self.merge = dspy.ChainOfThought(MergeConcepts)
        self.decide = dspy.ChainOfThought(DecideIngest)
        self.diff = dspy.ChainOfThought(KnowledgeDiff)
        self.answer = dspy.ChainOfThought(AnswerQuery)

    def forward(self, sources: dict[str, str], pages: dict[str, PageState], known_entities: list[str]):
        extractions = [Extraction(source=name, triage=self.triage(source_name=name, text=text).triage,
                                  claims=self.extract(source_name=name, numbered_text=number_lines(text),
                                                      known_entities=known_entities).claims)
                       for name, text in sources.items()]                   # phase 1: every source
        concepts = self.merge(extractions=extractions, known_entities=known_entities).concepts   # phase 2
        decisions, diffs = [], {}
        for concept in concepts:                                              # phase 3: per existing page
            page = pages.get(concept.slug)
            if page is None:
                decisions.append(IngestDecision(slug=concept.slug, action="create", rationale="no page yet"))
                continue
            decisions.append(self.decide(page=page, concept=concept).decision)
            diffs[concept.slug] = self.diff(page=page, concept=concept).diff
        return dspy.Prediction(compiled=Compiled(extractions=extractions, concepts=concepts,
                                                 decisions=decisions, diffs=diffs))
```

Models, signatures with their rule docstrings, and the metric:
[example_wiki_compile.py](example_wiki_compile.py).

## The metric encodes the rules that prompts only state

`compile_metric(gold, pred)` returns `dspy.Prediction(score, feedback)`; every
axis is deterministic, so GEPA cannot optimise the compiler into confident
prose.

| Axis | Weight | Deficit named in feedback |
|---|---|---|
| citations resolve | 0.30 | `citation does not resolve: file:start-end 'quote'` |
| decisions legal | 0.25 | `illegal decision: update on <slug> (status reviewed, conflicts […])` |
| merge | 0.20 | disagreement with one source; `contradicted` status without a pending disagreement (or vice versa); sources outside the batch; no citation |
| diff consistent | 0.15 | challenged item without a conflict; `new` item already on the page |
| links + language | 0.10 | known entity not wrapped as `[[…]]`; definition not in the source language |

## Rules the program enforces on the pipeline

1. **Sources are never modified.** The program reads numbered text and returns citations into it.
2. **Extract everything before merging anything.** A concept that appears in
   three sources is one draft with three sources, not three drafts.
3. **A reviewed page is authoritative.** Conflicting content is `flag`ged with
   the conflicts listed; `update` is only for additions without dispute.
   The metric scores a protected update at 0.
4. **The diff is the receipt.** Print reinforced / challenged / new / gaps to
   the human before anything is written; challenged must map to a conflict.
5. **Drafts go to a candidates area.** Promotion is a human step that pins
   the reviewed content hash; the program never writes pages.
6. **Health before lint.** Broken links, unindexed pages and stale source
   hashes are free checks; run them before spending tokens on semantic lint.

## Gold set and optimization

Twenty to thirty sources with hand-checked claims, five of which must `flag`
an existing reviewed page, and a handful of deliberately truncated bodies.
`dspy.GEPA(auto="light")` over `extract`, `merge` and `decide` with
`compile_metric`; keep `answer` on a separate gold set of question → cited
answer pairs. For sources above ~100k tokens, extract with `dspy.RLM`
(`dspy-rlm-module`) and record `truncated: false`.

## Anti-patterns

- Merging inside the extraction loop; the second source then overwrites the first.
- Letting the LM decide `create` for a slug that has no page; that is a lookup, not a judgement.
- Storing citations as page names instead of line ranges; the lint can then prove nothing.
- Treating `flag` as failure; it is the moment the human learns something.
- A metric with an LLM judge as the floor; the judge may sit on top of these axes, never below them.

## Where to go next

- The precision gate before a draft is promoted → `dspy-clarify`
- Contested merges (supersede, delete) → `dspy-tetraframe`
- An independent review of a page before it becomes `reviewed` → `dspy-adversarial-review`
- Questions the wiki cannot answer → `dspy-deep-refine`
- Full reference (models, enums, feedback strings, lint boundary) → [reference.md](reference.md)
- Runnable example → [example_wiki_compile.py](example_wiki_compile.py)
