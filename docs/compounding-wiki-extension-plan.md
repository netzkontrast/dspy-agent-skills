# Extending the Kohärenz Protokoll wiki with compounding engineering

A plan. Nothing here is implemented, and by design almost nothing here requires
changing code that exists.

Source of the ideas: `netzkontrast/dspy-compounding-engineering`, a DSPy
implementation of the compounding-engineering premise — *each unit of
engineering work should make subsequent units easier*. Its mechanisms were read
from source; several of its claims do not survive that reading, and this plan
says which, because the parts worth copying and the parts worth avoiding are
mixed together in the same repository.

## The one-sentence finding

**The wiki already produces learnings. It has no path that reads them back.**

Kohärenz Protokoll writes down what it learns in at least three places, all
structured, all durable:

| Source | Shape | Where |
|---|---|---|
| author decisions | a table: ID, open point, decision, rationale, affected chapters | `Plan/drafting/decision-log*.md`, D-01…D-21 and the Akt II/III log |
| agent memory | headed markdown, appended by the agent during a run | `.claude/agent-memory/<agent>/MEMORY.md` |
| editorial findings | per-chapter reports plus accept/reject triage | `Plan/quality/lit-critic/kap-NN.md` |

Every one of these is written and then read only by a human who happens to
remember it exists. `SourceIngest` does not see D-07 when it ingests a source
about Kapitel 6 sensorics. The lit-critic gate re-raises a finding the author
rejected three chapters ago. That gap — produce but never re-inject — is
exactly what compounding engineering closes, and closing it is additive.

## What the upstream actually does

Four mechanisms, reported as verified from its source.

**1. A three-headed store.** SQLite is the source of truth, a vector index is
the search layer, and a rendered `AI.md` is the human-readable view. One
write funnel, `codify_learning(...)`, feeds all three.

**2. Injection by wrapper, not by retriever.** `KBPredict.wrap(module,
kb_tags=[...])` is a `dspy.Module` that wraps any other module, builds a query
by concatenating the caller's string arguments, retrieves matching learnings,
and **string-prepends** them into the longest string argument. No extra input
field, no demos, no optimizer involvement. That is the design decision worth
studying: it is why adoption required no change to any agent.

**3. Codification at four triggers** — review findings, work outcomes, triage
decisions, and manual entry. Each calls the same funnel.

**4. Metadata-driven agent discovery.** A review agent is any module exporting
a signature with the right class variables; invalid metadata means the agent is
skipped with a warning.

## What not to copy

These are defects in the upstream, verified in its code, not stylistic
disagreements:

- **The injected context string is built with a literal backslash-n** in a non-raw string, so the KB context arrives as one unbroken line with visible escape sequences. The headline feature is degraded by a typo.
- **Similarity is hard-coded to 0.9** in the pattern-matching helper, and its `threshold` argument is ignored. Every vector hit is "similar".
- **A field-name mismatch** between what the codifier emits and what the store expects means auto-codified rows land with an empty title and empty content. Downstream, deduplication skips rows with empty descriptions — so dedup effectively no-ops on exactly the rows the system generates itself.
- **The README claims agents are optimized via DSPy teleprompters.** There is no optimizer, no metric, no trainset and no compiled artifact anywhere in the repository.
- **The README claims JSON storage.** It is SQLite plus a vector index; the JSON path is disabled, and the verification command still checks for the disabled format.
- **`compounding work` defaults to editing your current branch in place**, with a tool allowlist that includes `python`. The worktree alternative calls a cleanup method that does not exist, so every isolated run ends in an `AttributeError`.
- **There is no LICENSE file**, though the manifest and README both claim MIT.

Take the architecture. Do not take the code.

## The extension: a learnings layer

Four additions, none of which modify an existing module's behaviour by default.

### E1 — A typed `Learning`, in the project's own idiom

The upstream stores untyped dictionaries with a free-text category. Kohärenz
Protokoll already runs closed `Literal` enums through `tools/kpwiki/schema.py`
and a deterministic lint. A learning should be a peer of `Claim`, not an
exception to it:

```python
LearningSource = Literal["decision-log", "lit-critic", "agent-memory", "ingest", "manual"]
LearningScope  = Literal["canon", "prose", "process", "tooling"]

class Learning(BaseModel):
    id: str                       # D-07, kap-06-finding-3, …
    source: LearningSource
    scope: LearningScope
    statement: str                # one sentence, imperative, German if it quotes canon
    rationale: str
    applies_to: list[str]         # chapter numbers, codex slugs, file paths
    citation: Citation            # reuse the existing model; a learning without a source does not exist
    supersedes: str = ""
    status: Literal["active", "superseded", "parked"] = "active"
```

`Citation` is reused deliberately: principle 3 of the knowledge-system concept
says a claim without a verifiable line does not exist, and a learning is a
claim about the project.

Storage: `Wiki/learnings/*.md` with this frontmatter. Not a new database. The
wiki is already a file tree with a free lint, and adding SQLite would introduce
a second source of truth — the exact failure the upstream's own README/code
divergence demonstrates.

### E2 — Codify from what already exists

Three importers, all read-only against their sources, all deterministic where
they can be:

| From | How |
|---|---|
| `decision-log*.md` | parse the existing table. Each D-xx row already has an ID, a decision, a rationale and affected chapters — the mapping is mechanical, no LLM needed |
| `lit-critic` reports | one learning per **rejected** finding, recording what was rejected and why, so it is not re-raised |
| `agent-memory/*.md` | one learning per bullet under the existing headed sections |

The decision-log importer is the highest-value and lowest-risk piece in this
plan: it is a parser, it needs no model, and it converts twenty-one already-made
author decisions into a retrievable corpus in one pass.

Nothing writes back to the decision log. It stays the author's document.

### E3 — Injection as an opt-in wrapper

Copy the upstream's wrapper shape, not its implementation:

```python
class WithLearnings(dspy.Module):
    """Wraps any kpwiki module; retrieves matching learnings and passes them
    through a declared input field. Off unless explicitly constructed."""

    def __init__(self, inner: dspy.Module, scopes: tuple[LearningScope, ...],
                 retrieve: LearningRetriever = no_learnings):
        ...
```

Two deliberate departures from the upstream:

- **A declared input field, not a string prepend.** The upstream mutates the longest string argument. A declared `learnings: list[Learning]` field keeps the signature honest, keeps the lint able to check it, and lets an optimizer see it as structure rather than as prose.
- **Default off.** `no_learnings` mirrors the existing `no_canon_retrieval` default in `programs.py`, so every current caller behaves exactly as it does today and each opt-in is one constructor argument.

Retrieval reuses whatever the canon retriever ends up being — see the sibling
plan for the `CanonRetriever` seam. There is no reason for two retrieval stacks.

### E4 — Close the loop at the gates

The lit-critic gate and `wiki_lint` are where a learning proves its worth:

- Before raising a finding, check whether an active learning already rejected it. If so, suppress it and cite the learning. That is the compounding effect made concrete: the third chapter's review is cheaper than the first's.
- After the author triages, write the new learnings back.

This is additive at both ends: a suppression list and an append. Neither
changes how a finding is computed.

## What this deliberately does not do

- **Never writes `Canon/`, `ncp*.json` or the graph.** Learnings are a Wiki-layer artifact. Promotion still requires a D-xx decision, unchanged.
- **Adds no service.** No SQLite, no vector database, no daemon. Files and the existing lint.
- **Does not touch the agency engine.** The provenance graph stays the system of record for canon; learnings are about the *process*, not about the world.
- **Does not auto-suppress on a model's judgment.** Suppression fires only on an explicit, cited, author-triaged learning.
- **Changes no existing default.** Every extension point is off until constructed with an argument.

## Sequence

| # | Step | Verification | Risk |
|---|---|---|---|
| 1 | `Learning` model plus schema entry and lint rules | `wiki_lint` accepts a valid learning and rejects a citation-less one | none: new file kinds only |
| 2 | Decision-log importer | all D-xx rows round-trip; the log itself is unmodified | low: a parser |
| 3 | lit-critic and agent-memory importers | spot-check against the source documents | low |
| 4 | `WithLearnings` wrapper, default off | existing programs byte-identical with the default | low |
| 5 | Opt in `SourceIngest` to process-scope learnings | ingest a source that D-07 bears on; the decision appears in context | medium: changes ingest output |
| 6 | Suppression in the lit-critic gate | a previously rejected finding is suppressed with a citation | medium: changes gate output |

Steps 1 and 2 are worth doing alone. They convert an existing, hand-maintained
decision log into a machine-readable corpus, which is useful even if no
injection is ever built.

## Measuring whether it compounds

The claim is that later work gets cheaper. State it as a measurement before
building, or the system will be believed rather than checked:

| Metric | Read it as |
|---|---|
| repeat findings per chapter review | should fall as learnings accumulate |
| suppression precision | a suppressed finding the author would have accepted is a regression, and the expensive error |
| decisions re-litigated | how often a question with a D-xx answer is asked again |
| learnings cited per ingest | zero means retrieval is not reaching the corpus |

Suppression precision is the one to watch. A system that hides real findings to
look efficient is worse than no system, and it will look like an improvement in
every other metric.

## Open questions for the author

1. Should a learning ever be German? The decisions are German because they quote canon; the surrounding engineering is English. The `statement` field probably mirrors the claim rule — quote in the source language, explain in English.
2. Should rejected lit-critic findings suppress silently, or always surface with the citation attached? Silent suppression is cheaper; visible suppression is auditable.
3. Do agent `MEMORY.md` files become generated views of the learnings corpus, or stay hand-maintained alongside it? Two writers on one file is the failure mode.
