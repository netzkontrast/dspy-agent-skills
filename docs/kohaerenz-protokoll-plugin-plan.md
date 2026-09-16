# Making this pack the plugin repo for Kohärenz Protokoll

A plan, not a change. Nothing here is implemented in Kohärenz Protokoll (KP)
yet; the only code this document ships is the seam scaffold in
[`scaffolding/kp_canon_retriever.py`](../scaffolding/kp_canon_retriever.py),
which is inert until KP imports it.

## Why these two repos already fit

KP's `requirements-dspy.txt` says, in its own comment, that its DSPy version is
*"pinned to the DSPy release the dspy-agent-skills pack is validated against"*.
The dependency already exists informally. This plan makes it explicit and
useful in both directions: KP consumes skills, and this pack carries the
integration knowledge KP needs.

KP's DSPy layer (`tools/kpwiki/`, 523 lines) is already built the way these
skills teach: typed `dspy.Signature` classes with closed `Literal` enums, no
prompt strings, and metrics returning `dspy.Prediction(score, feedback)` so
GEPA can consume them. There is nothing to retrofit.

## The seam that is already waiting

`tools/kpwiki/programs.py` defines the integration point and leaves it empty on
purpose:

```python
CanonRetriever = Callable[[list[Claim]], str]

def no_canon_retrieval(_claims: list[Claim]) -> str:
    """Retriever used in dry runs: nothing retrieved, so nothing can conflict."""
    return ""

class SourceIngest(dspy.Module):
    def __init__(self, retrieve_canon: CanonRetriever = no_canon_retrieval):
```

Its module docstring states the intended evolution: *"BM25 first, graph
`match_codex_entries` later."* Everything in Part 3 below fills that callable.
Because it is injected, each step is testable offline and reversible by
swapping the argument back.

## Part 1 — What this repo has to become

| Item | State | Action |
|---|---|---|
| `.claude-plugin/plugin.json` + `marketplace.json` | present, v0.7.0 | none |
| Skills KP needs | present | none |
| DSPy version contract | implicit in a comment | state it in both READMEs and keep the pins equal |
| Integration knowledge for KP's sources | this document plus six new skills | keep current as the upstreams move |
| Scaffolding KP can import | `scaffolding/` | grow only when KP asks |

KP installs it like any plugin:

```bash
/plugin marketplace add netzkontrast/dspy-agent-skills
/plugin install dspy-agent-skills@dspy-agent-skills
```

Skills KP's CLAUDE.md should name, beyond the five it already lists:

| Skill | KP use |
|---|---|
| `dspy-retrieval` | the `CanonRetriever` seam; recall@k separate from answer quality |
| `dspy-optimizer-selection` | choosing an optimizer for `ingest_metric`, instead of defaulting to GEPA |
| `dspy-production` | pinning, caching and tracing the ingest runs |
| `dspy-drg-kg` | Sources to codex-graph extraction |
| `dspy-autodialectics` | gate for generated canon; complements `lit_critic_gate.py` |
| `dspy-clarify` | the promotion boundary Wiki → Canon, where a D-xx decision is required |

**The version contract is the load-bearing part.** KP pins `dspy==3.2.1`
(`requirements-dspy.txt`, commented "Pinned to the DSPy release the
dspy-agent-skills pack is validated against"). That comment is now stale:
**this pack moved first.** It validates against 3.3.1 and its floor is
`dspy>=3.3.0`, because DSPy 3.3.0 renamed `dspy.RLM`'s `max_iterations` to
`max_iters` and swapped `interpreter=` for `interpreter_factory=`.

The bump is safe on KP's side, and that is a checked claim rather than an
assumption. `tools/kpwiki/` uses exactly twelve DSPy symbols — `ChainOfThought`,
`Evaluate`, `Example`, `InputField`, `LM`, `Module`, `OutputField`, `Predict`,
`Prediction`, `Signature`, `configure`, `context` — and none of them changed
between 3.2.1 and 3.3.1. KP touches no RLM, `ProgramOfThought`, `CodeAct` or
`dspy.Image` call site, which is where every breaking rename landed. So KP can
move its pin to `dspy==3.3.1` and re-run `scripts/setup_dspy.sh --check`
without editing a program.

That pin is KP's to change, not this pack's, so nothing in the KP repo was
touched here. What this section now owes KP is the notice the contract asks
for, and this is it.

## Part 2 — Port verdicts, per upstream

Read the matching skill before acting on any row.

| Upstream | Verdict | What actually moves |
|---|---|---|
| **dspy-refrag** | **partial — one file** | `sensor_advanced.py` (MMR and adaptive selection), vendored under MIT with attribution. Nothing else. |
| **drg-kg** | **adopt as a dependency** | No code copied. `pip install "drg-kg[dspy]"` and a KP-specific schema. |
| **self-corrective-rag (TARA)** | **port the pattern, not the code** | The 4D context score and progressive-leniency retry, reimplemented in `tools/kpwiki/`. |
| **dspy-rlm-hooks** | **adopt as a dependency, later** | `pip install dspy-rlm-hooks` when KP has an RLM step. No code copied. |
| **dspytools** | **do not port** | Overlaps the agency engine KP already runs, and adds FalkorDB and Redis. |
| **context-engineering-dspy-book** | **do not port** | Reference material. Cite notebooks in review comments. |

Three rows deserve their reasons stated, because they are the ones someone
would otherwise get wrong.

**dspy-refrag is mostly not worth taking.** Its fragment selection never
reduces the prompt — `forward` joins every passage and merely annotates
`(selected: bool)` — so adopting it for context compression would deliver
nothing measurable. Its FAISS and Pinecone backends raise
`NotImplementedError`. Importing it drags in psycopg2. Its Weaviate pin
contradicts its own Weaviate code. One file, `sensor_advanced.py`, is genuinely
good and self-contained: MMR selection is exactly what KP needs when the top-k
canon passages are near-duplicates of each other, which they are, because the
codex is dense with related entries. Take that file. Leave the package.

**dspytools would fight the agency engine.** KP already has a provenance graph
in `.agency/session.db`, capability verbs that auto-record invocations, and
skill walking. dspytools brings its own skill graph, its own registry and a
FalkorDB service. Two systems of record for the same concern is the failure
this pack's own consolidation rule exists to prevent.

**TARA's code is unlicensed.** Its README claims MIT and links a `LICENSE`
file that is not in the repository. The 4D scoring idea is public in the paper
and cheap to reimplement; the repository is not safe to copy from until the
authors fix that. Reimplement, do not vendor.

## Part 3 — The four concrete integrations

### 3.1 Fill `CanonRetriever` (highest value, lowest risk)

Today `SourceIngest` runs with `no_canon_retrieval`, so `CheckCanonConflict`
never fires and every ingest reports zero conflicts. That is the single
biggest correctness gap in the ingest loop: it cannot contradict canon it never
retrieved.

```python
retrieve = CanonIndex.load("Plan/wiki/index/canon")      # scaffolding/
ingest = SourceIngest(retrieve_canon=retrieve)
```

Build it with `dspy.Embeddings` over `Canon/**/*.md` plus the rendered
`Codex/` views, chunked per section. Select with MMR from the vendored sensor,
not by raw top-k, because near-duplicate codex entries otherwise fill the
context with the same fact four times.

Sequence: build the index, measure recall@k against a hand-built devset of
claims whose canon location is known, then enable it. `dspy-retrieval` gives
the recall-versus-answer diagnosis table that tells you whether a bad conflict
check is a retrieval problem or a signature problem.

Reversible at any point by passing `no_canon_retrieval` again.

### 3.2 Sources to codex graph with drg-kg

KP's codex already is a typed graph: entries with a closed `kind` enum
(`concept`, `location`, `faction`, `artefact`, `minor-character`), WorldAxioms
with `severity ∈ {hard, soft}`, StoryTimeEvents, and typed edges. DRG's
`EnhancedDRGSchema` expresses exactly that shape, so the schema is a
transcription of rules KP has already decided, not a new ontology.

The value is not replacing the agency verbs that write the graph. It is the
**proposal** step: read a Source, propose typed entities and relations against
KP's schema, and hand them to the existing `/ingest` path for human decision.
DRG's `evidence_for` and `explain` give the provenance that a D-xx decision
needs.

Two rules that are not optional here:

- Set `DRG_REQUIRE_LM=1`. Without it, a missing key returns an empty graph and a green run. A canon pipeline that silently ingests nothing is worse than one that fails.
- Keep `enable_implicit_relationships` off for canon work. It adds LLM-inferred edges, and inferred edges must not enter canon without passing the same gate as any other claim.

Nothing DRG proposes may be written to `Canon/` or the graph without the D-xx
decision the existing rules require. It proposes; the author decides.

### 3.3 A 4D score for canon conflicts

`CheckCanonConflict` currently receives whatever the retriever returned and has
no way to say *the passages are relevant but insufficient*. TARA's four
dimensions solve exactly that, and each maps to a repair KP can actually make:

| Dimension | Failing it means | Repair |
|---|---|---|
| relevance | wrong canon passages | reformulate the query from the claim entities |
| coverage | right area, missing entries | retrieve per entity rather than per claim |
| specificity | too general to adjudicate | pull the codex entry, not the chapter |
| sufficiency | cannot decide from this | raise an OpenQuestion instead of guessing |

That last row matters most: KP already has `RaiseQuestions` and an explicit
Rule 0 that says ask rather than assume. A sufficiency score below threshold is
the machine-checkable trigger for it.

Adopt the dimensions. **Do not adopt the progressive leniency.** TARA lowers
its threshold on each retry with a floor of 20, so a context scoring 27 out of
100 is accepted at retry 3. For a canon gate, the correct terminal state is an
OpenQuestion, not a lowered bar.

### 3.4 RLM hooks for the knowledge fence

Only when KP adds an RLM step. KP's scene-writing loop has a hard constraint —
a character may only know what they have learned by that scene
(`what_does_X_know_as_of`) — and `PreIterationOutput.prompt_context` is
injected into the model's view without being executed. That is the right shape
for a fence: the knowledge state is stated per iteration, and
`post_iteration_hook` can stop the run when a violation appears.

Deferred because it patches private DSPy internals. Adopt it when there is an
RLM step to instrument, and pin DSPy and the hooks package together.

## Part 4 — Sequence

Each step is independently valuable and independently revertible.

| # | Step | Verification | Blocks |
|---|---|---|---|
| 1 | Declare the plugin dependency and equalize the DSPy pins | both READMEs name it; pins match | nothing |
| 2 | Vendor `sensor_advanced.py` into `tools/kpwiki/selection.py` with attribution | MMR beats plain top-k on a near-duplicate fixture | 3 |
| 3 | Build the canon index and fill `CanonRetriever` | recall@k on a known-location devset | 4, 5 |
| 4 | Add the 4D context score to the conflict check | conflicts found on a seeded contradiction; sufficiency routes to OpenQuestion | — |
| 5 | Optimize `SourceIngest` against `ingest_metric` | compiled beats baseline on a held-out set | — |
| 6 | DRG schema for the codex, proposal-only | proposed entities validate against the `kind` enum in force (see D2) | — |
| 7 | RLM hooks, if and when an RLM step exists | fence violation stops the run | — |

Step 3 is where the real gain is. Steps 1 and 2 exist to make it safe.

## Part 5 — What not to do

- Do not let any of this write to `Canon/` or the provenance graph without a D-xx decision. These tools propose.
- Do not widen the codex `kind` enum from inside KP. D2 approves the change, but the enum is the agency engine's; KP keeps the `**Kategorie:**` bridge until the engine ships it.
- Do not adopt `dspy-refrag` as a package for context compression; measure first and you will find nothing to measure.
- Do not copy TARA source while its license file is missing.
- Do not introduce FalkorDB, Redis or a second skill graph alongside the agency engine.
- Do not move either DSPy pin unilaterally.
- Do not translate canon prose. These are English engineering tools operating on German canon; claims quote the source language, which `metrics.py` already enforces with `language_kept`.
- Do not skip the recall measurement in step 3. An unmeasured retriever that returns plausible passages will make the conflict check look like it is working.

## Author decisions

The three questions this plan opened are answered. They are recorded here
because each one changes what gets built, not just how it is described.

### D1 — The canon index covers `Canon/` and the graph only

`Manuscript/` prose stays out of the index. The conflict check fires against
author-locked truth exclusively: `Canon/*.md`, WorldAxioms, CodexEntries.

The reason to want draft prose in the index is real — it would catch
continuity drift no codex entry records. It loses to a worse failure. A
chapter that is drafted but not revised is not yet true, and indexing it
makes the retriever able to return a draft's own error as the canon a later
draft is checked against. The error then reads as confirmed. Keeping the
index author-locked means a retrieved passage is always something a D-xx
decision put there.

Consequence for step 3: the recall@k devset is built from `Canon/` and codex
locations only, so it can be scored without any chapter reaching `revised`.

### D2 — The codex `kind` enum grows in the engine

Proposals no longer have to squeeze into
`{concept, location, faction, artefact, minor-character}` with the real
category demoted to a `**Kategorie:**` body line. `rule`, `motif`, `theme`,
`voice` and `character` become real enum members, and DRG extraction targets
them directly.

This is an engine-ontology change and it is the largest item in this plan, so
treat it as its own step rather than a detail of step 6:

- It is an **agency engine** change, not a KP change. The `kind` enum is
  enforced by the capability, so KP cannot widen it unilaterally.
- The ~600 existing entries carry their true category in the body's first
  line. Migration is mechanical — read `**Kategorie:**`, set `kind`, drop the
  line — but it must be idempotent against graph ground truth, per the
  partial-block persistence gotcha, not against a ledger of what was done.
- Until the engine ships the wider enum, DRG proposals keep using the body
  line. The workaround is the bridge, not the destination.
- Anything that parses the `**Kategorie:**` convention — including
  `scripts/render_codex_views.py` — has to read the field after migration and
  the body line before it, so it must handle both while the migration is in
  flight.

### D3 — Three LM roles, and KP already has them

The ingest loop uses a cheap extractor, a strong independent judge, and a
separate reflection model for GEPA. No new configuration is needed:
`tools/kpwiki/lm.py` already defines exactly this split, overridable per role
through the environment.

| Plan role | KP role in `lm.py` | Default | Temperature |
|---|---|---|---|
| extractor | `worker` | `anthropic/claude-haiku-4-5` | 0.0 |
| judge | `task` | `anthropic/claude-opus-5` | 0.0 |
| reflection | `reflection` | `anthropic/claude-opus-5` | 1.0 |

The separation is the point, not the model choice. The judge must not be the
model that produced the extraction it is judging, and GEPA's reflection model
must not be the judge it is optimising against — otherwise optimisation moves
the program toward the judge's own errors, which is the failure
`dspy-book-metrics` documents from chapter 5. The chapter 6 finding applies to
the extractor slot specifically: the expensive model is not reliably the
better one, so `worker` is the slot to measure before paying to upgrade.
