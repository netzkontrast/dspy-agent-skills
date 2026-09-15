# DSPy Deep Refine — Reference

Sources: DeepRefine (HKUST-KnowComp, arXiv:2605.10488) and its agent adapter
`DeepRefine-Skill` v0.2.0 (`agent_prompts.py`, `action_review.py`,
`agent_loop.py::validate_trace`, `references/deeprefine-workflow.md`),
mapped onto DSPy 3.2.1.

## Constants (from DeepRefine)

```text
MAX_HOPS = 4                  # retrieval steps per question
INCREMENT_HOP = 1             # num_hops = (step - 1) * INCREMENT_HOP
BASE_TOP_K = 10
MAX_TRIPLE_NUM_BY_STEP = [5, 10, 15, 20]   # cap on triples shown to the judge per step
HISTORY_HORIZON = 4           # last N steps passed to abduction
MAX_ACTIONS = 10
```

## Models

```python
from pydantic import BaseModel, Field
from typing import Literal

class Step(BaseModel):
    step: int
    num_hops: int
    base_top_k: int = 10
    query: str
    retrieval_method: str                    # "search", "k_hop_expansion", "search+k_hop_expansion"
    retrieved_subgraph: list[tuple[str, str, str]]
    answerable: bool

class Abduction(BaseModel):
    incompleteness: list[str] = Field(default_factory=list)
    incorrectness: list[str] = Field(default_factory=list)
    redundancy: list[str] = Field(default_factory=list)

class RefinementAction(BaseModel):
    kind: Literal["insert_edge", "delete_edge", "replace_node"]
    args: list[str] = Field(min_length=2, max_length=3)
    # insert_edge/delete_edge: [subject, relation, object]; replace_node: [old, new]

class ActionReview(BaseModel):
    action: RefinementAction
    confidence: Literal["HIGH", "MEDIUM", "LOW"]
    evidence: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    suggested_replacement: RefinementAction | None = None

class LoopTrace(BaseModel):
    schema_version: int = 1
    query: str
    query_id: str                            # sha1(query)[:16]
    constants: dict
    interaction_history: list[Step]
    abduction: Abduction | None = None
    actions: list[RefinementAction] = Field(default_factory=list)
    early_exit: bool = False
```

## Signatures (instructions carry the DeepRefine prompt intent)

The three original prompts (judgement, error abduction, KG refinement) become
Signature docstrings; the tagged outputs (`<judge>`, `<abduction>`,
`<refinement>`) become typed fields, so no tag parsing exists anywhere.

| Signature | Inputs | Outputs | Original rule kept |
|---|---|---|---|
| `JudgeAnswerable` | `question`, `triples` | `answerable: bool` | "think carefully about the question and the KG context before judging" |
| `AbduceErrors` | `question`, `interaction_history` | `abduction: Abduction` | three perspectives: incompleteness, incorrectness, redundancy |
| `ProposeRefinements` | `question`, `triples`, `abduction` | `actions: list[RefinementAction]` | ≤ 10 actions; "do not delete irrelevant triples; keep the original KG as much as possible" |

## Control flow (must match `DeepRefine.refine()`)

```text
history = []
for step in 1..MAX_HOPS:
    triples = retrieve(question, hop=step-1, previous=triples)   # step 1: search; later: k-hop expansion
    triples = dedupe(triples)[: MAX_TRIPLE_NUM_BY_STEP[step-1]]
    answerable = judge(question, triples)
    history.append(Step(...))
    if answerable: break
if len(history) <= 1:  -> early_exit = True; no abduction, no actions
else:
    abduction = abduce(question, history[-HISTORY_HORIZON:])
    actions   = propose(question, last_triples, abduction)[:MAX_ACTIONS]
return Prediction(history, early_exit, abduction, actions)      # NO WRITE
```

Critical rule (verbatim from the adapter): refinement runs when
`len(interaction_history) > 1`, not only when every judgement was `False`. A
question answered at hop 2 still exposed a retrieval gap worth an edit.

### Retriever contract

`retrieve(question: str, hop: int, previous: list[Triple]) -> list[Triple]`

- `hop == 0`: lexical/semantic search over the base (BM25, FTS, embeddings).
- `hop >= 1`: expand 1-hop neighbours of the subjects/objects in `previous`
  (`k_hop_expansion`), optionally unioned with a fresh search.
- Must widen: if `set(new) == set(previous)` the module may stop early and
  record `retrieval_method = "exhausted"`.

## Evidence review (deterministic)

Port of `action_review.review_action`, simplified to the graph shape
`{"nodes": [{"id", "label", "aliases", "source"}], "edges": [{"source", "target", "relation"}]}`:

1. Parse the action; malformed → LOW with the parse error.
2. For each entity argument: match nodes by label/alias/id (case-folded);
   `"src::Name"` qualifies by source. Record `Node exists: …` evidence.
3. Warnings (each forces LOW):
   - bare name in `AMBIGUOUS_LABELS` = `{main, main(), run, run(), train, train(), test, test(), setup, setup(), untitled, new_page, draft, index, home, introduction, overview, notes, todo}`;
   - a name matching more than one node → suggest the qualified candidates;
   - no node evidence at all.
4. `insert_edge` / `delete_edge`: exact edge present → HIGH evidence; both
   endpoints present → MEDIUM evidence. Direct source evidence (a `[[Name]]`
   wikilink in the subject's page for wiki bases; a `def`/`class`/call for code
   bases) → HIGH.
5. `replace_node`: source node exists → HIGH evidence; target existing is noted.
6. `suggested_replacement`: the same action with ambiguous names replaced by the
   first qualified candidate.

Confidence: any warning → LOW; else HIGH evidence present → HIGH; else any
evidence → MEDIUM; else LOW.

## Apply semantics

```python
def apply_actions(graph, actions, *, allow_low=False) -> dict:
    """Return a NEW graph; refuse LOW actions unless allow_low; never mutate input."""
```

- `insert_edge(s, r, o)`: create missing nodes (tagged `contributor="deeprefine"`), add the edge if absent.
- `delete_edge(s, r, o)`: remove only the exact edge; unknown → no-op with a warning.
- `replace_node(old, new)`: relabel; merge into an existing `new` node if present, rewiring edges.
- Stage → validate → swap: apply on a copy, regenerate derived views (index, wiki
  pages) on the copy, validate them, then replace production atomically. If any
  step fails, production is untouched (DeepRefine `wiki_refresh`).
- Keep a backup of the pre-state and a checkpoint of the post-state per run.

## Queue discipline

Refine the **pending queue** first: every past query with `refined != true`,
deduplicated by `query_id = sha1(query)[:16]`, in first-seen order; fall back to
the current question only when the queue is empty. Mark `refined = true` only
after review (and apply, when approved) completed.

## `validate_trace` checks

| Check | Rule |
|---|---|
| judgement per hop | every `Step` has `answerable` set |
| stop condition | steps end at the first `answerable=True` or at `max_hops` |
| hop arithmetic | `num_hops == (step - 1) * INCREMENT_HOP` |
| caps | `len(retrieved_subgraph) <= MAX_TRIPLE_NUM_BY_STEP[step-1]` |
| early exit | `early_exit == (len(history) <= 1)`; then no abduction, no actions |
| refinement path | abduction present and non-empty on at least one axis; actions only with abduction; `len(actions) <= 10` |

## GEPA on the refiner

```python
optimizer = dspy.GEPA(metric=refine_metric, auto="light",
                      reflection_lm=dspy.LM("openai/gpt-5", temperature=1.0, max_tokens=32000),
                      track_stats=True, log_dir="./gepa_logs")
optimized = optimizer.compile(student=DeepRefine(retrieve), trainset=trainset, valset=valset)
```

Gold example fields: `question`, `graph` (snapshot dict), `expected_triples`
(what a correct refinement must make retrievable), `answerable_at_hop0` (bool).
The metric applies the non-LOW actions on a copy and checks the expected
triples deterministically; the judge model is not in the metric loop, so
optimization cannot reward "judge everything answerable".

## Failure modes

| Symptom | Cause | Fix |
|---|---|---|
| every action LOW | bare names | the `propose` instruction must demand `source::Name`; GEPA learns it from the metric feedback |
| loop always early-exits | judge too lenient | add `answerable_at_hop0 == False` golds; the metric penalizes wrong early exits |
| actions delete unrelated triples | instruction drift | Tier check in the metric: deleted edges must appear in `abduction.incorrectness` or `redundancy` |
| retrieval never widens | retriever ignores `hop` | implement k-hop expansion; assert the set grows |
| apply corrupts derived views | no staging | stage → validate → swap; keep backups |
