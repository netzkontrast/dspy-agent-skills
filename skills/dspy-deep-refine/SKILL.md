---
name: dspy-deep-refine
description: Refine the knowledge base a DSPy program retrieves from, using the DeepRefine loop as a typed DSPy module — judge answerability, expand retrieval hop by hop, abduce why a question stayed unanswerable (incompleteness, incorrectness, redundancy), propose bounded graph or wiki refinement actions, grade each action's evidence HIGH/MEDIUM/LOW deterministically, and stop for approval before any write. The loop's own metric (does the refined base answer the question?) makes it GEPA-optimizable, so refinement is part of self-optimization, not a manual chore.
when_to_use: >-
  User says "deep refine", "refine the wiki/graph/knowledge base", "why can't
  it answer this", "the retriever keeps missing", or wants a RAG/wiki program
  that improves its own knowledge base from failed queries; or a query loop
  where unanswerable questions should turn into reviewed edits with an audit trace.
---

# DSPy Deep Refine (3.3.x)

Port of DeepRefine (HKUST-KnowComp, arXiv:2605.10488; agent adapter
`DeepRefine-Skill`) into DSPy. DeepRefine treats a question the knowledge base
cannot answer as a *defect in the base*, not in the prompt: it retrieves wider,
judges again, abduces the error, proposes minimal edits, and applies them only
after evidence review and explicit approval. Here every LLM step is a typed
Signature, the evidence review and the trace validation stay deterministic, and
the whole loop carries a metric — so `dspy.GEPA` can optimize how the program
abduces and refines. That is the self-optimizing part: at runtime the base gets
better, at compile time the refiner gets better.

## Loop → DSPy construct

| DeepRefine step | Prose/CLI original | DSPy construct | Deterministic |
|---|---|---|---|
| 1 Judge | `<judge>Yes/No</judge>` | `JudgeAnswerable` → `answerable: bool` | — |
| 2 Expand | k-hop expansion, caps `[5, 10, 15, 20]` | retriever callable injected into the module | hop loop, caps, dedupe |
| 3 Abduce | `<abduction>…</abduction>` | `AbduceErrors` → `Abduction` (three axes) | required iff `len(history) > 1` |
| 4 Propose | `<refinement>insert_edge(...)\|…</refinement>` | `ProposeRefinements` → `list[RefinementAction]` (≤ 10) | count cap, closed action enum |
| 5 Review | HIGH / MEDIUM / LOW evidence labels | `review_actions(graph, actions)` | fully deterministic |
| 6 Apply | `deeprefine apply` after approval | `apply_actions(graph, actions, allow_low=False)` — **never called by the module** | LOW refused by default |
| Trace | `loop_trace_<id>.json` | `LoopTrace` model, `validate_trace()` | schema + control-flow checks |

## Canonical program

```python
import dspy
from pydantic import BaseModel, Field
from typing import Callable, Literal

Triple = tuple[str, str, str]                      # (subject, relation, object)
Retriever = Callable[[str, int, list[Triple]], list[Triple]]   # (question, hop, previous) -> triples
MAX_HOPS, CAPS, HORIZON = 4, [5, 10, 15, 20], 4

class JudgeAnswerable(dspy.Signature):
    """Decide whether the question is answerable from the given triples alone."""
    question: str = dspy.InputField()
    triples: str = dspy.InputField(desc="one 'subject | relation | object' per line")
    answerable: bool = dspy.OutputField()

class Abduction(BaseModel):
    incompleteness: list[str] = Field(default_factory=list, description="facts or links the base lacks")
    incorrectness: list[str] = Field(default_factory=list, description="wrong or conflicting triples")
    redundancy: list[str] = Field(default_factory=list, description="duplicates that confuse retrieval")

class AbduceErrors(dspy.Signature):
    """Explain why the question stayed unanswerable across the retrieval steps, along
    the three DeepRefine axes. Cite the triples or gaps you mean; do not propose edits."""
    question: str = dspy.InputField()
    interaction_history: str = dspy.InputField(desc="last steps: hop, triples, judgement")
    abduction: Abduction = dspy.OutputField()

class RefinementAction(BaseModel):
    kind: Literal["insert_edge", "delete_edge", "replace_node"]
    args: list[str] = Field(min_length=2, max_length=3)

class ProposeRefinements(dspy.Signature):
    """Propose at most 10 minimal actions that make the question answerable. Keep the
    original base as intact as possible; never delete unrelated triples; use
    source-qualified node names ('file.md::Name') whenever a bare name is ambiguous."""
    question: str = dspy.InputField()
    triples: str = dspy.InputField()
    abduction: Abduction = dspy.InputField()
    actions: list[RefinementAction] = dspy.OutputField()

class DeepRefine(dspy.Module):
    def __init__(self, retrieve: Retriever, max_hops: int = MAX_HOPS):
        super().__init__()
        self.judge = dspy.ChainOfThought(JudgeAnswerable)
        self.abduce = dspy.ChainOfThought(AbduceErrors)
        self.propose = dspy.Predict(ProposeRefinements)
        self.retrieve, self.max_hops = retrieve, max_hops

    def forward(self, question: str) -> dspy.Prediction:
        history, triples = [], []
        for step in range(1, self.max_hops + 1):
            triples = dedupe(self.retrieve(question, step - 1, triples))[: CAPS[step - 1]]
            verdict = self.judge(question=question, triples=fmt(triples)).answerable
            history.append({"step": step, "num_hops": step - 1, "triples": triples, "answerable": verdict})
            if verdict:
                break
        if len(history) <= 1:                                   # early exit: answerable at hop 0
            return dspy.Prediction(history=history, early_exit=True, abduction=None, actions=[])
        abduction = self.abduce(question=question, interaction_history=fmt_history(history[-HORIZON:])).abduction
        actions = self.propose(question=question, triples=fmt(triples), abduction=abduction).actions[:10]
        return dspy.Prediction(history=history, early_exit=False, abduction=abduction, actions=actions)
```

The module **returns** actions; it never touches the base. Review and apply are
separate functions with their own gate (below).

## Evidence review and the apply gate

```python
reviews = review_actions(graph, pred.actions)        # HIGH | MEDIUM | LOW per action, with evidence + warnings
print(render(reviews))                                # show the user; STOP here in a normal run
# only after the user's *next* message explicitly approves:
new_graph = apply_actions(graph, [r for r in reviews if r.confidence != "LOW"])
```

Labels (deterministic, from DeepRefine's `action_review`):

| Label | When |
|---|---|
| HIGH | the exact edge already exists, or the node/edge is backed by direct source evidence (`[[wikilink]]` in the page, definition in the file) |
| MEDIUM | both endpoint nodes exist; the relation is inferred by the loop |
| LOW | a bare ambiguous name (`main()`, `index`, `overview`, `notes`…), a name matching several nodes, or no node evidence at all |

`apply_actions` refuses LOW by default; `allow_low=True` only when the user's
approval message explicitly accepts that risk. A generated action list, a valid
trace, or a successful review is **not** approval.

## The metric that makes refinement self-optimizing

```python
def refine_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    if pred.early_exit:
        return dspy.Prediction(score=1.0 if gold.answerable_at_hop0 else 0.0,
                               feedback="Early exit was correct." if gold.answerable_at_hop0
                               else "Judged answerable at hop 0 but the gold says the base lacks the fact.")
    reviews = review_actions(gold.graph, pred.actions)
    low = [r for r in reviews if r.confidence == "LOW"]
    staged = apply_actions(gold.graph, [r for r in reviews if r.confidence != "LOW"])   # on a copy
    answerable_now = judge_offline(gold.question, staged, gold.expected_triples)    # deterministic: expected triples present
    parts = []
    if low:
        parts.append(f"{len(low)} LOW-confidence actions ({low[0].warnings[0]}); qualify node names with their source.")
    if not answerable_now:
        parts.append("After applying the non-LOW actions the expected triples are still missing; the abduction named the wrong axis or the actions are too indirect.")
    if len(pred.actions) > 5:
        parts.append("More than 5 actions for one question; prefer the minimal edit set.")
    score = 0.6 * float(answerable_now) + 0.3 * (1 - len(low) / max(1, len(reviews))) + 0.1 * float(len(pred.actions) <= 5)
    return dspy.Prediction(score=score, feedback=" ".join(parts) or "Minimal, well-grounded actions that make the question answerable.")
```

Gold examples are (question, graph snapshot, expected triples, answerable-at-hop-0);
20–40 of them from real failed queries are enough for GEPA `auto="light"`. GEPA
then rewrites the `abduce` and `propose` instructions against this metric —
refinement quality improves without touching the prompts by hand.

## Trace

Every run writes a `LoopTrace` (schema v1: `query`, `constants`,
`interaction_history[]` with `step/num_hops/triples/answerable/retrieval`,
`abduction`, `actions`, `early_exit`). `validate_trace()` enforces the control
flow: judgement present on every hop, hops stop on the first `True` or at
`max_hops`, abduction present iff `len(history) > 1`, actions only after
abduction, ≤ 10 actions. Persist it next to the base so a later reviewer can
see which query exposed the gap and why the edit was made.

## Anti-patterns

- Applying inside `forward()` — the module proposes; humans (or an explicitly configured auto-apply for HIGH-only actions) apply.
- Skipping abduction because "the actions are obvious" — abduction is what the metric and GEPA reason about.
- Free-text actions — the closed `kind` enum plus argument count is what keeps `apply_actions` safe.
- Refining only the latest question — process the pending queue of unanswered queries first, deduplicated by query id.
- Judging with the same triples every hop — the retriever must widen (`num_hops` grows); assert the retrieved set changes or stop early.
- Treating the review as approval.

## Where to go next

- The metric contract → `dspy-evaluation-harness`
- Optimizing `abduce`/`propose` → `dspy-gepa-optimizer`
- Using the loop inside a context-heavy pipeline → `dspy-rlm-workflow`
- Full reference (models, trace schema, review rules, apply semantics) → [reference.md](reference.md)
- Runnable example → [example_deep_refine.py](example_deep_refine.py)
