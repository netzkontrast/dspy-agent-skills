"""dspy-deep-refine — runnable smoke test.

Constructs the DeepRefine loop as a DSPy module with an injected retriever and
exercises its deterministic parts offline: k-hop expansion over a toy triple
graph, the HIGH/MEDIUM/LOW evidence review, the apply gate (LOW refused), and
the metric. No LM call in --dry-run.

Usage:
    uv run python example_deep_refine.py --dry-run
    OPENAI_API_KEY=... uv run python example_deep_refine.py
"""

from __future__ import annotations

import argparse
import copy
import os
from typing import Literal

from pydantic import BaseModel, Field

Triple = tuple[str, str, str]
MAX_HOPS, CAPS, HORIZON, MAX_ACTIONS = 4, [5, 10, 15, 20], 4, 10
AMBIGUOUS = {"main", "main()", "run", "run()", "index", "home", "overview", "notes", "todo", "draft"}

TOY_GRAPH = {
    "nodes": [
        {"id": "kael", "label": "Kael", "aliases": ["System Kael"], "source": "characters.md"},
        {"id": "juna", "label": "Juna", "aliases": [], "source": "characters.md"},
        {"id": "kw1", "label": "KW1", "aliases": ["Kernwelt 1"], "source": "worlds.md"},
        {"id": "overview", "label": "Overview", "aliases": [], "source": "index.md"},
    ],
    "edges": [
        {"source": "kael", "target": "kw1", "relation": "lives_in"},
        {"source": "juna", "target": "kael", "relation": "part_of"},
    ],
}


class Abduction(BaseModel):
    incompleteness: list[str] = Field(default_factory=list)
    incorrectness: list[str] = Field(default_factory=list)
    redundancy: list[str] = Field(default_factory=list)


class RefinementAction(BaseModel):
    kind: Literal["insert_edge", "delete_edge", "replace_node"]
    args: list[str] = Field(min_length=2, max_length=3)


class ActionReview(BaseModel):
    action: RefinementAction
    confidence: Literal["HIGH", "MEDIUM", "LOW"]
    evidence: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def _label(node: dict) -> str:
    return node["label"]


def _matches(graph: dict, name: str) -> list[dict]:
    src, _, bare = name.rpartition("::")
    target = bare.casefold()
    out = []
    for n in graph["nodes"]:
        names = {n["id"], n["label"], *n.get("aliases", [])}
        if target in {x.casefold() for x in names} and (not src or n.get("source", "").endswith(src)):
            out.append(n)
    return out


def _edge_exists(graph: dict, subs: list[dict], rel: str, objs: list[dict]) -> bool:
    s_ids, o_ids = {n["id"] for n in subs}, {n["id"] for n in objs}
    return any(e["source"] in s_ids and e["target"] in o_ids and e["relation"] == rel for e in graph["edges"])


def review_action(graph: dict, action: RefinementAction) -> ActionReview:
    """Deterministic HIGH/MEDIUM/LOW grading (port of DeepRefine action_review)."""
    evidence, warnings = [], []
    entity_idx = [0, 2] if action.kind != "replace_node" else [0, 1]
    matches = {}
    for i in entity_idx:
        if i >= len(action.args):
            warnings.append(f"missing argument {i}")
            continue
        name = action.args[i]
        found = _matches(graph, name)
        matches[i] = found
        if "::" not in name and name.casefold() in AMBIGUOUS:
            warnings.append(f"Ambiguous bare node name: {name!r}; include a source path.")
        if len(found) > 1:
            warnings.append(f"Ambiguous node name: {name!r} matches {len(found)} nodes.")
        if found:
            evidence.append(f"Node exists: {name}")
    if action.kind in {"insert_edge", "delete_edge"} and len(action.args) == 3:
        subs, objs = matches.get(0, []), matches.get(2, [])
        if _edge_exists(graph, subs, action.args[1], objs):
            evidence.append("Exact edge already exists.")
        elif subs and objs:
            evidence.append("Both endpoint nodes exist; relation inferred by the loop.")
    if action.kind == "replace_node" and matches.get(0):
        evidence.append("Replacement source node exists.")
    if warnings:
        confidence = "LOW"
    elif any(e.startswith(("Exact edge", "Replacement source")) for e in evidence):
        confidence = "HIGH"
    elif evidence:
        confidence = "MEDIUM"
    else:
        confidence, warnings = "LOW", ["No node or edge evidence found."]
    return ActionReview(action=action, confidence=confidence, evidence=evidence, warnings=warnings)


def review_actions(graph: dict, actions: list[RefinementAction]) -> list[ActionReview]:
    return [review_action(graph, a) for a in actions]


def apply_actions(graph: dict, reviews: list[ActionReview], *, allow_low: bool = False) -> dict:
    """Return a new graph; LOW-confidence actions are refused unless allow_low."""
    new = copy.deepcopy(graph)
    for r in reviews:
        if r.confidence == "LOW" and not allow_low:
            continue
        a = r.action
        if a.kind == "insert_edge":
            s, rel, o = a.args
            sid = _ensure_node(new, s)
            oid = _ensure_node(new, o)
            if not any(e == {"source": sid, "target": oid, "relation": rel} for e in new["edges"]):
                new["edges"].append({"source": sid, "target": oid, "relation": rel})
        elif a.kind == "delete_edge":
            s, rel, o = a.args
            s_ids = {n["id"] for n in _matches(new, s)}
            o_ids = {n["id"] for n in _matches(new, o)}
            new["edges"] = [e for e in new["edges"]
                            if not (e["source"] in s_ids and e["target"] in o_ids and e["relation"] == rel)]
        else:
            old, target = a.args
            for n in _matches(new, old):
                n["label"] = target.rpartition("::")[2]
    return new


def _ensure_node(graph: dict, name: str) -> str:
    found = _matches(graph, name)
    if found:
        return found[0]["id"]
    bare = name.rpartition("::")[2]
    nid = "deeprefine_" + "".join(c if c.isalnum() else "_" for c in bare.casefold()).strip("_")
    graph["nodes"].append({"id": nid, "label": bare, "aliases": [], "source": "deeprefine"})
    return nid


def toy_retriever(question: str, hop: int, previous: list[Triple]) -> list[Triple]:
    """hop 0: label search; hop >= 1: 1-hop expansion over TOY_GRAPH."""
    by_id = {n["id"]: n for n in TOY_GRAPH["nodes"]}
    edges = [(by_id[e["source"]]["label"], e["relation"], by_id[e["target"]]["label"]) for e in TOY_GRAPH["edges"]]
    if hop == 0:
        return [t for t in edges if any(w.casefold() in question.casefold() for w in (t[0], t[2]))]
    seen = {x for t in previous for x in (t[0], t[2])}
    return [t for t in edges if t[0] in seen or t[2] in seen] + previous


def build(retrieve):
    import dspy

    class JudgeAnswerable(dspy.Signature):
        """Decide whether the question is answerable from the given triples alone."""

        question: str = dspy.InputField()
        triples: str = dspy.InputField()
        answerable: bool = dspy.OutputField()

    class AbduceErrors(dspy.Signature):
        """Explain why the question stayed unanswerable, along incompleteness,
        incorrectness and redundancy; cite triples; do not propose edits."""

        question: str = dspy.InputField()
        interaction_history: str = dspy.InputField()
        abduction: Abduction = dspy.OutputField()

    class ProposeRefinements(dspy.Signature):
        """Propose at most 10 minimal actions that make the question answerable;
        keep the base intact; use 'source::Name' for ambiguous names."""

        question: str = dspy.InputField()
        triples: str = dspy.InputField()
        abduction: Abduction = dspy.InputField()
        actions: list[RefinementAction] = dspy.OutputField()

    def fmt(triples: list[Triple]) -> str:
        return "\n".join(f"{s} | {r} | {o}" for s, r, o in triples)

    class DeepRefine(dspy.Module):
        def __init__(self, max_hops: int = MAX_HOPS):
            super().__init__()
            self.judge = dspy.ChainOfThought(JudgeAnswerable)
            self.abduce = dspy.ChainOfThought(AbduceErrors)
            self.propose = dspy.Predict(ProposeRefinements)
            self.max_hops = max_hops

        def forward(self, question: str):
            history, triples = [], []
            for step in range(1, self.max_hops + 1):
                triples = list(dict.fromkeys(retrieve(question, step - 1, triples)))[: CAPS[step - 1]]
                verdict = self.judge(question=question, triples=fmt(triples)).answerable
                history.append({"step": step, "num_hops": step - 1, "triples": triples, "answerable": verdict})
                if verdict:
                    break
            if len(history) <= 1:
                return dspy.Prediction(history=history, early_exit=True, abduction=None, actions=[])
            hist = "\n".join(f"step {h['step']} hops={h['num_hops']} answerable={h['answerable']}\n{fmt(h['triples'])}"
                             for h in history[-HORIZON:])
            abduction = self.abduce(question=question, interaction_history=hist).abduction
            actions = self.propose(question=question, triples=fmt(triples), abduction=abduction).actions[:MAX_ACTIONS]
            return dspy.Prediction(history=history, early_exit=False, abduction=abduction, actions=actions)

    def refine_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        if pred.early_exit:
            ok = bool(gold.answerable_at_hop0)
            return dspy.Prediction(score=1.0 if ok else 0.0,
                                   feedback="Early exit was correct." if ok else "Judged answerable at hop 0 but the base lacks the fact.")
        reviews = review_actions(gold.graph, pred.actions)
        low = [r for r in reviews if r.confidence == "LOW"]
        staged = apply_actions(gold.graph, reviews)
        by_id = {n["id"]: n["label"] for n in staged["nodes"]}
        present = {(by_id[e["source"]], e["relation"], by_id[e["target"]]) for e in staged["edges"]}
        answerable_now = all(tuple(t) in present for t in gold.expected_triples)
        parts = []
        if low:
            parts.append(f"{len(low)} LOW-confidence action(s): {low[0].warnings[0]}")
        if not answerable_now:
            parts.append("Expected triples still missing after applying non-LOW actions.")
        if len(pred.actions) > 5:
            parts.append("More than 5 actions; prefer the minimal edit set.")
        score = 0.6 * answerable_now + 0.3 * (1 - len(low) / max(1, len(reviews))) + 0.1 * (len(pred.actions) <= 5)
        return dspy.Prediction(score=score, feedback=" ".join(parts) or "Minimal, grounded actions; question now answerable.")

    return DeepRefine, refine_metric


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Skip all LM calls")
    ap.add_argument("--model", default=os.getenv("DSPY_MODEL", "openai/gpt-4o"))
    ap.add_argument("--question", default="Which Kernwelt does Juna belong to?")
    args = ap.parse_args()

    import dspy

    DeepRefine, refine_metric = build(toy_retriever)
    program = DeepRefine()

    if args.dry_run:
        hop0 = toy_retriever(args.question, 0, [])
        hop1 = toy_retriever(args.question, 1, hop0)
        assert len(hop1) > len(hop0), "retriever must widen with hops"
        actions = [
            RefinementAction(kind="insert_edge", args=["Juna", "lives_in", "KW1"]),          # MEDIUM
            RefinementAction(kind="insert_edge", args=["Kael", "lives_in", "KW1"]),          # HIGH (exists)
            RefinementAction(kind="replace_node", args=["overview", "index.md::Overview"]),  # LOW (ambiguous)
        ]
        reviews = review_actions(TOY_GRAPH, actions)
        labels = [r.confidence for r in reviews]
        assert labels == ["MEDIUM", "HIGH", "LOW"], labels
        staged = apply_actions(TOY_GRAPH, reviews)
        assert len(staged["edges"]) == 3 and len(TOY_GRAPH["edges"]) == 2, "apply must copy, and refuse LOW"
        gold = dspy.Example(question=args.question, graph=TOY_GRAPH, answerable_at_hop0=False,
                            expected_triples=[("Juna", "lives_in", "KW1")])
        pred = dspy.Prediction(history=[{}, {}], early_exit=False, abduction=Abduction(), actions=actions)
        metric = refine_metric(gold, pred)
        assert 0.6 < metric.score < 1.0 and "LOW-confidence" in metric.feedback
        names = [n for n, _ in program.named_predictors()]
        print("OK: DeepRefine constructed with predictors", names)
        print(f"    hop widening {len(hop0)} -> {len(hop1)} triples; review labels {labels}; LOW refused on apply")
        print(f"    metric score={metric.score:.2f} feedback={metric.feedback!r}")
        return 0

    dspy.configure(lm=dspy.LM(args.model), track_usage=True)
    pred = program(question=args.question)
    print("early_exit:", pred.early_exit)
    for r in review_actions(TOY_GRAPH, pred.actions):
        print(f"  [{r.confidence}] {r.action.kind}{tuple(r.action.args)}  {'; '.join(r.evidence + r.warnings)}")
    print("Nothing was applied. Approve explicitly to apply non-LOW actions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
