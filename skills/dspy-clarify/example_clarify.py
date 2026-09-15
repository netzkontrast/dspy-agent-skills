"""dspy-clarify — runnable smoke test.

Builds the clarify gate: a Signature that rewrites a claim so its scope,
terms and assumptions are explicit without adding meaning, listing every
unresolved ambiguity as a question for the human instead of guessing. The
dry run exercises the deterministic metric on a good and a bad clarification
without any LM call.

Usage:
    uv run python example_clarify.py --dry-run
    OPENAI_API_KEY=... uv run python example_clarify.py
"""

from __future__ import annotations

import argparse
import os
import re
from typing import Literal

from pydantic import BaseModel, Field

HEDGES = ("irgendwie", "meist", "meistens", "wohl", "vielleicht", "ungefähr", "manchmal",
          "eigentlich", "somehow", "probably", "maybe", "roughly", "sometimes", "kind of", "sort of")
QUANTIFIERS = ("alle", "jede", "jeder", "jedes", "immer", "nie", "niemals", "kein", "keine", "nur",
               "all", "every", "always", "never", "only")
UNSPECIFIED = "unspecified"


class Ambiguity(BaseModel):
    phrase: str = Field(description="the ambiguous span, verbatim")
    readings: list[str] = Field(min_length=2, description="the distinct readings")
    question: str = Field(description="what the author must decide; ends with '?'")


class Binding(BaseModel):
    mention: str
    slug: str


class Scope(BaseModel):
    world: str = UNSPECIFIED
    act: str = UNSPECIFIED
    part: str = UNSPECIFIED


class Clarification(BaseModel):
    clarified_text: str
    scope: Scope = Field(default_factory=Scope)
    assumptions: list[str] = Field(default_factory=list)
    ambiguities: list[Ambiguity] = Field(default_factory=list)
    bindings: list[Binding] = Field(default_factory=list)
    verdict: Literal["clear", "needs-author", "not-promotable"]


def _count(text: str, markers: tuple[str, ...]) -> int:
    padded = f" {text.lower()} "
    return sum(padded.count(f" {m} ") for m in markers)


def _present(text: str, markers: tuple[str, ...]) -> set[str]:
    padded = f" {text.lower()} "
    return {m for m in markers if f" {m} " in padded}


def _words(text: str) -> set[str]:
    return set(re.findall(r"[\wäöüÄÖÜß\-]+", text.lower()))


def clarify_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """Deterministic gate: meaning kept, scope grounded, hedges resolved, questions asked."""
    import dspy

    c: Clarification = pred.clarification
    original, excerpt = gold.claim_text, gold.source_excerpt
    context = " ".join([original, excerpt, getattr(gold, "canon_context", "")])
    glossary = [g.strip() for g in gold.glossary_terms.split(",") if g.strip()]
    problems: list[str] = []

    # 1. meaning kept: no glossary term smuggled in, no entity dropped, no new quantifier
    new_terms = [g for g in glossary if g.lower() in c.clarified_text.lower() and g.lower() not in context.lower()]
    dropped = [e for e in gold.entities if e.lower() not in c.clarified_text.lower()
               and not any(b.mention.lower() == e.lower() for b in c.bindings)]
    new_quant = _present(c.clarified_text, QUANTIFIERS) - _present(context, QUANTIFIERS)
    meaning = 1.0 - min(1.0, 0.5 * len(new_terms) + 0.5 * len(dropped) + 0.5 * len(new_quant))
    if new_terms:
        problems.append(f"Introduced terms absent from the source: {new_terms}.")
    if dropped:
        problems.append(f"Dropped entities: {dropped}.")
    if new_quant:
        problems.append(f"Added quantifiers the source does not state: {sorted(new_quant)}.")

    # 2. scope grounded: a value other than 'unspecified' must occur in the source/context
    ungrounded = [v for v in (c.scope.world, c.scope.act, c.scope.part)
                  if v != UNSPECIFIED and v.lower() not in context.lower()]
    grounded = 0.0 if ungrounded else 1.0
    if ungrounded:
        problems.append(f"Scope values not found in the source: {ungrounded}; use 'unspecified'.")

    # 3. hedges: fewer than the original, and any that remain are declared ambiguities
    left = _present(c.clarified_text, HEDGES)
    declared = {a.phrase.lower() for a in c.ambiguities}
    undeclared = [h for h in left if not any(h in d for d in declared)]
    hedges = 1.0 if not undeclared and _count(c.clarified_text, HEDGES) <= _count(original, HEDGES) else 0.0
    if undeclared:
        problems.append(f"Hedges left unresolved and undeclared: {undeclared}; resolve from the source or list them as ambiguities.")

    # 4. bindings: every slug must be a known glossary term
    bad_slugs = [b.slug for b in c.bindings if b.slug not in glossary]
    bindings = 1.0 if not bad_slugs else 0.0
    if bad_slugs:
        problems.append(f"Bindings to unknown glossary slugs: {bad_slugs}.")

    # 5. questions well-formed and verdict consistent
    malformed = [a.phrase for a in c.ambiguities if not a.question.strip().endswith("?")]
    consistent = (c.verdict != "clear") == bool(c.ambiguities)
    questions = 1.0 if not malformed and consistent else 0.0
    if malformed:
        problems.append(f"Ambiguities without a question: {malformed}.")
    if not consistent:
        problems.append("Verdict inconsistent with the ambiguity list ('clear' needs an empty list).")

    # 6. language kept: no English function words when the original is German
    german = bool(_words(original) & {"der", "die", "das", "und", "nicht", "wird", "ist"})
    english = bool(_words(c.clarified_text) & {"the", "and", "is", "not"})
    language = 0.0 if german and english else 1.0
    if german and english:
        problems.append("The claim was translated; keep the source language.")

    score = 0.3 * meaning + 0.15 * grounded + 0.15 * hedges + 0.15 * bindings + 0.15 * questions + 0.1 * language
    return dspy.Prediction(score=score, feedback=" ".join(problems) or "Clarified without adding or losing meaning.")


def build():
    import dspy

    class ClarifyClaim(dspy.Signature):
        """Rewrite the claim so that a reader with the glossary understands exactly what
        it asserts, and nothing more: make the scope explicit (world, act, character part)
        only where the source states it, bind names to glossary slugs, state implicit
        assumptions as assumptions, and turn every remaining ambiguity into a question
        for the author instead of choosing a reading. Never add, drop or generalise
        content; keep the source language."""

        claim_text: str = dspy.InputField()
        source_excerpt: str = dspy.InputField(desc="the cited lines of the source")
        entities: list[str] = dspy.InputField()
        glossary_terms: str = dspy.InputField(desc="comma-separated known slugs")
        canon_context: str = dspy.InputField(desc="canon passages about the same entities, or empty")
        clarification: Clarification = dspy.OutputField()

    class ClarifyGate(dspy.Module):
        def __init__(self):
            super().__init__()
            self.clarify = dspy.ChainOfThought(ClarifyClaim)

        def forward(self, claim_text: str, source_excerpt: str, entities: list[str],
                    glossary_terms: str, canon_context: str = ""):
            out = self.clarify(claim_text=claim_text, source_excerpt=source_excerpt, entities=entities,
                               glossary_terms=glossary_terms, canon_context=canon_context)
            return dspy.Prediction(clarification=out.clarification)

    return ClarifyGate


def fixture():
    import dspy

    return dspy.Example(
        claim_text="AEGIS wird meist nicht beim Namen genannt.",
        source_excerpt="AEGIS wird in Akt I nicht beim Namen genannt.",
        entities=["AEGIS"], glossary_terms="aegis, kael, kw1, kw3", canon_context="",
    ).with_inputs("claim_text", "source_excerpt", "entities", "glossary_terms", "canon_context")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Skip all LM calls")
    ap.add_argument("--model", default=os.getenv("DSPY_MODEL", "openai/gpt-4o"))
    args = ap.parse_args()

    import dspy

    ClarifyGate = build()
    gate = ClarifyGate()
    gold = fixture()

    if args.dry_run:
        good = Clarification(
            clarified_text="AEGIS wird in Akt I nicht beim Namen genannt.",
            scope=Scope(act="Akt I"), bindings=[Binding(mention="AEGIS", slug="aegis")], verdict="clear")
        bad = Clarification(
            clarified_text="AEGIS wird meist in KW3 nie beim Namen genannt.",
            scope=Scope(world="KW3"), bindings=[Binding(mention="AEGIS", slug="aegis-core")], verdict="clear")
        asks = Clarification(
            clarified_text="AEGIS wird meist nicht beim Namen genannt.",
            bindings=[Binding(mention="AEGIS", slug="aegis")],
            ambiguities=[Ambiguity(phrase="meist", readings=["in Akt I nie", "im ganzen Roman selten"],
                                   question="Gilt 'meist' für Akt I oder für den ganzen Roman?")],
            verdict="needs-author")
        g = clarify_metric(gold, dspy.Prediction(clarification=good))
        b = clarify_metric(gold, dspy.Prediction(clarification=bad))
        a = clarify_metric(gold, dspy.Prediction(clarification=asks))
        assert g.score > 0.99, g
        assert b.score < 0.5 and "KW3" in b.feedback and "nie" in b.feedback and "aegis-core" in b.feedback, b
        assert a.score > 0.99, a
        names = [n for n, _ in gate.named_predictors()]
        print("OK: ClarifyGate constructed with predictors", names)
        print(f"    metric good={g.score:.2f} asks={a.score:.2f} bad={b.score:.2f}")
        print(f"    bad feedback: {b.feedback}")
        return 0

    dspy.configure(lm=dspy.LM(args.model), track_usage=True)
    pred = gate(**gold.inputs())
    print(pred.clarification.model_dump_json(indent=2))
    print(clarify_metric(gold, pred))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
