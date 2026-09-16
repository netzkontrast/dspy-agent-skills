"""dspy-adversarial-review — runnable smoke test.

An independent judge that cannot rewrite. A reviewer LM that is provably not
the writer's LM scores an artifact against its evidence (overstated claims
with the evidence they would need, unsupported claims, strengths, weaknesses,
a 1–10 score); a citation-support check grades each cited span; a demotion
rule turns too many overstated claims into a status change; a refine loop
re-runs the writer until the review score meets a target; judgements are
cached by content hash. Patterns from synthadoc (adversarial review gate
with threshold demotion), AutoSci (/review with an independent model,
/refine to a target score), quicky-wiki (redteam of high-confidence claims)
and llm-wiki-compiler (citation-support judge with cached judgements).
The dry run checks the independence guard, the demotion rule, the cache and
the judge metric without any LM call.

Usage:
    uv run python example_adversarial_review.py --dry-run
    OPENAI_API_KEY=... uv run python example_adversarial_review.py
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
from typing import Literal

from pydantic import BaseModel, Field

Difficulty = Literal["standard", "hard", "adversarial"]
Verdict = Literal["supported", "partial", "unsupported"]
DEMOTE_AT = 3
TARGET_SCORE = 0.8


class Overstated(BaseModel):
    quote: str
    why: str
    evidence_needed: str


class Review(BaseModel):
    score: int = Field(ge=1, le=10)
    overstated: list[Overstated] = Field(default_factory=list)
    unsupported: list[str] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)


class Support(BaseModel):
    verdict: Verdict
    reason: str


# --- deterministic rules ---------------------------------------------------------

def same_lm(a, b) -> bool:
    if a is b:
        return True
    return getattr(a, "model", None) == getattr(b, "model", None) and getattr(a, "kwargs", None) == getattr(b, "kwargs", None)


def assert_independent(reviewer, writer) -> None:
    """The judge may not be the author: same object or same model+kwargs is refused."""
    if writer is None:
        raise ValueError("no writer LM configured; call dspy.configure(lm=...) for the writer first")
    if same_lm(reviewer, writer):
        raise ValueError(f"reviewer must differ from the writer LM ({getattr(writer, 'model', writer)!r})")


def demotion(review: Review, demote_at: int = DEMOTE_AT) -> Literal["keep", "contested"]:
    """Too many overstated claims demote the artifact's status; its body is never touched."""
    return "contested" if len(review.overstated) >= demote_at else "keep"


def digest(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:16]


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _matches(flagged: list[str], gold: list[str]) -> tuple[list[str], list[str]]:
    """(missed gold items, spurious flagged items) with substring tolerance either way."""
    hit = lambda a, b: _norm(a) in _norm(b) or _norm(b) in _norm(a)  # noqa: E731
    missed = [g for g in gold if not any(hit(g, f) for f in flagged)]
    spurious = [f for f in flagged if not any(hit(g, f) for g in gold)]
    return missed, spurious


def _f1(flagged: list[str], gold: list[str]) -> float:
    if not flagged and not gold:
        return 1.0
    missed, spurious = _matches(flagged, gold)
    tp = len(flagged) - len(spurious)
    precision = tp / len(flagged) if flagged else 0.0
    recall = (len(gold) - len(missed)) / len(gold) if gold else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def judge_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """Score the reviewer itself: F1 on overstated and unsupported claims against a labelled set."""
    import dspy

    review: Review = pred.review
    flagged = [o.quote for o in review.overstated]
    missed_o, spurious_o = _matches(flagged, gold.overstated)
    missed_u, spurious_u = _matches(review.unsupported, gold.unsupported)
    parts = []
    if missed_o:
        parts.append(f"missed overstated claims: {missed_o}")
    if spurious_o:
        parts.append(f"flagged supported claims as overstated: {spurious_o}")
    if missed_u:
        parts.append(f"missed unsupported claims: {missed_u}")
    if spurious_u:
        parts.append(f"flagged cited claims as unsupported: {spurious_u}")
    score = round(0.5 * _f1(flagged, gold.overstated) + 0.5 * _f1(review.unsupported, gold.unsupported), 3)
    return dspy.Prediction(score=score, feedback="; ".join(parts) or "every overstated and unsupported claim found, none invented")


# --- program -------------------------------------------------------------------

def build():
    import dspy

    class ReviewArtifact(dspy.Signature):
        """Review the artifact against the evidence only. Quote every overstated claim verbatim and say what
        evidence would be needed; list claims with no support in the evidence; score 1–10. 'hard' demands a
        citation for every factual sentence; 'adversarial' also attacks the framing and the omissions.
        Never propose replacement text."""

        artifact: str = dspy.InputField()
        evidence: str = dspy.InputField(desc="the pages or spans the artifact may rely on")
        difficulty: Difficulty = dspy.InputField()
        review: Review = dspy.OutputField()

    class CitationSupport(dspy.Signature):
        """Does the cited span support the claim: fully, partially, or not at all?"""

        claim: str = dspy.InputField()
        span: str = dspy.InputField()
        support: Support = dspy.OutputField()

    class AdversarialReviewer(dspy.Module):
        def __init__(self, reviewer_lm, demote_at: int = DEMOTE_AT):
            super().__init__()
            self.review = dspy.ChainOfThought(ReviewArtifact)
            self.support = dspy.Predict(CitationSupport)
            self.reviewer_lm = reviewer_lm
            self.demote_at = demote_at
            self.cache: dict[str, dspy.Prediction] = {}

        def forward(self, artifact: str, evidence: str, difficulty: Difficulty = "standard"):
            assert_independent(self.reviewer_lm, dspy.settings.lm)
            key = digest(artifact, evidence, difficulty, str(self.reviewer_lm.model))
            if key in self.cache:
                return self.cache[key]
            with dspy.context(lm=self.reviewer_lm):
                review = self.review(artifact=artifact, evidence=evidence, difficulty=difficulty).review
            self.cache[key] = dspy.Prediction(review=review, verdict=demotion(review, self.demote_at), cache_key=key)
            return self.cache[key]

        def check_citation(self, claim: str, span: str) -> Support:
            with dspy.context(lm=self.reviewer_lm):
                return self.support(claim=claim, span=span).support

    def review_reward(reviewer: AdversarialReviewer, evidence: str, difficulty: Difficulty = "standard"):
        """Reward for dspy.Refine around the writer: the independent review score in [0, 1]."""

        def reward(args: dict, pred) -> float:
            return reviewer(artifact=pred.text, evidence=evidence, difficulty=difficulty).review.score / 10

        return reward

    def review_refine(writer: dspy.Module, reviewer: AdversarialReviewer, evidence: str, rounds: int = 3):
        """The writer is re-run until the reviewer's score reaches the target or the rounds are spent."""
        return dspy.Refine(module=writer, N=rounds, reward_fn=review_reward(reviewer, evidence), threshold=TARGET_SCORE)

    return AdversarialReviewer, review_refine


# --- fixture -------------------------------------------------------------------

ARTIFACT = ("Juna lebt in KW2. Der Schleier ist das wichtigste Ritual aller Kernwelten. "
            "Der Schleier hält bis Kapitel 13. Kael kennt Juna seit Kapitel 1.")
EVIDENCE = "a.md:1 Juna lebt in KW2.\na.md:2 Der Schleier hält bis Kapitel 13.\nb.md:2 Der Schleier ist ein Ritual."
GOLD = {"overstated": ["das wichtigste Ritual aller Kernwelten"], "unsupported": ["Kael kennt Juna seit Kapitel 1"]}


def fixture_review() -> Review:
    return Review(score=6, strengths=["Wohnort und Dauer sind belegt"], weaknesses=["Superlativ ohne Beleg"],
                  overstated=[Overstated(quote="das wichtigste Ritual aller Kernwelten", why="no source ranks rituals",
                                         evidence_needed="a passage comparing rituals across Kernwelten")],
                  unsupported=["Kael kennt Juna seit Kapitel 1"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Skip all LM calls")
    ap.add_argument("--model", default=os.getenv("DSPY_MODEL", "openai/gpt-4o"))
    ap.add_argument("--reviewer", default=os.getenv("DSPY_REVIEWER", "openai/gpt-4o-mini"))
    args = ap.parse_args()

    import dspy

    AdversarialReviewer, review_refine = build()
    writer, reviewer_lm = dspy.LM(args.model), dspy.LM(args.reviewer)   # construction never hits the network
    gold = dspy.Example(artifact=ARTIFACT, evidence=EVIDENCE, **GOLD).with_inputs("artifact", "evidence")

    if args.dry_run:
        try:
            assert_independent(writer, writer)
            raise AssertionError("same LM must be refused")
        except ValueError as e:
            assert "must differ" in str(e)
        assert_independent(reviewer_lm, writer)
        review = fixture_review()
        assert demotion(review) == "keep" and demotion(review, demote_at=1) == "contested"
        exact = judge_metric(gold, dspy.Prediction(review=review))
        assert exact.score == 1.0, exact
        sloppy = Review(score=9, overstated=[Overstated(quote="Juna lebt in KW2", why="?", evidence_needed="?")])
        judged = judge_metric(gold, dspy.Prediction(review=sloppy))
        assert judged.score < 0.5 and "missed overstated" in judged.feedback and "flagged supported" in judged.feedback
        assert "missed unsupported" in judged.feedback
        reviewer = AdversarialReviewer(reviewer_lm)
        assert digest(ARTIFACT, EVIDENCE, "standard") == digest(ARTIFACT, EVIDENCE, "standard")
        print(f"OK: AdversarialReviewer constructed with predictors {[n for n, _ in reviewer.named_predictors()]}")
        print(f"    exact review metric: {exact.score} · sloppy review: {judged.score} — {judged.feedback[:120]}…")
        return 0

    dspy.configure(lm=writer)
    reviewer = AdversarialReviewer(reviewer_lm)
    result = reviewer(artifact=ARTIFACT, evidence=EVIDENCE, difficulty="hard")
    print(result.verdict, result.review)
    print(judge_metric(gold, result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
