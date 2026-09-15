"""dspy-reflect-loop — runnable smoke test.

Builds the reflect loop (pre-filter + ExtractLearningSignal), the fingerprint
ledger with promotion threshold, the conversion of a correction into a gold
example + metric, and the reflector metric. The dry run exercises every
deterministic part on a toy transcript without any LM call.

Usage:
    uv run python example_reflect_loop.py --dry-run
    OPENAI_API_KEY=... uv run python example_reflect_loop.py
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
from typing import Literal

from pydantic import BaseModel, Field

PREFILTER = re.compile(
    r"(?i)\b(instead of|statt|don't|do not|never|niemals|always|immer|actually|nein,|no,|"
    r"use \w+ instead|verwende|benutze|remember:|merk dir|perfect|exactly|genau so|"
    r"works? (perfectly|great)|have you considered|why not)\b"
)
MIN_MESSAGE_CHARS = 10
PROMOTION_THRESHOLD = 2


class LearningSignal(BaseModel):
    is_learning: bool
    kind: Literal["correction", "approval", "observation", "explicit"] | None = None
    confidence: Literal["HIGH", "MEDIUM", "LOW"] | None = None
    old_behavior: str = ""
    new_behavior: str = ""
    learning: str = ""
    scope_hint: Literal["program", "project", "global"] = "program"


def fingerprint(learning: str) -> str:
    return hashlib.sha256(" ".join(learning.lower().split()).encode()).hexdigest()[:16]


class Ledger:
    """In-memory ledger; persist as JSONL in real use."""

    def __init__(self) -> None:
        self.rows: dict[str, dict] = {}

    def record(self, sig: LearningSignal, context_id: str) -> str:
        fp = fingerprint(sig.learning)
        row = self.rows.setdefault(fp, {"learning": sig.learning, "confidence": sig.confidence,
                                        "contexts": [], "count": 0, "status": "pending"})
        if context_id not in row["contexts"]:
            row["contexts"].append(context_id)
        row["count"] += 1
        return fp

    def eligible_for_promotion(self, threshold: int = PROMOTION_THRESHOLD) -> list[str]:
        return [fp for fp, r in self.rows.items() if len(r["contexts"]) >= threshold and r["status"] != "promoted"]


def candidate_turns(transcript: list[dict]) -> list[int]:
    """Deterministic pre-filter: user turns long enough that match a signal pattern."""
    return [i for i, t in enumerate(transcript)
            if t["role"] == "user" and len(t["content"]) >= MIN_MESSAGE_CHARS and PREFILTER.search(t["content"])]


def build():
    import dspy

    class ExtractLearningSignal(dspy.Signature):
        """Decide whether a user message contains a reusable learning for the program
        (not a one-off instruction). Corrections are HIGH, approvals MEDIUM,
        suggestions LOW; keep the learning in the user's language."""

        message: str = dspy.InputField()
        prior_assistant_turn: str = dspy.InputField()
        program_name: str = dspy.InputField()
        signal: LearningSignal = dspy.OutputField()

    class ReflectLoop(dspy.Module):
        def __init__(self):
            super().__init__()
            self.extract = dspy.ChainOfThought(ExtractLearningSignal)

        def forward(self, transcript: list[dict], program_name: str):
            signals = []
            for i in candidate_turns(transcript):
                prior = next((t["content"] for t in reversed(transcript[:i]) if t["role"] == "assistant"), "")
                sig = self.extract(message=transcript[i]["content"], prior_assistant_turn=prior,
                                   program_name=program_name).signal
                if sig.is_learning:
                    signals.append(sig)
            return dspy.Prediction(signals=signals)

    def learning_to_example(sig: LearningSignal, failing_input: dict) -> dspy.Example:
        return dspy.Example(**failing_input, expected_behavior=sig.new_behavior,
                            forbidden_behavior=sig.old_behavior,
                            feedback=f"User corrected this: {sig.learning}").with_inputs(*failing_input)

    def corrections_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        text = str(pred.toDict()).lower()
        satisfied = not gold.expected_behavior or gold.expected_behavior.lower() in text
        # A violation is the old behaviour showing up *without* the corrected one
        # (the corrected form may contain the old one as a substring, e.g. "uv pip install").
        violated = bool(gold.forbidden_behavior) and gold.forbidden_behavior.lower() in text and not satisfied
        score = 0.0 if violated else (1.0 if satisfied else 0.5)
        fb = gold.feedback if (violated or not satisfied) else "Follows the learned correction."
        return dspy.Prediction(score=score, feedback=fb)

    def reflector_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        got = {(s.learning.lower(), s.confidence) for s in pred.signals}
        want = {(s["learning"].lower(), s["confidence"]) for s in gold.accepted}
        noise = [s for s in pred.signals if s.learning.lower() in {x.lower() for x in gold.skipped}]
        tp, fn = len(got & want), len(want - got)
        score = (tp / max(1, len(want))) * (1 - len(noise) / max(1, len(pred.signals)))
        fb = []
        if fn:
            fb.append(f"Missed {fn} accepted learning(s).")
        if noise:
            fb.append(f"{len(noise)} proposal(s) the user skipped before.")
        return dspy.Prediction(score=score, feedback=" ".join(fb) or "All accepted learnings found, no skipped noise.")

    return ReflectLoop, learning_to_example, corrections_metric, reflector_metric


TRANSCRIPT = [
    {"role": "user", "content": "Create a small Python project with tests."},
    {"role": "assistant", "content": "I set it up with pip and unittest."},
    {"role": "user", "content": "No, use uv instead of pip. And always pytest, never unittest."},
    {"role": "assistant", "content": "Switched to uv and pytest."},
    {"role": "user", "content": "Yes, perfect!"},
    {"role": "user", "content": "ok"},
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Skip all LM calls")
    ap.add_argument("--model", default=os.getenv("DSPY_MODEL", "openai/gpt-4o"))
    args = ap.parse_args()

    import dspy

    ReflectLoop, learning_to_example, corrections_metric, reflector_metric = build()
    loop = ReflectLoop()

    if args.dry_run:
        picked = candidate_turns(TRANSCRIPT)
        assert picked == [2, 4], picked                     # correction + approval; "ok" filtered
        correction = LearningSignal(is_learning=True, kind="correction", confidence="HIGH",
                                    old_behavior="pip install", new_behavior="uv pip install",
                                    learning="Use uv instead of pip", scope_hint="global")
        ledger = Ledger()
        ledger.record(correction, "repo-a")
        assert ledger.eligible_for_promotion() == []
        fp = ledger.record(correction, "repo-b")
        assert ledger.eligible_for_promotion() == [fp]
        gold = learning_to_example(correction, {"task": "create project"})
        bad = corrections_metric(gold, dspy.Prediction(commands="pip install fastapi"))
        good = corrections_metric(gold, dspy.Prediction(commands="uv pip install fastapi"))
        assert bad.score == 0.0 and "User corrected" in bad.feedback and good.score == 1.0
        rgold = dspy.Example(accepted=[{"learning": "Use uv instead of pip", "confidence": "HIGH"}],
                             skipped=["ok"])
        rpred = dspy.Prediction(signals=[correction])
        assert reflector_metric(rgold, rpred).score == 1.0
        names = [n for n, _ in loop.named_predictors()]
        print("OK: ReflectLoop constructed with predictors", names)
        print(f"    pre-filter picked turns {picked}; ledger promotes after {PROMOTION_THRESHOLD} contexts ({fp})")
        print(f"    corrections_metric bad={bad.score:.1f} good={good.score:.1f}; reflector_metric=1.0")
        return 0

    dspy.configure(lm=dspy.LM(args.model), track_usage=True)
    pred = loop(transcript=TRANSCRIPT, program_name="python-project-creator")
    for s in pred.signals:
        print(f"  [{s.confidence}] {s.kind}: {s.learning}  (old={s.old_behavior!r} new={s.new_behavior!r})")
    print("Nothing was applied. Review each signal (accept / modify / skip) before it becomes gold.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
