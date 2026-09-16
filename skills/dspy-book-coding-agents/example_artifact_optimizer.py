"""dspy-book-coding-agents — runnable smoke test.

The chapter's recipe for optimizing a text artifact, as checkable pieces: the
evaluator-signature trap that silently drops data, the benchmark case shape,
the weighted mechanical-plus-judge metric, and the regression list you must
read before shipping. The dry run needs no LM and no gepa install.

Usage:
    uv run python example_artifact_optimizer.py --dry-run
"""

from __future__ import annotations

import argparse
import inspect
import math
import re
from dataclasses import dataclass

REQUIRED_EVALUATOR_PARAM = "example"      # optimize_anything introspects this name
MECHANICAL_WEIGHT = 0.5                   # noise-free signal outweighs judge signal
JUDGE_WEIGHT = 0.5
SMOKE_METRIC_CALLS = 3
PRODUCTION_METRIC_CALLS = 200
VERDICT_RE = re.compile(r"VERDICT:\s*(PASS|FAIL)", re.IGNORECASE)


@dataclass(frozen=True)
class BenchmarkCase:
    """A convention becomes testable as task / antipattern / tell."""

    task: str                 # terse and UNDER-specified: that is when defaults leak
    antipattern: str          # the habit this task is designed to elicit
    tell: str                 # the concrete observable the judge looks for

    def __post_init__(self) -> None:
        for field in ("task", "antipattern", "tell"):
            if not getattr(self, field).strip():
                raise ValueError(f"{field} must not be empty")


CASES = [
    BenchmarkCase("Load config.json and return the port.",
                  "defensive try/except on a happy path",
                  "try/except around json.load with no failure mode in the request"),
    BenchmarkCase("Fix the type error on line 12.",
                  "silencing instead of fixing",
                  "# type: ignore rather than handling the Optional"),
    BenchmarkCase("Add a helper that formats currency.",
                  "new module when one exists",
                  "creates utils.py beside an existing formatters module"),
    BenchmarkCase("Write a test for the discount rule.",
                  "asserting on the mock",
                  "assertions reference the mock object, not the computed result"),
    BenchmarkCase("Summarize what you changed.",
                  "self-congratulatory summary",
                  "checkmarks and praise instead of a plain list of edits"),
]


def evaluator_signature_ok(fn) -> tuple[bool, str]:
    """optimize_anything reads the evaluator's parameter NAMES, not positions."""
    params = list(inspect.signature(fn).parameters)
    if len(params) < 2:
        return False, "evaluator needs (candidate, example)"
    if params[1] != REQUIRED_EVALUATOR_PARAM:
        return False, (f"second parameter is {params[1]!r}, must be "
                       f"{REQUIRED_EVALUATOR_PARAM!r}; GEPA silently drops the dataset otherwise")
    return True, "signature accepted"


def good_evaluator(candidate: str, example: dict) -> tuple[float, dict]:
    """Correct shape: returns a score AND the reason the reflector needs."""
    return 1.0, {"Prompt": example.get("task", ""), "JudgeReasoning": "..."}


def bad_evaluator(candidate: str, task: dict) -> tuple[float, dict]:
    """Same body, renamed parameter — the silent failure."""
    return 1.0, {}


def parse_verdict(text: str) -> bool:
    """Binary verdict; an unparseable judge raises rather than scoring zero."""
    match = VERDICT_RE.search(text)
    if not match:
        raise ValueError(f"judge returned no parseable verdict: {text[:60]!r}")
    return match.group(1).upper() == "PASS"


def composite_score(mechanical_checks: list[bool], judge_wins: list[bool]) -> float:
    """Weighted: deterministic checks carry more than judge opinion."""
    mech = sum(mechanical_checks) / len(mechanical_checks) if mechanical_checks else 0.0
    judged = sum(judge_wins) / len(judge_wins) if judge_wins else 0.0
    return MECHANICAL_WEIGHT * mech + JUDGE_WEIGHT * judged


def regressions(cases: list[BenchmarkCase], baseline: list[float],
                optimized: list[float]) -> list[str]:
    """GEPA drops rules that never fired. This is how you find them."""
    return [c.antipattern for c, b, o in zip(cases, baseline, optimized) if b > o]


def side_info_is_useful(side_info: dict) -> bool:
    """The reflection LM sees only the score and this dict."""
    return bool(side_info) and any("reason" in k.lower() or "judge" in k.lower()
                                   for k in side_info)


def dry_run() -> None:
    ok, why = evaluator_signature_ok(good_evaluator)
    print(f"good evaluator: {ok} | {why}")
    assert ok
    ok, why = evaluator_signature_ok(bad_evaluator)
    print(f"renamed param : {ok} | {why}")
    assert not ok, "the renamed-parameter trap must be caught"

    print(f"benchmark cases: {len(CASES)}")
    for case in CASES[:3]:
        print(f"  {case.antipattern:34s} tell: {case.tell[:44]}")
    try:
        BenchmarkCase("do a thing", "", "something")
    except ValueError:
        pass
    else:
        raise AssertionError("an empty antipattern must be rejected")

    assert parse_verdict("Reasoning here.\nVERDICT: PASS") is True
    assert parse_verdict("VERDICT: fail") is False
    try:
        parse_verdict("I think it is probably fine")
    except ValueError:
        print("unparseable verdict raises, so a broken judge cannot look like a failing candidate")
    else:
        raise AssertionError("an unparseable verdict must raise")

    all_mech = composite_score([True] * 5, [False, False])
    all_judge = composite_score([False] * 5, [True, True])
    print(f"mechanical only={all_mech:.2f} judge only={all_judge:.2f}")
    assert math.isclose(all_mech, MECHANICAL_WEIGHT)
    assert math.isclose(all_judge, JUDGE_WEIGHT)

    baseline = [1.0, 0.0, 1.0, 0.0, 1.0]
    optimized = [1.0, 1.0, 0.0, 1.0, 1.0]
    lost = regressions(CASES, baseline, optimized)
    print(f"net change {sum(optimized) - sum(baseline):+.1f}, but regressed: {lost}")
    assert lost == ["new module when one exists"], (
        "a net gain can still hide a regression; that is why you print the list"
    )

    assert side_info_is_useful(good_evaluator("x", {"task": "t"})[1])
    assert not side_info_is_useful(bad_evaluator("x", {})[1])
    print(f"budgets: smoke={SMOKE_METRIC_CALLS} production={PRODUCTION_METRIC_CALLS}")
    print("dry run OK")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Run without an LM or gepa")
    ap.parse_args()
    dry_run()


if __name__ == "__main__":
    main()
