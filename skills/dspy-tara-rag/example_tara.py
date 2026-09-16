"""dspy-tara-rag — runnable smoke test.

Implements the transferable core of TARA as standalone functions: the
four-dimensional context score, the progressive-leniency retry threshold, and
the tool-enablement policy per corpus type. The dry run exercises all three
without the repo, an index, or any LLM call.

Usage:
    uv run python example_tara.py --dry-run
    uv run python example_tara.py --score 25 20 18 15
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Literal

# Dimension ceilings from EvaluationSignature; they sum to 100.
MAX_RELEVANCE, MAX_COVERAGE, MAX_SPECIFICITY, MAX_SUFFICIENCY = 30, 25, 25, 20
DEFAULT_QUALITY_THRESHOLD = 40
THRESHOLD_DECAY_PER_RETRY = 5
THRESHOLD_FLOOR = 20                 # the code floors here; the docstring omits it
DEFAULT_MAX_RETRY = 3

CORE_TOOLS = ("search_passages", "decompose_query", "evaluate_passages", "get_passage_detail")
DOMAIN_TOOLS = ("list_document_sections", "get_terminology")
NUMERIC_TOOLS = ("calculate",)
ALL_TOOLS = CORE_TOOLS + DOMAIN_TOOLS + NUMERIC_TOOLS

Action = Literal["output", "refine", "route_to_agent"]


@dataclass(frozen=True)
class ContextScore:
    """The 4D assessment. Each dimension localizes a different failure."""

    relevance: int
    coverage: int
    specificity: int
    sufficiency: int

    def __post_init__(self) -> None:
        for name, value, ceiling in (
            ("relevance", self.relevance, MAX_RELEVANCE),
            ("coverage", self.coverage, MAX_COVERAGE),
            ("specificity", self.specificity, MAX_SPECIFICITY),
            ("sufficiency", self.sufficiency, MAX_SUFFICIENCY),
        ):
            if not 0 <= value <= ceiling:
                raise ValueError(f"{name}={value} outside 0..{ceiling}")

    @property
    def total(self) -> int:
        return self.relevance + self.coverage + self.specificity + self.sufficiency

    def weakest(self) -> str:
        """Which dimension to act on — the point of scoring four instead of one."""
        fractions = {
            "relevance": self.relevance / MAX_RELEVANCE,
            "coverage": self.coverage / MAX_COVERAGE,
            "specificity": self.specificity / MAX_SPECIFICITY,
            "sufficiency": self.sufficiency / MAX_SUFFICIENCY,
        }
        return min(fractions, key=fractions.get)


REPAIR = {
    "relevance": "wrong passages: reformulate the query, do not just fetch more",
    "coverage": "right topic, missing pieces: decompose the question and retrieve per sub-question",
    "specificity": "too general: browse document sections or map the user's terms to document terms",
    "sufficiency": "unanswerable from this context: widen retrieval or admit the gap",
}


def effective_threshold(retry: int, base: int = DEFAULT_QUALITY_THRESHOLD) -> int:
    """Progressive leniency: the bar drops per retry, with a floor."""
    return max(base - retry * THRESHOLD_DECAY_PER_RETRY, THRESHOLD_FLOOR)


def decide(score: ContextScore, retry: int, max_retry: int = DEFAULT_MAX_RETRY) -> Action:
    if score.total >= effective_threshold(retry):
        return "output"
    if retry >= max_retry:
        return "route_to_agent"
    return "refine"


def enabled_tools(corpus: str) -> tuple[str, ...]:
    """Domain-adaptive tools are dead weight on an unstructured corpus."""
    if corpus == "wikipedia":
        return CORE_TOOLS                       # structure/terminology add nothing here
    if corpus == "financial":
        return CORE_TOOLS + DOMAIN_TOOLS + NUMERIC_TOOLS
    if corpus == "enterprise":
        return CORE_TOOLS + DOMAIN_TOOLS
    return ALL_TOOLS


def estimated_llm_calls(n_tool_calls: int) -> int:
    """decompose_query and evaluate_passages each make their own LLM call,
    so the agent's iteration count understates real cost."""
    return n_tool_calls + 2 * max(0, n_tool_calls - 2) + 1


def dry_run() -> None:
    good = ContextScore(relevance=28, coverage=22, specificity=20, sufficiency=18)
    thin = ContextScore(relevance=26, coverage=8, specificity=18, sufficiency=6)
    wrong = ContextScore(relevance=5, coverage=6, specificity=12, sufficiency=4)

    for label, score in (("good", good), ("thin coverage", thin), ("wrong passages", wrong)):
        weak = score.weakest()
        print(f"{label:15s} total={score.total:3d} weakest={weak:12s} -> {REPAIR[weak]}")
    assert good.total == 88 and thin.weakest() == "sufficiency"
    assert wrong.weakest() == "sufficiency" or wrong.weakest() == "relevance"

    print("thresholds:", [effective_threshold(r) for r in range(6)])
    assert effective_threshold(0) == 40 and effective_threshold(4) == THRESHOLD_FLOOR
    assert effective_threshold(99) == THRESHOLD_FLOOR, "the floor must hold"

    borderline = ContextScore(relevance=12, coverage=8, specificity=8, sufficiency=5)  # 33
    print(f"borderline total={borderline.total}: "
          f"retry0={decide(borderline, 0)} retry2={decide(borderline, 2)} "
          f"retry3={decide(borderline, 3)}")
    assert decide(borderline, 0) == "refine"
    assert decide(borderline, 2) == "output", "leniency should admit it by retry 2"
    assert decide(good, 0) == "output"

    # Progressive leniency terminates the loop by lowering the bar, not by
    # escalating. Anything at or above the floor is eventually accepted.
    assert decide(wrong, DEFAULT_MAX_RETRY) == "output", (
        f"total={wrong.total} clears the decayed threshold "
        f"{effective_threshold(DEFAULT_MAX_RETRY)} — accepted, not escalated"
    )
    hopeless = ContextScore(relevance=3, coverage=2, specificity=2, sufficiency=2)   # 9
    assert decide(hopeless, DEFAULT_MAX_RETRY) == "route_to_agent"
    print(f"wrong total={wrong.total} at retry {DEFAULT_MAX_RETRY} "
          f"(threshold {effective_threshold(DEFAULT_MAX_RETRY)}) -> {decide(wrong, DEFAULT_MAX_RETRY)}; "
          f"only total<{THRESHOLD_FLOOR} escalates")

    for corpus in ("wikipedia", "financial", "enterprise", "unknown"):
        tools = enabled_tools(corpus)
        print(f"{corpus:11s} -> {len(tools)} tools: {', '.join(tools)}")
    assert len(enabled_tools("wikipedia")) == 4
    assert len(enabled_tools("unknown")) == 7, "all seven tools, not the README's six"

    for calls in (1, 5, 10):
        print(f"{calls:2d} tool calls -> ~{estimated_llm_calls(calls)} LLM calls")
    assert estimated_llm_calls(5) > 5, "internal tool LLM calls must be counted"
    print("dry run OK")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Run without the repo or any LLM")
    ap.add_argument("--score", nargs=4, type=int, metavar=("REL", "COV", "SPEC", "SUF"),
                    help="Score one context and print the repair action")
    ap.add_argument("--retry", type=int, default=0)
    args = ap.parse_args()
    if args.score:
        score = ContextScore(*args.score)
        weak = score.weakest()
        print(f"total={score.total} threshold={effective_threshold(args.retry)} "
              f"action={decide(score, args.retry)}")
        print(f"weakest={weak}: {REPAIR[weak]}")
        return
    dry_run()


if __name__ == "__main__":
    main()
