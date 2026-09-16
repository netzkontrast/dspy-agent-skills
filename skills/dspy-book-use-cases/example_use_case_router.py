"""dspy-book-use-cases — runnable smoke test.

Routes a task description to the closest worked example, and implements the
two metric patterns worth stealing: a deterministic gate that refuses to call
the judge on a wrong answer, and trajectory grading that penalizes a right
answer reached badly. No LM, no network.

Usage:
    uv run python example_use_case_router.py --dry-run
    uv run python example_use_case_router.py --task "pull fields out of scanned forms"
"""

from __future__ import annotations

import argparse
import math

NUMERIC_TOLERANCE = 0.05          # relative
POISONED_SCORE, DRIFTED_SCORE, CLEAN_SCORE = 0.3, 0.7, 1.0

USE_CASES = {
    "classification": ("sentiment-classifier",
                       "inline signature; the demos carry the label boundary, not the prompt"),
    "extraction": ("invoice-extraction",
                   "partial-credit metric, per-field voting, a committed benchmark artifact"),
    "grounded_qa": ("customer-service-rag",
                    "narrow tools inside a sandwich guardrail"),
    "research": ("news-researcher",
                 "decompose, fan out in parallel, synthesize preserving conflicts"),
    "computational": ("financial-analyst",
                      "code execution plus a metric that grades the reasoning trajectory"),
    "creative_text": ("blog-writer",
                      "embedding distance to reference passages when no gold string exists"),
    "creative_media": ("video-generator",
                       "quality gate BEFORE the expensive irreversible call"),
}

KEYWORDS = {
    "classification": ("classify", "label", "sentiment", "category", "triage"),
    "extraction": ("extract", "fields", "parse", "invoice", "form", "structured"),
    "grounded_qa": ("support", "answer from", "our docs", "customer", "policy"),
    "research": ("research", "multi-hop", "sources", "investigate", "brief"),
    "computational": ("calculate", "financial", "numbers", "arithmetic", "filing"),
    "creative_text": ("write", "blog", "copy", "style", "tone"),
    "creative_media": ("image", "video", "render", "generate a picture"),
}


def route(task: str) -> tuple[str, str]:
    lowered = task.lower()
    best, best_hits = None, 0
    for shape, words in KEYWORDS.items():
        hits = sum(w in lowered for w in words)
        if hits > best_hits:
            best, best_hits = shape, hits
    if best is None:
        return ("no match", "describe the task in terms of its output shape")
    return USE_CASES[best]


def parse_number(text: str) -> float:
    """Normalize currency and percentages; RAISE on unparseable, never guess."""
    cleaned = text.strip().replace(",", "").replace("$", "").replace("€", "")
    multiplier = 1.0
    for suffix, factor in (("B", 1e9), ("billion", 1e9), ("M", 1e6), ("million", 1e6)):
        if cleaned.lower().endswith(suffix.lower()):
            cleaned, multiplier = cleaned[: -len(suffix)].strip(), factor
            break
    if cleaned.endswith("%"):
        cleaned = cleaned[:-1]
    return float(cleaned) * multiplier


def graded_metric(gold_value: float, predicted_text: str, *,
                  poisoning_detected: bool = False, drift_detected: bool = False,
                  judge_calls: list | None = None) -> float:
    """Deterministic gate first; the judge is only called on a correct answer."""
    judge_calls = judge_calls if judge_calls is not None else []
    try:
        value = parse_number(predicted_text)
    except ValueError:
        return 0.0                                  # unparseable: no judge call
    if not math.isclose(value, gold_value, rel_tol=NUMERIC_TOLERANCE):
        return 0.0                                  # wrong: no judge call
    judge_calls.append(predicted_text)              # only correct answers reach the judge
    if poisoning_detected:
        return POISONED_SCORE
    if drift_detected:
        return DRIFTED_SCORE
    return CLEAN_SCORE


def guard_sandwich(message: str, *, regex_hits: bool, llm_says_injection: bool) -> tuple[bool, str]:
    """Cheap check first so most attacks cost nothing; tools never reached on a block."""
    if regex_hits:
        return False, "blocked by regex; no LLM call, no tool call"
    if llm_says_injection:
        return False, "blocked by classifier; no tool call"
    return True, "reaches the agent"


def dry_run() -> None:
    for task in ("classify these tickets by department",
                 "extract fields from scanned invoices",
                 "answer customer questions from our policy docs",
                 "research this topic across many sources",
                 "calculate the revenue change from these filings",
                 "write blog copy in our house style",
                 "generate a product video"):
        example, move = route(task)
        print(f"{task[:44]:46s} -> {example:24s} {move[:46]}")
    assert route("extract fields from invoices")[0] == "invoice-extraction"
    assert route("generate a product video")[0] == "video-generator"

    assert math.isclose(parse_number("$1.25B"), 1.25e9)
    assert math.isclose(parse_number("-2.8%"), -2.8)
    assert math.isclose(parse_number("42 million"), 42e6)
    try:
        parse_number("about forty-two")
    except ValueError:
        print("unparseable number raises rather than scoring silently")
    else:
        raise AssertionError("an unparseable answer must raise")

    calls: list = []
    print(f"wrong answer    -> {graded_metric(100.0, '250', judge_calls=calls):.1f}")
    print(f"unparseable     -> {graded_metric(100.0, 'roughly a hundred', judge_calls=calls):.1f}")
    assert calls == [], "the judge must not be called on a wrong or unparseable answer"

    print(f"right, clean    -> {graded_metric(100.0, '$100', judge_calls=calls):.1f}")
    print(f"right, drifted  -> {graded_metric(100.0, '$100', drift_detected=True, judge_calls=calls):.1f}")
    print(f"right, poisoned -> {graded_metric(100.0, '$100', poisoning_detected=True, judge_calls=calls):.1f}")
    assert len(calls) == 3, "only correct answers reach the judge"
    assert graded_metric(100.0, "$100", poisoning_detected=True) < graded_metric(100.0, "$100")

    for label, kwargs in (("clean message", dict(regex_hits=False, llm_says_injection=False)),
                          ("obvious attack", dict(regex_hits=True, llm_says_injection=True)),
                          ("subtle attack", dict(regex_hits=False, llm_says_injection=True))):
        allowed, why = guard_sandwich("...", **kwargs)
        print(f"{label:16s} allowed={allowed} | {why}")
    assert not guard_sandwich("x", regex_hits=True, llm_says_injection=False)[0]
    print("dry run OK")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Run without an LM")
    ap.add_argument("--task", help="Route a task description to a worked example")
    args = ap.parse_args()
    if args.task:
        example, move = route(args.task)
        print(f"closest example: {example}\nwhat to copy: {move}")
        return
    dry_run()


if __name__ == "__main__":
    main()
