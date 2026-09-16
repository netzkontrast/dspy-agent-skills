"""dspy-book-metrics — runnable smoke test.

The chapter's metric ladder as deterministic functions, plus the routing table
the chapter never states and the judge-calibration trust bar. The dry run
needs no LM, no embeddings and no network.

Usage:
    uv run python example_metric_recipes.py --dry-run
    uv run python example_metric_recipes.py --route "a paraphrase of a known answer"
"""

from __future__ import annotations

import argparse
import math
import re

KEYWORD_BOUNDARY = r'\b{}\b'
JUDGE_TRUST_FLOOR = 0.80          # chapter: 80-90% agreement before trusting a judge
JUDGE_TRUST_GOOD = 0.90
RUBRIC_MIN, RUBRIC_MAX = 1, 5

# Task shape -> (metric, the trap that shape invites)
ROUTING = {
    "one canonical short answer": ("answer_exact_match (built-in)", "'The capital is Paris.' fails against 'Paris'"),
    "required structure, free wording": ("regex over structure", "binary; prefer partial credit"),
    "a set of required points": ("partial-credit keyword overlap", "anchor on word boundaries"),
    "ocr or spelling repair": ("Levenshtein threshold", "ignores meaning entirely"),
    "a paraphrase of a known answer": ("embedding cosine", "cannot detect negation"),
    "an extractive span": ("token F1", "rewards overlap, not correctness"),
    "a translation or summary": ("BLEU / ROUGE", "chapter rates these below embeddings"),
    "subjective quality": ("calibrated LLM judge", "untrusted below 80% agreement"),
    "several quality axes": ("weighted rubric", "check weights at both extremes"),
    "a multi-step pipeline": ("trace-aware per-predictor feedback", "latency on every call"),
}


def keyword_binary(answer: str, keywords: list[str]) -> float:
    """All-or-nothing: the shape the chapter warns against."""
    lowered = answer.lower()
    return float(all(re.search(KEYWORD_BOUNDARY.format(re.escape(k.lower())), lowered)
                     for k in keywords))


def keyword_partial(answer: str, keywords: list[str]) -> float:
    """Partial credit: an optimizer can see progress between 0 and 1."""
    if not keywords:
        return 1.0
    lowered = answer.lower()
    hits = sum(bool(re.search(KEYWORD_BOUNDARY.format(re.escape(k.lower())), lowered))
               for k in keywords)
    return hits / len(keywords)


def token_f1(gold: str, pred: str) -> float:
    g, p = set(gold.lower().split()), set(pred.lower().split())
    if not g or not p:
        return 0.0
    overlap = len(g & p)
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(p), overlap / len(g)
    return 2 * precision * recall / (precision + recall)


def normalize_rubric(scores: dict[str, int], weights: dict[str, float]) -> float:
    """Weighted 1-5 rubric mapped onto 0-1, as the chapter does it."""
    if not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-9):
        raise ValueError(f"weights must sum to 1.0, got {sum(weights.values())}")
    weighted = sum(scores[k] * w for k, w in weights.items())
    return (weighted - RUBRIC_MIN) / (RUBRIC_MAX - RUBRIC_MIN)


def judge_verdict(agreement: float) -> str:
    """The chapter's trust bar for an LLM judge."""
    if agreement >= JUDGE_TRUST_GOOD:
        return "trustworthy as an optimization target"
    if agreement >= JUDGE_TRUST_FLOOR:
        return "usable, but re-measure after optimizing against it"
    return "not trustworthy: collect more labels or refine the judge signature"


def route(task_shape: str) -> tuple[str, str]:
    key = task_shape.lower().strip()
    if key in ROUTING:
        return ROUTING[key]
    for shape, value in ROUTING.items():
        if key in shape or shape in key:
            return value
    return ("unknown shape", f"choose from: {', '.join(sorted(ROUTING))}")


def gives_gradient(scores: list[float]) -> bool:
    """A metric an optimizer can climb takes more than two distinct values."""
    return len(set(scores)) > 2


def dry_run() -> None:
    # Fixed requirement, progressively better answers — what an optimizer sees.
    keywords = ["seeded", "shuffle", "validation", "stratified"]
    answers = [
        "Just split the data.",
        "Use a seeded split.",
        "Use a seeded shuffle.",
        "Use a seeded shuffle with a validation split.",
        "Use a seeded, stratified shuffle with a validation split.",
    ]
    partial_scores = [keyword_partial(a, keywords) for a in answers]
    binary_scores = [keyword_binary(a, keywords) for a in answers]
    print(f"partial credit as the answer improves: {[round(s, 2) for s in partial_scores]}")
    print(f"all-or-nothing as the answer improves: {binary_scores}")
    assert gives_gradient(partial_scores), "partial credit must give the optimizer a gradient"
    assert not gives_gradient(binary_scores), "binary metrics are the failure the chapter names"

    print(f"token_f1 exact       = {token_f1('the treaty was signed', 'the treaty was signed'):.2f}")
    print(f"token_f1 negated     = {token_f1('the treaty was signed', 'the treaty was not signed'):.2f}")
    assert token_f1('the treaty was signed', 'the treaty was not signed') > 0.8, (
        "overlap metrics score a reversed fact highly; that is the caveat"
    )

    weights = {"clarity": 0.3, "persuasiveness": 0.4, "brand_fit": 0.2, "emotional_appeal": 0.1}
    best = {"clarity": 5, "persuasiveness": 5, "brand_fit": 5, "emotional_appeal": 5}
    worst = {"clarity": 1, "persuasiveness": 1, "brand_fit": 1, "emotional_appeal": 1}
    mixed = {"clarity": 5, "persuasiveness": 2, "brand_fit": 4, "emotional_appeal": 1}
    print(f"rubric best={normalize_rubric(best, weights):.2f} "
          f"worst={normalize_rubric(worst, weights):.2f} "
          f"mixed={normalize_rubric(mixed, weights):.2f}")
    assert math.isclose(normalize_rubric(best, weights), 1.0, abs_tol=1e-9), (
        "the chapter's sanity check: the best case must land at 1.0"
    )
    assert math.isclose(normalize_rubric(worst, weights), 0.0, abs_tol=1e-9), (
        "the chapter's sanity check: the worst case must land at 0.0"
    )
    try:
        normalize_rubric(best, {"clarity": 0.5})
    except ValueError:
        pass
    else:
        raise AssertionError("weights not summing to 1.0 must be rejected")

    for agreement in (0.95, 0.83, 0.62):
        print(f"judge agreement {agreement:.0%} -> {judge_verdict(agreement)}")
    assert "not trustworthy" in judge_verdict(0.62)

    for shape in ("subjective quality", "a multi-step pipeline", "an extractive span"):
        metric, trap = route(shape)
        print(f"{shape:32s} -> {metric:36s} | {trap}")
    assert route("subjective quality")[0].startswith("calibrated")
    print("dry run OK")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Run without an LM")
    ap.add_argument("--route", metavar="SHAPE", help="Which metric for a task shape")
    args = ap.parse_args()
    if args.route:
        metric, trap = route(args.route)
        print(f"metric: {metric}\nwatch for: {trap}")
        return
    dry_run()


if __name__ == "__main__":
    main()
