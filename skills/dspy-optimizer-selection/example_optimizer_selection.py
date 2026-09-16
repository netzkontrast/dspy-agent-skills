"""dspy-optimizer-selection — runnable smoke test.

Implements the selection matrix as a deterministic `recommend()` function and,
in the live path, constructs the recommended optimizer. The dry run exercises
the routing table on several situations and asserts the constructor and
`compile` signatures the skill teaches, so an upstream API change fails here
instead of in a user's compile run.

Usage:
    uv run python example_optimizer_selection.py --dry-run
    OPENAI_API_KEY=... uv run python example_optimizer_selection.py
"""

from __future__ import annotations

import argparse
import inspect
import os
from dataclasses import dataclass
from typing import Literal

MetricShape = Literal["scalar", "feedback", "none"]

# Trainset-size thresholds from the skill's "Choosing by trainset size" table.
TINY_TRAINSET = 10
SMALL_TRAINSET = 50
LARGE_TRAINSET = 100


@dataclass(frozen=True)
class Situation:
    """What you actually have, before choosing anything."""

    n_examples: int
    metric_shape: MetricShape = "scalar"
    baseline_measured: bool = True
    prompt_plateaued: bool = False
    finetunable_lm: bool = False
    candidate_programs: int = 0
    per_input_demos: bool = False


@dataclass(frozen=True)
class Choice:
    optimizer: str
    why: str


def recommend(s: Situation) -> Choice:
    """Deterministic routing: the cheapest optimizer whose needs are met."""
    if not s.baseline_measured:
        return Choice("none", "Measure the uncompiled program first; nothing to compare against.")
    if s.candidate_programs > 1:
        return Choice("dspy.Ensemble", f"{s.candidate_programs} candidates already exist; combine instead of recompiling.")
    if s.metric_shape == "none" or s.n_examples < TINY_TRAINSET:
        return Choice("dspy.LabeledFewShot", "Too few examples or no metric; use labeled demos as a floor.")
    if s.prompt_plateaued and s.finetunable_lm:
        return Choice("dspy.BetterTogether", "Prompts plateaued and a fine-tunable LM is available.")
    if s.prompt_plateaued:
        return Choice("dspy.MIPROv2", "Prompts plateaued but no fine-tunable LM; search instructions and demos.")
    if s.per_input_demos:
        return Choice("dspy.KNNFewShot", "Inputs vary enough that each one wants its own demos.")
    if s.metric_shape == "feedback":
        return Choice("dspy.GEPA", "Failures carry text feedback; reflection has something to read.")
    if s.n_examples >= LARGE_TRAINSET:
        return Choice("dspy.MIPROv2", "Enough examples for Bayesian instruction and demo search.")
    if s.n_examples >= SMALL_TRAINSET:
        return Choice("dspy.BootstrapFewShotWithRandomSearch", "Enough examples to search several demo sets.")
    return Choice("dspy.BootstrapFewShot", "Small trainset; bootstrap demos with metric filtering.")


def build(name: str, metric, trainset=None, reflection_lm=None):
    """Construct the recommended optimizer with the arguments the skill teaches."""
    import dspy

    if name == "dspy.LabeledFewShot":
        return dspy.LabeledFewShot(k=16)
    if name == "dspy.BootstrapFewShot":
        return dspy.BootstrapFewShot(metric=metric, max_bootstrapped_demos=4, max_labeled_demos=16)
    if name == "dspy.BootstrapFewShotWithRandomSearch":
        return dspy.BootstrapFewShotWithRandomSearch(metric=metric)
    if name == "dspy.KNNFewShot":
        return dspy.KNNFewShot(k=3, trainset=trainset or [],
                               vectorizer=dspy.Embedder("openai/text-embedding-3-small"))
    if name == "dspy.MIPROv2":
        return dspy.MIPROv2(metric=metric, auto="light", seed=9)
    if name == "dspy.SIMBA":
        return dspy.SIMBA(metric=metric, bsize=32, max_steps=8)
    if name == "dspy.GEPA":
        return dspy.GEPA(metric=metric, auto="light", reflection_lm=reflection_lm)
    if name == "dspy.Ensemble":
        return dspy.Ensemble(reduce_fn=dspy.majority)
    if name == "dspy.BetterTogether":
        return dspy.BetterTogether(
            metric=metric,
            bootstrap=dspy.BootstrapFewShotWithRandomSearch(metric=metric),
            gepa=dspy.GEPA(metric=metric, auto="light", reflection_lm=reflection_lm),
        )
    raise ValueError(f"no constructor for {name!r}")


def assert_api_surface() -> None:
    """Fail loudly if the signatures this skill teaches have drifted."""
    import dspy

    assert dspy.BootstrapRS is dspy.BootstrapFewShotWithRandomSearch, "BootstrapRS alias is gone"
    assert "k" in inspect.signature(dspy.LabeledFewShot.__init__).parameters
    assert "metric" not in inspect.signature(dspy.LabeledFewShot.__init__).parameters, (
        "LabeledFewShot gained a metric parameter; the skill says it has none"
    )
    # SIMBA is keyword-only and carries its seed on compile(), not __init__.
    simba_init = inspect.signature(dspy.SIMBA.__init__).parameters
    assert all(p.kind is p.KEYWORD_ONLY for n, p in simba_init.items() if n != "self")
    assert "seed" in inspect.signature(dspy.SIMBA.compile).parameters
    assert "seed" not in simba_init
    # Ensemble.compile takes a list of programs, not (student, trainset).
    ensemble_compile = list(inspect.signature(dspy.Ensemble.compile).parameters)
    assert "programs" in ensemble_compile and "trainset" not in ensemble_compile
    # KNNFewShot takes the trainset up front and compiles without one.
    assert "trainset" in inspect.signature(dspy.KNNFewShot.__init__).parameters
    assert "trainset" not in inspect.signature(dspy.KNNFewShot.compile).parameters
    # BetterTogether takes arbitrary named optimizers, not a fixed pair.
    bt = inspect.signature(dspy.BetterTogether.__init__).parameters
    assert any(p.kind is p.VAR_KEYWORD for p in bt.values()), "BetterTogether lost **optimizers"


SITUATIONS = [
    ("no baseline yet", Situation(n_examples=200, baseline_measured=False)),
    ("6 examples", Situation(n_examples=6)),
    ("30 examples, scalar metric", Situation(n_examples=30)),
    ("80 examples, scalar metric", Situation(n_examples=80)),
    ("300 examples, scalar metric", Situation(n_examples=300)),
    ("40 examples, rich feedback", Situation(n_examples=40, metric_shape="feedback")),
    ("varied inputs", Situation(n_examples=60, per_input_demos=True)),
    ("plateaued, no finetune", Situation(n_examples=300, prompt_plateaued=True)),
    ("plateaued, finetunable LM", Situation(n_examples=300, prompt_plateaued=True, finetunable_lm=True)),
    ("3 good candidates", Situation(n_examples=300, candidate_programs=3)),
]


def dry_run() -> None:
    for label, situation in SITUATIONS:
        choice = recommend(situation)
        print(f"{label:28s} -> {choice.optimizer:38s} {choice.why}")
    assert recommend(Situation(n_examples=300, baseline_measured=False)).optimizer == "none"
    assert recommend(Situation(n_examples=40, metric_shape="feedback")).optimizer == "dspy.GEPA"
    assert recommend(Situation(n_examples=6)).optimizer == "dspy.LabeledFewShot"
    assert_api_surface()
    import dspy

    # MIPROv2 and GEPA validate their LMs at construction time, so a stub LM must
    # exist even in a dry run. Constructing dspy.LM makes no network call.
    stub = dspy.LM("openai/gpt-4o-mini", api_key="dry-run")
    dspy.configure(lm=stub)
    metric = lambda gold, pred, trace=None, pred_name=None, pred_trace=None: 1.0
    for name in ("dspy.LabeledFewShot", "dspy.BootstrapFewShot",
                 "dspy.BootstrapFewShotWithRandomSearch", "dspy.MIPROv2",
                 "dspy.SIMBA", "dspy.GEPA", "dspy.Ensemble", "dspy.BetterTogether"):
        built = build(name, metric, reflection_lm=stub)
        print(f"constructed {name}: {type(built).__name__}")
    print("dry run OK")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Skip all LM calls")
    ap.add_argument("--model", default=os.environ.get("DSPY_MODEL", "openai/gpt-4o-mini"))
    args = ap.parse_args()
    if args.dry_run:
        dry_run()
        return
    import dspy

    dspy.configure(lm=dspy.LM(args.model), track_usage=True)
    situation = Situation(n_examples=120, metric_shape="scalar")
    choice = recommend(situation)
    print(f"recommended: {choice.optimizer} — {choice.why}")
    metric = lambda gold, pred, trace=None, pred_name=None, pred_trace=None: float(
        gold.answer.lower() in str(pred.answer).lower()
    )
    print("constructed:", type(build(choice.optimizer, metric)).__name__)
    print("Run .compile(program, trainset=trainset) with a real trainset to optimize.")


if __name__ == "__main__":
    main()
