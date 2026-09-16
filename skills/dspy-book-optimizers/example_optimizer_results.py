"""dspy-book-optimizers — runnable smoke test.

The chapter's measured results as queryable data, plus the three checks the
table exists to teach: an optimizer can lose to the baseline, validation gain
can evaporate on test, and cost does not rank quality. No LM, no network.

Usage:
    uv run python example_optimizer_results.py --dry-run
    uv run python example_optimizer_results.py --rank uplift
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

BASELINE_TEST = 53.75
WEIGHT_OPTIMIZER_BASELINE = 51.25      # local student, different starting point
OVERFIT_GAP_THRESHOLD = 5.0            # validation minus test, in points


@dataclass(frozen=True)
class Run:
    name: str
    validation: float
    test: float
    cost_usd: float
    seconds: float
    cost_is_lower_bound: bool = False
    baseline: float = BASELINE_TEST

    @property
    def uplift(self) -> float:
        return round(self.test - self.baseline, 2)

    @property
    def overfit_gap(self) -> float:
        return round(self.validation - self.test, 2)

    @property
    def helped(self) -> bool:
        return self.test > self.baseline


RUNS = [
    Run("GEPA", 80.00, 80.00, 0.5824, 616.7),
    Run("KNNFewShot", 71.67, 72.50, 0.0, 0.0),
    Run("BootstrapFinetune", 70.00, 70.00, 0.8651, 1026.3, baseline=WEIGHT_OPTIMIZER_BASELINE),
    Run("Ensemble", 65.00, 70.00, 0.8881, 1151.2),
    Run("LabeledFewShot", 63.33, 67.50, 0.0, 0.0),
    Run("BootstrapFewShot", 66.67, 67.50, 0.0030, 4.5),
    Run("MIPROv2", 76.67, 66.25, 0.3052, 270.8, cost_is_lower_bound=True),
    Run("BootstrapRS", 61.67, 65.00, 0.8766, 1119.1),
    Run("BetterTogether", 60.00, 65.00, 0.8445, 1740.0, baseline=WEIGHT_OPTIMIZER_BASELINE),
    Run("COPRO", 53.33, 50.00, 0.0732, 861.1, cost_is_lower_bound=True),
    Run("SIMBA", 51.67, 47.50, 1.1413, 321.3),
]


def free_runs() -> list[Run]:
    return [r for r in RUNS if r.cost_usd == 0.0]


def regressions() -> list[Run]:
    return [r for r in RUNS if not r.helped]


def overfitters(threshold: float = OVERFIT_GAP_THRESHOLD) -> list[Run]:
    return [r for r in RUNS if r.overfit_gap > threshold]


def cost_ranks_quality() -> bool:
    """Does spending more predict a better result? The table says no."""
    by_cost = [r.name for r in sorted(RUNS, key=lambda r: -r.cost_usd)]
    by_test = [r.name for r in sorted(RUNS, key=lambda r: -r.test)]
    return by_cost == by_test


def best_value() -> Run:
    """Most uplift per dollar; free runs win by construction, so exclude them."""
    paid = [r for r in RUNS if r.cost_usd > 0 and r.helped]
    return max(paid, key=lambda r: r.uplift / r.cost_usd)


def dry_run() -> None:
    print(f"baseline test accuracy: {BASELINE_TEST}%  ({len(RUNS)} optimizer runs recorded)")

    losers = regressions()
    print(f"optimizers that made it WORSE: {[(r.name, r.uplift) for r in losers]}")
    assert {r.name for r in losers} == {"COPRO", "SIMBA"}, "the regression pair is the lesson"
    worst = min(RUNS, key=lambda r: r.test)
    assert worst.name == "SIMBA" and worst.cost_usd == max(r.cost_usd for r in RUNS), (
        "the worst result was also the most expensive run"
    )
    print(f"worst result ({worst.name}) was also the costliest at ${worst.cost_usd:.2f}")

    free = free_runs()
    print(f"free optimizers: {[(r.name, r.test) for r in free]}")
    beaten_by_free = [r.name for r in RUNS
                      if r.cost_usd > 0 and r.test < max(f.test for f in free)]
    print(f"paid optimizers beaten by the best free one: {len(beaten_by_free)}")
    assert len(beaten_by_free) >= 5, "free optimizers out-performing paid ones is the point"

    gaps = overfitters()
    print(f"validation-to-test overfit (> {OVERFIT_GAP_THRESHOLD} pts): "
          f"{[(r.name, r.overfit_gap) for r in gaps]}")
    assert any(r.name == "MIPROv2" for r in gaps)
    gepa = next(r for r in RUNS if r.name == "GEPA")
    assert gepa.overfit_gap == 0.0, "GEPA was the only run where validation matched test"

    assert not cost_ranks_quality(), "if cost ranked quality there would be nothing to teach"
    value = best_value()
    print(f"best uplift per dollar among paid runs: {value.name} "
          f"({value.uplift:+.2f} pts for ${value.cost_usd:.2f})")
    assert value.name == "BootstrapFewShot", "a third of a cent bought +13.75 points"
    print("dry run OK")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Verify the recorded results")
    ap.add_argument("--rank", choices=["test", "uplift", "cost", "time"],
                    help="Print the runs ranked by a column")
    args = ap.parse_args()
    if args.rank:
        key = {"test": lambda r: -r.test, "uplift": lambda r: -r.uplift,
               "cost": lambda r: r.cost_usd, "time": lambda r: r.seconds}[args.rank]
        for r in sorted(RUNS, key=key):
            flag = " (lower bound)" if r.cost_is_lower_bound else ""
            print(f"{r.name:20s} test={r.test:6.2f} uplift={r.uplift:+6.2f} "
                  f"${r.cost_usd:.4f}{flag} {r.seconds:7.1f}s")
        return
    dry_run()


if __name__ == "__main__":
    main()
