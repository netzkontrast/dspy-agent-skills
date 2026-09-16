"""dspy-book-eight-steps — runnable smoke test.

The eight steps as an ordered checklist with prerequisites, so skipping one is
detectable, plus the dual-mode metric contract and the best-of-N calling
convention that differs from it. No LM, no network.

Usage:
    uv run python example_eight_steps.py --dry-run
    uv run python example_eight_steps.py --next signatures,modules,dataset
"""

from __future__ import annotations

import argparse
import inspect

METRIC_ARITY_FOR_OPTIMIZER = 5      # (example, pred, trace, pred_name, pred_trace)

STEPS = [
    ("signatures", (), "the task, and its judge if the metric is a model"),
    ("modules", ("signatures",), "Predict or ChainOfThought over each signature"),
    ("explore", ("modules",), "run a handful by hand; end with inspect_history"),
    ("dataset", ("signatures",), "dspy.Example objects with a seeded split"),
    ("metrics", ("dataset",), "one per thing measured; five-argument form"),
    ("baseline", ("metrics", "modules"), "a number to beat, before any compile"),
    ("optimize", ("baseline",), "judge first, then the task"),
    ("iterate", ("optimize",), "best-of-N, and a run on a different provider"),
]
STEP_INDEX = {name: i for i, (name, _, _) in enumerate(STEPS)}


def next_step(done: set[str]) -> tuple[str, str] | None:
    for name, requires, produces in STEPS:
        if name in done:
            continue
        missing = [r for r in requires if r not in done]
        if missing:
            return name, f"blocked: needs {', '.join(missing)} first"
        return name, produces
    return None


def violations(done: set[str]) -> list[str]:
    """Which steps were completed before their prerequisites."""
    out = []
    for name, requires, _ in STEPS:
        if name in done:
            for r in requires:
                if r not in done:
                    out.append(f"{name} done without {r}")
    return out


def metric_is_optimizer_ready(fn) -> tuple[bool, str]:
    """A three-argument metric must be rewritten before reflective optimization."""
    params = list(inspect.signature(fn).parameters)
    if len(params) >= METRIC_ARITY_FOR_OPTIMIZER:
        return True, f"{len(params)} parameters; usable by Evaluate and an optimizer"
    return False, (f"{len(params)} parameters; widen to "
                   f"(example, pred, trace=None, pred_name=None, pred_trace=None) now, it costs nothing")


def three_arg_metric(example, pred, trace=None):
    return 1.0


class Prediction:
    """Stand-in for dspy.Prediction so this example needs no dspy import.

    The real metric returns `dspy.Prediction(score=..., feedback=...)`. It must
    not return a plain mapping: dspy.Evaluate aggregates with sum(), which
    raises TypeError on one.
    """

    def __init__(self, score: float, feedback: str = "") -> None:
        self.score, self.feedback = score, feedback


def dual_mode_metric(example, pred, trace=None, pred_name=None, pred_trace=None):
    """Plain number for Evaluate; score plus feedback when an optimizer asks."""
    score = 1.0
    if pred_name:
        return Prediction(score=score, feedback=getattr(example, "notes", ""))
    return score


def best_of_n_reward_shape_ok(fn) -> tuple[bool, str]:
    """BestOfN.reward_fn takes (kwargs_dict, pred) — NOT the metric signature."""
    params = list(inspect.signature(fn).parameters)
    if len(params) == 2:
        return True, "two positional parameters: (kwargs, pred)"
    return False, (f"{len(params)} parameters; BestOfN passes a kwargs dict and a prediction, "
                   f"not the metric's argument list")


def dry_run() -> None:
    done: set[str] = set()
    order = []
    while (step := next_step(done)) is not None:
        name, note = step
        assert "blocked" not in note, f"walking in order should never block: {name} {note}"
        order.append(name)
        done.add(name)
    print(f"walked in order: {' -> '.join(order)}")
    assert order == [name for name, _, _ in STEPS]

    skipped = {"signatures", "modules", "dataset", "metrics", "optimize"}
    print(f"optimize without a baseline -> {violations(skipped)}")
    assert violations(skipped) == ["optimize done without baseline"], (
        "skipping the baseline is the failure this checklist exists to catch"
    )
    name, note = next_step({"signatures", "modules"})
    print(f"after signatures+modules, next is {name!r}: {note}")

    ok, why = metric_is_optimizer_ready(three_arg_metric)
    print(f"three-arg metric ready: {ok} | {why[:74]}")
    assert not ok
    ok, why = metric_is_optimizer_ready(dual_mode_metric)
    print(f"five-arg metric ready : {ok} | {why}")
    assert ok

    class _Ex:
        notes = "model confused an account question for a technical one"
    assert dual_mode_metric(_Ex(), None) == 1.0, "plain evaluation returns a number"
    optimizer_view = dual_mode_metric(_Ex(), None, pred_name="predictor")
    assert isinstance(optimizer_view, Prediction) and optimizer_view.feedback, (
        "when an optimizer asks, the dataset's notes become the feedback"
    )
    assert not isinstance(optimizer_view, dict), (
        "a mapping return crashes dspy.Evaluate's sum() aggregation"
    )
    print(f"optimizer sees feedback: {optimizer_view.feedback[:52]}...")

    ok, why = best_of_n_reward_shape_ok(lambda kwargs, pred: 1.0)
    print(f"best-of-N reward shape: {ok} | {why}")
    assert ok
    assert not best_of_n_reward_shape_ok(dual_mode_metric)[0], (
        "passing the metric straight to BestOfN is the documented mismatch"
    )
    print("dry run OK")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Run without an LM")
    ap.add_argument("--next", metavar="DONE", help="Comma-separated completed steps")
    args = ap.parse_args()
    if args.next:
        done = {s.strip() for s in args.next.split(",") if s.strip()}
        unknown = done - set(STEP_INDEX)
        if unknown:
            raise SystemExit(f"unknown steps: {sorted(unknown)}; valid: {list(STEP_INDEX)}")
        for v in violations(done):
            print(f"out of order: {v}")
        step = next_step(done)
        print(f"next: {step[0]} — {step[1]}" if step else "all eight steps complete")
        return
    dry_run()


if __name__ == "__main__":
    main()
