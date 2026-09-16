#!/usr/bin/env python
"""Validate the DSPy API surface this skill pack teaches.

Run with an explicit DSPy wheel, for example:

    env -u UV_EXCLUDE_NEWER uv run --with dspy==3.3.1 \
        python scripts/check_dspy_surface.py --expected-version 3.3.1
"""

from __future__ import annotations

import argparse
import inspect
import sys
from collections.abc import Callable


def _params(obj: Callable) -> dict[str, inspect.Parameter]:
    return dict(inspect.signature(obj).parameters)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _assert_params(name: str, obj: Callable, required: set[str]) -> None:
    params = _params(obj)
    missing = sorted(required - set(params))
    _require(not missing, f"{name} missing params: {', '.join(missing)}")


class _EchoModule:
    def __call__(self, question: str):
        import dspy

        return dspy.Prediction(answer=question)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-version", default="3.3.1")
    args = parser.parse_args()

    import dspy

    version = getattr(dspy, "__version__", None)
    _require(
        version == args.expected_version,
        f"Expected dspy {args.expected_version}, got {version!r}",
    )

    _assert_params(
        "dspy.GEPA",
        dspy.GEPA,
        {
            "metric",
            "auto",
            "max_full_evals",
            "max_metric_calls",
            "reflection_minibatch_size",
            "candidate_selection_strategy",
            "reflection_lm",
            "skip_perfect_score",
            "instruction_proposer",
            "component_selector",
            "use_merge",
            "max_merge_invocations",
            "track_stats",
            "track_best_outputs",
            "use_mlflow",
            "gepa_kwargs",
        },
    )
    _require(
        str(_params(dspy.GEPA)["candidate_selection_strategy"].default) == "pareto",
        "dspy.GEPA candidate_selection_strategy default changed",
    )

    better_together_params = _params(dspy.BetterTogether)
    _require("metric" in better_together_params, "BetterTogether missing metric")
    _require(
        any(p.kind is inspect.Parameter.VAR_KEYWORD for p in better_together_params.values()),
        "BetterTogether should accept arbitrary named optimizers via **kwargs",
    )
    _assert_params(
        "dspy.BetterTogether.compile",
        dspy.BetterTogether.compile,
        {"student", "trainset", "valset", "strategy", "optimizer_compile_args"},
    )

    _assert_params(
        "dspy.Evaluate",
        dspy.Evaluate,
        {"devset", "metric", "num_threads", "provide_traceback", "save_as_json"},
    )
    _assert_params(
        "dspy.LM",
        dspy.LM,
        {"model", "model_type", "temperature", "max_tokens", "cache", "use_developer_role"},
    )
    _assert_params(
        "dspy.SIMBA",
        dspy.SIMBA,
        {"metric", "bsize", "num_candidates", "max_steps", "max_demos"},
    )
    _assert_params(
        "dspy.Embedder.__call__",
        dspy.Embedder.__call__,
        {"inputs", "batch_size", "caching"},
    )
    _assert_params(
        "dspy.configure_cache",
        dspy.configure_cache,
        {"restrict_pickle", "safe_types"},
    )

    # Refine/BestOfN are dspy.Module subclasses: inspect __init__ (the class
    # signature is rewritten by the Module metaclass).
    for name in ("Refine", "BestOfN"):
        _assert_params(
            f"dspy.{name}.__init__",
            getattr(dspy, name).__init__,
            {"module", "N", "reward_fn", "threshold", "fail_count"},
        )

    for name in ("Reasoning", "File", "Code"):
        _require(hasattr(dspy, name), f"dspy.{name} is missing")

    # dspy.RLM is upstream-Experimental and renamed twice in 3.3.0: the pack
    # teaches these exact names, so pin them here rather than in one example.
    _assert_params(
        "dspy.RLM.__init__",
        dspy.RLM.__init__,
        {
            "signature",
            "max_iters",
            "max_llm_calls",
            "max_output_chars",
            "verbose",
            "tools",
            "sub_lm",
            "interpreter_factory",
        },
    )
    _require(
        "max_iterations" not in _params(dspy.RLM.__init__),
        "dspy.RLM re-introduced max_iterations; the 3.3.x rename may have been reverted",
    )
    _require(
        _params(dspy.RLM.__init__)["max_output_chars"].default == 10_000,
        "dspy.RLM max_output_chars default changed",
    )
    _require(
        "interpreter" in _params(dspy.RLM.forward),
        "dspy.RLM.forward lost its caller-owned interpreter argument",
    )

    # ProgramOfThought and CodeAct took the same interpreter -> interpreter_factory
    # move in 3.3.0; the module tables in dspy-fundamentals quote both.
    for name in ("ProgramOfThought", "CodeAct"):
        cls = getattr(dspy, name)
        _assert_params(f"dspy.{name}.__init__", cls.__init__, {"signature", "max_iters", "interpreter_factory"})
        _require(
            "interpreter" in _params(cls.forward),
            f"dspy.{name}.forward lost its caller-owned interpreter argument",
        )

    # dspy.PythonInterpreter is the 3.3.x top-level alias the RLM reference cites.
    for name in ("PythonInterpreter",):
        _require(hasattr(dspy, name), f"dspy.{name} is missing")


    def gepa_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        return dspy.Prediction(score=1.0, feedback=f"ok for {pred_name or 'program'}")

    dspy.GEPA(
        metric=gepa_metric,
        auto="light",
        reflection_lm=dspy.LM("openai/gpt-5", temperature=1.0, max_tokens=32000),
    )

    def evaluate_metric(gold, pred, trace=None):
        return dspy.Prediction(score=float(pred.answer == gold.answer), feedback="ok")

    example = dspy.Example(question="ping", answer="ping").with_inputs("question")
    result = dspy.Evaluate(devset=[example], metric=evaluate_metric)(_EchoModule())
    _require(result.score == 100.0, "Evaluate did not aggregate Prediction metric score")
    _require(
        isinstance(result.results[0][2], dspy.Prediction),
        "Evaluate did not preserve Prediction metric result",
    )

    print(f"OK: DSPy {version} surface matches dspy-agent-skills expectations.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
