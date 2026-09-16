"""dspy-rlm-hooks — runnable smoke test.

Builds the four lifecycle hooks and a speculation setup, and checks the two
rules that silently cost you correctness or speed: hooks must be enabled
before speculation, and a speculated tool must be pure.

The dry run works with or without the package installed. Without it, the
deterministic rules are still exercised; with it, the real API surface is
asserted too, so upstream drift fails here.

Usage:
    uv run python example_rlm_hooks.py --dry-run
    uv run --with dspy-rlm-hooks python example_rlm_hooks.py --dry-run
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect

PACKAGE = "dspy_rlm_hooks"
MIN_PYTHON = (3, 12)
EXPECTED_SPECULATION_DEFAULTS = {
    "max_inflight": 8,
    "max_dispatches_per_turn": 2048,
    "speculate_llm_query": True,
    "speculate_user_tools": False,
    "timeout_s": 5.0,
    "streaming": True,
}
REQUIRED_RLM_INTERNALS = (
    "_execute_iteration", "_aexecute_iteration",
    "_process_execution_result", "generate_action", "verbose",
)


def package_available() -> bool:
    return importlib.util.find_spec(PACKAGE) is not None


def composition_is_correct(order: list[str]) -> tuple[bool, str]:
    """enable_rlm_hooks must run before enable_rlm_speculation."""
    if "enable_rlm_speculation" not in order:
        return True, "No speculation; order does not matter."
    if "enable_rlm_hooks" not in order:
        return True, "Speculation without hooks is fine."
    if order.index("enable_rlm_hooks") < order.index("enable_rlm_speculation"):
        return True, "Hooks first, then speculation."
    return False, ("Speculation enabled before hooks: enable_rlm_hooks overwrites "
                   "_execute_code, so speculation is silently inactive.")


def may_speculate(*, pure: bool, writes: bool, charges: bool, reads_mutable: bool) -> tuple[bool, str]:
    """A speculated tool runs early, maybe twice, maybe for a branch never taken."""
    if writes:
        return False, "Tool writes state; a discarded speculation would leave the write behind."
    if charges:
        return False, "Tool costs money per call; evicted speculations are paid for and thrown away."
    if reads_mutable:
        return False, "Tool reads mutable state; the speculated read may be stale by claim time."
    if not pure:
        return False, "Not declared pure; speculate() refuses speculatable without pure."
    return True, "Pure and side-effect free; safe to speculate."


def build_hooks():
    """The four lifecycle hooks, with the real positional signatures."""
    from dspy_rlm_hooks import (PostExecutionOutput, PostIterationOutput,
                                PreExecutionOutput, PreIterationOutput)

    log: list[str] = []

    def pre_iteration(iteration, variables, history, input_args):
        return PreIterationOutput(extra_vars={"iteration_seen": iteration},
                                  prompt_context="Prefer already-loaded variables.")

    def pre_execution(iteration, code, variables, history, input_args):
        return PreExecutionOutput(code=code)          # rewrite point

    def post_execution(iteration, code, result, variables, history, input_args):
        return PostExecutionOutput(result=result)     # redact/cap point

    def post_iteration(iteration, pred, code, result, history):
        log.append(f"iteration {iteration}")
        return PostIterationOutput(history=history, stop=False)

    return {"pre_iteration_hook": pre_iteration, "pre_execution_hook": pre_execution,
            "post_execution_hook": post_execution, "post_iteration_hook": post_iteration}, log


def assert_api_surface() -> None:
    """Assert the surface this skill teaches, when the package is installed."""
    import dspy_rlm_hooks as pkg
    from dspy_rlm_hooks import (PreIterationOutput, SpeculationConfig,
                                enable_rlm_hooks, enable_rlm_speculation, speculate)

    hooks_params = inspect.signature(enable_rlm_hooks).parameters
    for stage in ("pre_iteration_hook", "pre_execution_hook",
                  "post_execution_hook", "post_iteration_hook"):
        assert stage in hooks_params, f"enable_rlm_hooks lost {stage}"
        assert hooks_params[stage].kind is inspect.Parameter.KEYWORD_ONLY

    assert set(PreIterationOutput.model_fields) == {
        "extra_vars", "python_code", "persistent_python_code", "prompt_context"
    }, "PreIterationOutput fields changed"

    spec_params = inspect.signature(enable_rlm_speculation).parameters
    for field, expected in EXPECTED_SPECULATION_DEFAULTS.items():
        assert spec_params[field].default == expected, (
            f"enable_rlm_speculation.{field} default moved from {expected}"
        )
    assert set(EXPECTED_SPECULATION_DEFAULTS) <= set(SpeculationConfig.model_fields)

    # speculate() must refuse an impure tool.
    try:
        speculate(lambda x: x, speculatable=True, pure=False)
    except ValueError:
        pass
    else:
        raise AssertionError("speculate() accepted speculatable without pure")

    for name in ("enable_rlm_hooks", "disable_rlm_hooks", "enable_rlm_speculation",
                 "disable_rlm_speculation", "speculative", "SpeculationPolicy"):
        assert hasattr(pkg, name), f"{name} is no longer exported"


def dry_run() -> None:
    ok, why = composition_is_correct(["enable_rlm_hooks", "enable_rlm_speculation"])
    print(f"hooks -> speculation: {ok} | {why}")
    assert ok

    ok, why = composition_is_correct(["enable_rlm_speculation", "enable_rlm_hooks"])
    print(f"speculation -> hooks: {ok} | {why}")
    assert not ok, "the reversed order must be rejected"

    cases = [
        ("pure lookup", dict(pure=True, writes=False, charges=False, reads_mutable=False)),
        ("writes a row", dict(pure=False, writes=True, charges=False, reads_mutable=False)),
        ("paid API", dict(pure=True, writes=False, charges=True, reads_mutable=False)),
        ("reads a queue", dict(pure=True, writes=False, charges=False, reads_mutable=True)),
    ]
    for label, kwargs in cases:
        allowed, reason = may_speculate(**kwargs)
        print(f"{label:16s} speculate={str(allowed):5s} | {reason}")
    assert may_speculate(**cases[0][1])[0]
    assert not any(may_speculate(**kw)[0] for _, kw in cases[1:])

    print(f"RLM internals this package patches: {', '.join(REQUIRED_RLM_INTERNALS)}")

    if package_available():
        assert_api_surface()
        hooks, log = build_hooks()
        assert set(hooks) == {"pre_iteration_hook", "pre_execution_hook",
                              "post_execution_hook", "post_iteration_hook"}
        print("package installed: API surface asserted, four hooks constructed")
    else:
        print(f"package not installed: skipped live API assertions "
              f"(pip install dspy-rlm-hooks, needs Python >= {MIN_PYTHON[0]}.{MIN_PYTHON[1]})")
    print("dry run OK")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Skip all LM calls")
    args = ap.parse_args()
    if args.dry_run:
        dry_run()
        return
    import dspy
    from dspy_rlm_hooks import disable_rlm_hooks, enable_rlm_hooks

    rlm = dspy.RLM("question -> answer")
    hooks, log = build_hooks()
    enable_rlm_hooks(rlm, **hooks)
    print("hooks enabled; run the RLM with a configured LM to see the log fill")
    disable_rlm_hooks(rlm)
    print(f"hooks disabled. iterations logged: {len(log)}")
    print("Benchmark without an LM: python -m dspy_rlm_hooks.benchmark --variants default spec")


if __name__ == "__main__":
    main()
