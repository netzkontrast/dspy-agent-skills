"""dspy-book-modules — runnable smoke test.

Routes a task to the right module and the right adapter, and encodes the two
configuration traps: a Pydantic output without JSONAdapter, and majority
voting without completions to vote over. No LM, no network.

Usage:
    uv run python example_module_choice.py --dry-run
    uv run python example_module_choice.py --task "add up these invoice lines"
"""

from __future__ import annotations

import argparse

RLM_CONTEXT_THRESHOLD_TOKENS = 50_000
CODEACT_TOOL_MUST_BE_PLAIN_FUNCTION = True

MODULE_FOR = {
    "arithmetic": ("dspy.ProgramOfThought", "execution beats reasoning; no tools needed"),
    "tools_in_code": ("dspy.CodeAct", "writes Python that calls your tools; tools must be plain functions"),
    "long_context": ("dspy.RLM", f"context beyond ~{RLM_CONTEXT_THRESHOLD_TOKENS:,} tokens; REPL plus a cheap sub-model"),
    "tool_loop": ("dspy.ReAct", "think/act/observe; accepts functions, callables or dspy.Tool"),
    "single_step": ("dspy.Predict", "one structured mapping"),
    "single_step_reasoned": ("dspy.ChainOfThought", "one mapping that benefits from visible reasoning"),
}


def choose_module(*, needs_tools: bool, needs_execution: bool,
                  context_tokens: int, wants_reasoning: bool = True) -> tuple[str, str]:
    if context_tokens >= RLM_CONTEXT_THRESHOLD_TOKENS:
        return MODULE_FOR["long_context"]
    if needs_execution and needs_tools:
        return MODULE_FOR["tools_in_code"]
    if needs_execution:
        return MODULE_FOR["arithmetic"]
    if needs_tools:
        return MODULE_FOR["tool_loop"]
    return MODULE_FOR["single_step_reasoned" if wants_reasoning else "single_step"]


def choose_adapter(*, output_is_pydantic: bool, model_is_reasoning: bool,
                   want_readable_transcript: bool = False) -> tuple[str, str]:
    """DSPy does NOT pick JSONAdapter automatically for a Pydantic output."""
    if model_is_reasoning:
        return ("dspy.TwoStepAdapter",
                "long chain-of-thought breaks field parsing; extract with a small model")
    if output_is_pydantic:
        return ("dspy.JSONAdapter",
                "schema-validated structured output; NOT selected automatically")
    if want_readable_transcript:
        return ("dspy.XMLAdapter", "human-readable field tags")
    return ("dspy.ChatAdapter", "the default; bracket field markers")


def codeact_tool_ok(tool) -> tuple[bool, str]:
    """CodeAct injects tools into the interpreter, so they must be plain functions."""
    import types
    if isinstance(tool, types.FunctionType):
        return True, "plain function"
    return False, f"{type(tool).__name__} is not a plain function; CodeAct requires one (ReAct accepts it)"


def majority_is_meaningful(n_completions: int) -> tuple[bool, str]:
    """dspy.majority counts completions; without config={'n': ...} there is one."""
    if n_completions <= 1:
        return False, "only one completion: voting is a no-op, pass config={'n': 5, 'temperature': 1.0}"
    return True, f"{n_completions} completions to vote over"


def vote(values: list[str], normalize=None) -> str:
    """Aliases must be folded before counting or the vote splits."""
    normalize = normalize or (lambda v: v)
    counts: dict[str, int] = {}
    for value in values:
        key = normalize(value)
        counts[key] = counts.get(key, 0) + 1
    return max(counts, key=counts.get)


class _NotAFunction:
    def __call__(self, x): return x


def dry_run() -> None:
    cases = [
        ("sum these invoice lines", dict(needs_tools=False, needs_execution=True, context_tokens=2_000)),
        ("look up prices then compute", dict(needs_tools=True, needs_execution=True, context_tokens=2_000)),
        ("answer from this 200k-token dump", dict(needs_tools=False, needs_execution=False, context_tokens=200_000)),
        ("search docs and answer", dict(needs_tools=True, needs_execution=False, context_tokens=3_000)),
        ("classify this sentence", dict(needs_tools=False, needs_execution=False, context_tokens=200)),
    ]
    for label, kwargs in cases:
        module, why = choose_module(**kwargs)
        print(f"{label:34s} -> {module:22s} {why}")
    assert choose_module(needs_tools=False, needs_execution=False, context_tokens=200_000)[0] == "dspy.RLM"
    assert choose_module(needs_tools=True, needs_execution=True, context_tokens=100)[0] == "dspy.CodeAct"

    adapters = [
        ("plain text fields", dict(output_is_pydantic=False, model_is_reasoning=False)),
        ("Pydantic output", dict(output_is_pydantic=True, model_is_reasoning=False)),
        ("reasoning model", dict(output_is_pydantic=True, model_is_reasoning=True)),
    ]
    for label, kwargs in adapters:
        adapter, why = choose_adapter(**kwargs)
        print(f"{label:34s} -> {adapter:22s} {why}")
    assert choose_adapter(output_is_pydantic=True, model_is_reasoning=False)[0] == "dspy.JSONAdapter", (
        "a Pydantic output needs JSONAdapter configured explicitly"
    )

    def real_tool(x: int) -> int:
        """A plain function."""
        return x
    ok, why = codeact_tool_ok(real_tool)
    print(f"plain function as CodeAct tool: {ok} | {why}")
    assert ok
    ok, why = codeact_tool_ok(_NotAFunction())
    print(f"callable object as CodeAct tool: {ok} | {why}")
    assert not ok

    ok, why = majority_is_meaningful(1)
    print(f"voting with 1 completion: {ok} | {why}")
    assert not ok
    assert majority_is_meaningful(5)[0]

    raw = ["NYC", "nyc", "New York City", "Boston"]
    print(f"vote without normalize: {vote(raw)!r}")
    print(f"vote with normalize   : "
          f"{vote(raw, normalize=lambda v: {'nyc': 'new york city'}.get(v.strip().lower(), v.strip().lower()))!r}")
    assert vote(raw, normalize=lambda v: {'nyc': 'new york city'}.get(v.strip().lower(), v.strip().lower())) == "new york city", (
        "without normalization the aliases split the vote"
    )
    print("dry run OK")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Run without an LM")
    ap.add_argument("--task", help="Describe a task to route it")
    args = ap.parse_args()
    if args.task:
        t = args.task.lower()
        module, why = choose_module(
            needs_tools=any(w in t for w in ("search", "look up", "fetch", "call")),
            needs_execution=any(w in t for w in ("add", "sum", "compute", "calculate", "sort", "count")),
            context_tokens=200_000 if any(w in t for w in ("long", "whole", "entire", "dump")) else 2_000)
        print(f"{module}: {why}")
        return
    dry_run()


if __name__ == "__main__":
    main()
