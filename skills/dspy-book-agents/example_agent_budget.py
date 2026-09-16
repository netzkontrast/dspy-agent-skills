"""dspy-book-agents — runnable smoke test.

The chapter's agent-budget decisions as checkable functions: fixed hops versus
agent-controlled iteration, the efficiency metric that replaces a hard cost
cap, history that must not carry trajectories, and the async rule MCP tools
impose. No LM, no MCP server, no network.

Usage:
    uv run python example_agent_budget.py --dry-run
"""

from __future__ import annotations

import argparse

DEFAULT_HOPS = 3
EFFICIENT_TOOL_CALLS = 2          # full credit at or below this
ACCEPTABLE_TOOL_CALLS = 4         # partial credit up to here
MAX_ITERS_BY_BREADTH = {"narrow": 5, "moderate": 8, "broad": 12, "open": 15}


def choose_loop(*, needs_predictable_cost: bool, question_depth_known: bool) -> tuple[str, str]:
    """The chapter's only stated reason for fixed depth: a predictable budget."""
    if needs_predictable_cost or question_depth_known:
        return ("fixed-depth module",
                f"for-loop over num_hops (default {DEFAULT_HOPS}), no early exit; cost is known up front")
    return ("agent-controlled ReAct",
            "terminates on the finish tool or max_iters; more capable, cost varies per question")


def max_iters_for(breadth: str) -> int:
    if breadth not in MAX_ITERS_BY_BREADTH:
        raise ValueError(f"breadth must be one of {sorted(MAX_ITERS_BY_BREADTH)}")
    return MAX_ITERS_BY_BREADTH[breadth]


def efficiency_score(trajectory: dict) -> float:
    """The chapter's real cost control: score it, then optimize against it."""
    calls = sum(1 for k, v in trajectory.items()
                if k.startswith("tool_name_") and v != "finish")
    if calls <= EFFICIENT_TOOL_CALLS:
        return 1.0
    if calls <= ACCEPTABLE_TOOL_CALLS:
        return 0.8
    return 0.5


def flat_trajectory(tool_names: list[str]) -> dict:
    """ReAct trajectories are a FLAT dict, not a list."""
    trajectory: dict = {}
    for i, name in enumerate(tool_names):
        trajectory[f"thought_{i}"] = f"thinking {i}"
        trajectory[f"tool_name_{i}"] = name
        trajectory[f"tool_args_{i}"] = {}
        trajectory[f"observation_{i}"] = f"result {i}"
    return trajectory


def read_trajectory(trajectory: dict, limit: int = 20) -> list[str]:
    """Iterate an index range and break on the first missing key."""
    names = []
    for i in range(limit):
        key = f"tool_name_{i}"
        if key not in trajectory:
            break
        names.append(trajectory[key])
    return names


def history_entry(result: dict) -> dict:
    """Store the answer, never the trajectory: otherwise every tool payload replays."""
    if "trajectory" in result and "answer" not in result:
        raise ValueError("refusing to store a trajectory in history; store the answer")
    return {"question": result["question"], "answer": result["answer"]}


def history_message_keys_ok(message: dict, signature_fields: set[str]) -> tuple[bool, str]:
    """dspy.History keys mirror the signature's fields, not role/content."""
    if {"role", "content"} & set(message):
        return False, "role/content is the OpenAI shape; dspy.History keys are signature field names"
    unknown = set(message) - signature_fields
    if unknown:
        return False, f"keys {sorted(unknown)} are not fields of the signature"
    return True, "keys match the signature fields"


def mcp_call_is_legal(*, tool_is_async: bool, called_synchronously: bool,
                      conversion_enabled: bool = False) -> tuple[bool, str]:
    if tool_is_async and called_synchronously and not conversion_enabled:
        return False, ("synchronous call with an async converted tool raises ValueError; "
                       "use await agent.acall(...)")
    return True, "call style is compatible"


def dedupe(existing: list[str], new: list[str]) -> list[str]:
    """Between hops, or hop two re-retrieves hop one."""
    return list(dict.fromkeys(existing + new))


def dry_run() -> None:
    for label, kwargs in (("unbounded research", dict(needs_predictable_cost=False, question_depth_known=False)),
                          ("fixed budget", dict(needs_predictable_cost=True, question_depth_known=False)),
                          ("known 2-hop lookup", dict(needs_predictable_cost=False, question_depth_known=True))):
        loop, why = choose_loop(**kwargs)
        print(f"{label:20s} -> {loop:24s} {why}")
    assert choose_loop(needs_predictable_cost=True, question_depth_known=False)[0] == "fixed-depth module"

    print(f"max_iters by breadth: {MAX_ITERS_BY_BREADTH}")
    assert max_iters_for("narrow") < max_iters_for("open")
    try:
        max_iters_for("enormous")
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown breadth must be rejected")

    for names in (["search", "finish"], ["search", "search", "lookup", "finish"],
                  ["search"] * 6 + ["finish"]):
        traj = flat_trajectory(names)
        calls = [n for n in read_trajectory(traj) if n != "finish"]
        print(f"{len(calls)} tool calls -> efficiency {efficiency_score(traj):.1f}")
    assert efficiency_score(flat_trajectory(["search", "finish"])) == 1.0
    assert efficiency_score(flat_trajectory(["s"] * 6 + ["finish"])) == 0.5

    entry = history_entry({"question": "q", "answer": "a"})
    print(f"history entry: {entry}")
    try:
        history_entry({"question": "q", "trajectory": {"observation_0": "10kb of payload"}})
    except ValueError:
        print("storing a trajectory in history is refused")
    else:
        raise AssertionError("trajectories must not enter history")

    fields = {"question", "answer"}
    ok, why = history_message_keys_ok({"question": "q", "answer": "a"}, fields)
    assert ok
    ok, why = history_message_keys_ok({"role": "user", "content": "hi"}, fields)
    print(f"OpenAI-shaped message rejected: {not ok} | {why}")
    assert not ok

    ok, why = mcp_call_is_legal(tool_is_async=True, called_synchronously=True)
    print(f"sync call on async tool: {ok} | {why}")
    assert not ok
    assert mcp_call_is_legal(tool_is_async=True, called_synchronously=False)[0]

    hop1 = ["passage-a", "passage-b"]
    merged = dedupe(hop1, ["passage-b", "passage-c"])
    print(f"hop dedup: {len(hop1)} + 2 -> {len(merged)} unique")
    assert merged == ["passage-a", "passage-b", "passage-c"]
    print("dry run OK")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Run without an LM or MCP server")
    ap.parse_args()
    dry_run()


if __name__ == "__main__":
    main()
