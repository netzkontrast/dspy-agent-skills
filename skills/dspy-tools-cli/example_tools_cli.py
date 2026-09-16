"""dspy-tools-cli — runnable smoke test.

Encodes the two things that decide whether a dspytools command will work:
which services it needs, and which optimizer name is valid. The dry run
verifies the routing and prints the first-run command sequence. It never
invokes the CLI, so it is safe without the package or any service.

Usage:
    uv run python example_tools_cli.py --dry-run
    uv run python example_tools_cli.py --plan qa
"""

from __future__ import annotations

import argparse
import shutil

FALKORDB_PORT = 6379
MLFLOW_PORT = 5000
LLAMA_CPP_PORT = 8080

# Command group -> the service it needs, or None.
SERVICE_REQUIREMENTS = {
    "configure": None, "signature": None, "module": None, "run": None,
    "compile": None, "evaluate": None, "data": None, "doctor": None,
    "generate": None, "export": None, "compare": None, "inspect": None,
    "skills": None, "agent": None, "tool": None, "pipeline": None,
    "distill": None, "self": None, "gfl": None, "mcp": None,
    "graph": "falkordb", "memory": "falkordb",
    "lora": "llama-cpp-server", "server": "llama-cpp-server",
}
SERVICE_HINTS = {
    "falkordb": f"docker compose -f docker-compose.redis.yml up -d   # port {FALKORDB_PORT}",
    "llama-cpp-server": f"start llama-cpp-server on port {LLAMA_CPP_PORT}",
}
REGISTRY_OPTIMIZERS = (
    "knn", "mipro", "gepa", "copro", "simba", "bootstrap-few-shot",
    "bootstrap-few-shot-random", "bootstrap-few-shot-optuna",
    "labeled-few-shot", "infer-rules",
)
HANDWRITTEN_OPTIMIZERS = (
    "submit", "flex", "better-together", "ensemble", "finetune",
    "gfl", "grpo", "avatar", "distill",
)
TEACHER_REQUIRED = {"gepa", "distill", "finetune"}


def service_for(group: str) -> str | None:
    if group not in SERVICE_REQUIREMENTS:
        raise KeyError(f"unknown command group {group!r}")
    return SERVICE_REQUIREMENTS[group]


def can_run(group: str, *, running_services: set[str]) -> tuple[bool, str]:
    needed = service_for(group)
    if needed is None:
        return True, "no service required"
    if needed in running_services:
        return True, f"{needed} is up"
    return False, f"needs {needed}: {SERVICE_HINTS[needed]}"


def validate_optimizer(name: str) -> tuple[bool, str]:
    if name in REGISTRY_OPTIMIZERS:
        note = "teacher LM required" if name in TEACHER_REQUIRED else "no teacher needed"
        return True, f"registry optimizer; {note}"
    if name in HANDWRITTEN_OPTIMIZERS:
        note = "teacher LM required" if name in TEACHER_REQUIRED else "no teacher needed"
        return True, f"hand-written command; {note}"
    return False, (f"unknown optimizer. Registry: {', '.join(REGISTRY_OPTIMIZERS)}")


def first_run_plan(module_name: str, dataset: str = "hotpot_qa") -> list[str]:
    """The shortest path from nothing to a compiled, evaluated module."""
    return [
        "dspytools doctor",
        "dspytools configure key set openai --stdin",
        "dspytools configure lm set openai/gpt-4o-mini --role default",
        f'dspytools signature new "question: str -> answer: str" --name {module_name.upper()}',
        f"dspytools module new {module_name} --signature {module_name.upper()} --type ChainOfThought",
        f'dspytools run predict "question -> answer" -i question="smoke test"',
        f"dspytools data load {dataset} --format huggingface --split train --limit 200 --name {module_name}-train",
        f"dspytools compile mipro {module_name} {module_name}-train --label v1",
        f"dspytools evaluate run {module_name} {module_name}-dev --metric semantic_f1 --num-threads 8",
    ]


def cli_installed() -> bool:
    return shutil.which("dspytools") is not None


def dry_run() -> None:
    running: set[str] = set()
    for group in ("run", "compile", "graph", "memory", "server"):
        ok, why = can_run(group, running_services=running)
        print(f"{group:10s} runnable={str(ok):5s} | {why}")
    assert can_run("run", running_services=running)[0]
    assert not can_run("graph", running_services=running)[0]
    assert can_run("graph", running_services={"falkordb"})[0]

    for name in ("mipro", "gepa", "better-together", "nonexistent"):
        ok, why = validate_optimizer(name)
        print(f"compile {name:16s} valid={str(ok):5s} | {why}")
    assert validate_optimizer("mipro")[0]
    assert not validate_optimizer("nonexistent")[0]
    assert "teacher LM required" in validate_optimizer("gepa")[1]
    assert "no teacher needed" in validate_optimizer("mipro")[1]

    total = len(REGISTRY_OPTIMIZERS) + len(HANDWRITTEN_OPTIMIZERS)
    print(f"optimizers: {len(REGISTRY_OPTIMIZERS)} registry + "
          f"{len(HANDWRITTEN_OPTIMIZERS)} hand-written = {total}")
    assert total == 19, "the counted optimizer total changed"

    print(f"CLI on PATH: {cli_installed()}")
    print("first-run plan:")
    for step in first_run_plan("qa"):
        print(f"  {step}")
    print("dry run OK")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Verify routing without invoking the CLI")
    ap.add_argument("--plan", metavar="MODULE", help="Print the first-run plan for a module name")
    args = ap.parse_args()
    if args.plan:
        for step in first_run_plan(args.plan):
            print(step)
        return
    dry_run()


if __name__ == "__main__":
    main()
