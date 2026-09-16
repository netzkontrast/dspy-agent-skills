---
name: dspy-tools-cli
description: >-
  Drive the DSPyTools CLI — a self-evolving command line for DSPy program
  management with roughly 24 command groups covering configure, signature,
  module, run, compile, evaluate, data, skills, graph and a Generative Feedback
  Loop. Covers the eight commands that get a first program compiled, which of
  the advertised subsystems need FalkorDB and Redis running, the repo's own
  authoring rules (never import dspy directly, teacher LM only for
  optimization), and which headline numbers do not survive a source count.
when_to_use: >-
  User says "dspytools", "dspy CLI", "compile from the command line", "GFL",
  "self-evolving CLI", "hot-swap inference", "skill graph"; DSPy programs are
  being managed as artifacts rather than written inline; a CLI-driven
  optimization workflow is being set up.
---

# DSPyTools CLI

A large CLI wrapping DSPy program management: generate signatures and modules,
run inference, compile with any of ten registry optimizers, evaluate, and keep
the artifacts in a graph. Commands are lazily imported, so the surface is big
but startup stays cheap.

Install (not on PyPI, needs Python >= 3.12):

```bash
pip install git+https://github.com/netzkontrast/dspytools
dspytools doctor        # start here: checks LLM, GPU, config and services
```

## What runs without infrastructure

This is the first thing to establish, because a third of the surface needs
services that are not installed by `pip`.

| Needs nothing | Needs FalkorDB + Redis on :6379 | Needs other services |
|---|---|---|
| `configure`, `signature`, `module`, `run`, `compile`, `evaluate`, `data`, `doctor`, `generate`, `export`, `compare` | `graph`, `memory`, semantic cache | `lora`, `server` need llama-cpp-server; MLflow calls need :5000 |

```bash
docker compose -f docker-compose.redis.yml up -d     # falkordb/falkordb, port 6379
```

The graph client circuit-breaks rather than degrading cleanly, so `graph
status` simply reports FAILED when the service is down. That is a missing
service, not a bug.

## The first eight commands

```bash
dspytools doctor
dspytools configure key set openai --stdin
dspytools configure lm set openai/gpt-4o-mini --role default   # roles: student|teacher|default
dspytools signature new "question: str -> answer: str" --name QA
dspytools module new qa --signature QA --type ChainOfThought
dspytools run predict "question -> answer" -i question="What is DSPy?"
dspytools data load hotpot_qa --format huggingface --split train --limit 200 --name qa-train
dspytools compile mipro qa qa-train --label v1
dspytools evaluate run qa qa-dev --metric semantic_f1 --num-threads 8
```

Every optimizer command shares one generated shape:
`dspytools compile <optimizer> MODULE_NAME TRAINSET_PATH [--label L] [--force]`.
Ten optimizers are registry-generated (`knn`, `mipro`, `gepa`, `copro`,
`simba`, `bootstrap-few-shot`, `bootstrap-few-shot-random`,
`bootstrap-few-shot-optuna`, `labeled-few-shot`, `infer-rules`); nine more are
hand-written (`submit`, `flex`, `better-together`, `ensemble`, `finetune`,
`gfl`, `grpo`, `avatar`, `distill`).

Choosing among them is a decision this CLI does not make for you — see
`dspy-optimizer-selection`.

## Configuration model

No environment variable is mandatory. Config resolves defaults →
`~/.config/dspytools/config.toml` → `<project>/.dspytools/config.toml`, with
mtime-checked hot reload. Every path has a `DSPYTOOLS_<NAME>_DIR` override.

The README's model names are **defaults, not requirements**. Local
llama-cpp-server, Qwen and DeepSeek are all swappable; the hard-coded fallback
is `openai/gpt-4o`. A teacher LM is required only for GEPA, distill and
finetune — `compile gepa` aborts with "No teacher LM configured" and nothing
else does.

Merely running `--help` creates `~/.config/dspytools/*`, because path helpers
`mkdir` on first access.

## The repo's own rules, if you contribute

Its `AGENTS.md` files carry a documentation contract and six golden rules worth
knowing before you touch the source:

1. `from dspytools.core._dspy import dspy` — never `import dspy` directly.
2. Teacher LM only in GEPA, distill and finetune; never for inference.
3. `split_holdout()` before any compile.
4. **No try/except around imports** — every package is a hard dependency.
5. Student is the local model for inference; teacher is the reflection model.
6. Ruff clean before commit.

Rule 4 is the one that bites users, not just contributors: a missing optional
dependency is a hard crash, not a degraded mode. Mojo and MCP loading are the
two deliberate exceptions, and Mojo is genuinely optional with a Python
fallback.

## Headline numbers that do not survive a count

The README advertises "24 command groups · 166 subcommands · 17+ optimizers ·
11 arXiv paper implementations". Counted from source: 23 groups plus one
standalone command, 183 leaf subcommands, 19 optimizers, and 10 distinct arXiv
identifiers. Four of those identifiers are dated 2026 and cannot be verified
offline. Treat the paper-implementation claims as unverified rather than wrong,
and do not repeat them as provenance for a technique.

## Anti-patterns

- Running `graph`, `memory` or cache commands without FalkorDB up, then reading the circuit-breaker failure as a bug.
- Expecting a missing optional dependency to degrade gracefully; rule 4 makes it a crash.
- Assuming Qwen or DeepSeek are required. Configure any provider.
- Citing the repo's arXiv list as evidence for a method without checking the identifier.
- Configuring a teacher LM for ordinary inference; it is for reflection only.
- Treating `--help` as free: with a compiled help program it may call an LLM.
- Expecting Ctrl-C to exit 130; its signal handlers exit 0.

## Where to go next

- Which optimizer to pass to `compile` → `dspy-optimizer-selection`
- Metrics behind `evaluate run` → `dspy-evaluation-harness`
- Shipping the compiled artifact this CLI produces → `dspy-production`
- Full reference (command groups, subsystems, env vars, deps) → [reference.md](reference.md)
- Runnable, dry-run capable → [example_tools_cli.py](example_tools_cli.py)
