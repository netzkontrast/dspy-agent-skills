# DSPy Agent Skills

[![DSPy 3.2.x](https://img.shields.io/badge/DSPy-3.2.x-0A7B83)](https://dspy.ai/)

**Production-grade DSPy 3.2.x skills for coding agents.** A synthesized, spec-compliant pack of twenty-nine agent skills that turns Claude Code, Codex CLI, and any other [agentskills.io](https://agentskills.io)-compatible agent into a DSPy expert.

- ✅ Validated against DSPy 3.2.1 (the real API, not inferred from stale docs)
- ✅ Single source of truth for both **Claude Code** and **Codex CLI**
- ✅ Progressive disclosure (short `SKILL.md` + deep `reference.md`)
- ✅ Runnable `example_*.py` scripts with offline `--dry-run`
- ✅ Includes a DSPy 3.2.x `BetterTogether` chaining example
- ✅ Plugin manifest + marketplace manifest for one-click install
- ✅ Validation tests for frontmatter spec, JSON schema, Python AST, skill-doc correctness, and version alignment

## What's inside

| Skill | When it auto-invokes |
|---|---|
| [`dspy-fundamentals`](skills/dspy-fundamentals/SKILL.md) | Any new DSPy code: Signatures, Modules, Predict/ChainOfThought/ReAct, save/load |
| [`dspy-evaluation-harness`](skills/dspy-evaluation-harness/SKILL.md) | Writing metrics, splitting dev/val sets, calling `dspy.Evaluate` |
| [`dspy-gepa-optimizer`](skills/dspy-gepa-optimizer/SKILL.md) | Optimizing/compiling DSPy programs with `dspy.GEPA` |
| [`dspy-rlm-module`](skills/dspy-rlm-module/SKILL.md) | Long context, codebase QA, recursive exploration via `dspy.RLM` |
| [`dspy-rlm-workflow`](skills/dspy-rlm-workflow/SKILL.md) | Context-heavy, multi-step work: distill → decompose → solve → synthesize → verify → iterate, verification as the metric |
| [`dspy-deep-refine`](skills/dspy-deep-refine/SKILL.md) | Refining the knowledge base a program retrieves from (DeepRefine loop, evidence-graded edits, approval gate) |
| [`dspy-reflect-loop`](skills/dspy-reflect-loop/SKILL.md) | Turning session corrections into gold + metric feedback, ledger, promotion, meta-learning |
| [`dspy-clarify`](skills/dspy-clarify/SKILL.md) | Precision gate: explicit scope, bound terms, open questions instead of guesses — before promotion, decomposition or refinement |
| [`dspy-tetraframe`](skills/dspy-tetraframe/SKILL.md) | Critical assessment of a contested decision: four isolated corners (P / not-P / both / neither), contradiction map, non-averaging P*, verification suite — before decisions that change authority |
| [`dspy-autodialectics`](skills/dspy-autodialectics/SKILL.md) | Keeping a program honest: immutable contract, thesis → antithesis → synthesis, independent verification, 12-dimension slop metric, gate, champion/challenger with canaries |
| [`dspy-optimizer-selection`](skills/dspy-optimizer-selection/SKILL.md) | Picking the cheapest optimizer that fits the data and metric — the whole family, baseline first, escalation on a measured plateau |
| [`dspy-retrieval`](skills/dspy-retrieval/SKILL.md) | The retrieval half: Embedder, Embeddings, FAISS, persisted indexes, multi-hop, and scoring recall separately from answers |
| [`dspy-production`](skills/dspy-production/SKILL.md) | Shipping and seeing: cache hardening, save formats, usage, async, streaming, callbacks, MLflow |
| [`dspy-refrag`](skills/dspy-refrag/SKILL.md) | REFRAG fragment selection, with its working parts separated from its unimplemented ones |
| [`dspy-rlm-hooks`](skills/dspy-rlm-hooks/SKILL.md) | Lifecycle hooks and speculative execution for dspy.RLM |
| [`dspy-drg-kg`](skills/dspy-drg-kg/SKILL.md) | Schema-driven knowledge-graph extraction from text with DRG |
| [`dspy-tara-rag`](skills/dspy-tara-rag/SKILL.md) | Self-corrective RAG: a ReAct agent with seven retrieval tools and a 4D context score |
| [`dspy-tools-cli`](skills/dspy-tools-cli/SKILL.md) | The DSPyTools CLI: compile, evaluate and manage DSPy programs from the shell |
| [`dspy-context-engineering-book`](skills/dspy-context-engineering-book/SKILL.md) | Routing 57 O'Reilly book notebooks — the worked example for almost any DSPy question |
| [`dspy-book-eight-steps`](skills/dspy-book-eight-steps/SKILL.md) | The eight-step build order, and optimizing the judge before the task (book ch. 1-3) |
| [`dspy-book-datasets`](skills/dspy-book-datasets/SKILL.md) | Where trainsets come from: conversion, seeded splits, difficulty tiers, synthetic data (book ch. 4) |
| [`dspy-book-metrics`](skills/dspy-book-metrics/SKILL.md) | Eleven metric recipes, and when an LLM judge may be trusted (book ch. 5) |
| [`dspy-book-optimizers`](skills/dspy-book-optimizers/SKILL.md) | Twelve optimizers measured on one task, with cost and wall time (book ch. 6) |
| [`dspy-book-modules`](skills/dspy-book-modules/SKILL.md) | Adapters, multimodal inputs, code execution, parallel and majority voting (book ch. 7) |
| [`dspy-book-agents`](skills/dspy-book-agents/SKILL.md) | MCP tools, conversation memory, and multi-hop budget control (book ch. 8) |
| [`dspy-book-use-cases`](skills/dspy-book-use-cases/SKILL.md) | Seven complete applications as a pattern library (book ch. 9) |
| [`dspy-book-production`](skills/dspy-book-production/SKILL.md) | Serving a compiled program, optimizable guardrails, MLflow tracing (book ch. 10) |
| [`dspy-book-coding-agents`](skills/dspy-book-coding-agents/SKILL.md) | Optimizing a SKILL.md or AGENTS.md as a text artifact (book ch. 11) |
| [`dspy-advanced-workflow`](skills/dspy-advanced-workflow/SKILL.md) | End-to-end builds — orchestrates the core skills, including the self-optimizing loop |

## Install

### Claude Code (via marketplace)

```text
/plugin marketplace add intertwine/dspy-agent-skills
/plugin install dspy-agent-skills@dspy-agent-skills
```

### Agent Skills CLI (`npx skills`)

```bash
npx skills add intertwine/dspy-agent-skills --list
npx skills add intertwine/dspy-agent-skills --skill '*' -a codex -y
```

The Vercel `skills` CLI currently expects a GitHub `owner/repo`, URL, well-known HTTPS endpoint, or local path as its source. The bare form `npx skills add dspy-agent-skills` is not resolvable unless the upstream CLI adds a source alias, so use `intertwine/dspy-agent-skills`.

### Claude Code + Codex (repo checkout)

```bash
git clone https://github.com/intertwine/dspy-agent-skills
cd dspy-agent-skills
./scripts/install.sh           # symlinks into ~/.claude/skills/ and ~/.agents/skills/
```

Flags: `--claude-only`, `--codex-only`, `--copy` (copy instead of symlink), `--uninstall`, `--dry-run`.

### Manual

Drop `skills/*` into `~/.claude/skills/` (Claude Code) or `~/.agents/skills/` (Codex CLI). See [docs/installation.md](docs/installation.md) for all options.

## Five-second demo

In your agent, say:

> "Build a DSPy sentiment classifier, optimize it with GEPA, and save the artifact."

The agent auto-loads `dspy-advanced-workflow`, which chains the other skills and outputs a full baseline → GEPA → export pipeline. No further prompting needed.

## End-to-end examples (current committed artifacts)

Three runnable demos under [`examples/`](examples/) exercise every skill against real LMs and ship with **committed baseline vs. GEPA-optimized numbers** plus explicit `3.1.3` vs. `3.2.0` comparison notes.

| Example | Artifact DSPy | Task LM | Baseline | Optimized | Δ | Status |
|---|---|---|---:|---:|---:|---|
| [01-rag-qa](examples/01-rag-qa/) | 3.2.0 | Ministral 3B 2512 | 80.47 | **100.00** | **+19.53** | Clean comparison refreshed on 2026-04-28 |
| [02-math-reasoning](examples/02-math-reasoning/) | 3.2.0 | Ministral 3B 2512 | 85.00 | **93.33** | **+8.33** | Refreshed on 2026-04-21 |
| [03-invoice-extraction](examples/03-invoice-extraction/) | 3.1.3 | Liquid LFM 2.5 1.2B (free) | 0.833 | **0.931** | **+0.098** | Historical artifact retained |

The refreshed `01` and `02` artifacts use the paid pair `openrouter/mistralai/ministral-3b-2512` + `openrouter/qwen/qwen3-30b-a3b-instruct-2507`. `03` stays on its historical DSPy `3.1.3` artifact because a clean DSPy `3.2.0` baseline on the same Liquid/Nemotron pair already reached `0.944`, leaving little useful headroom for a replacement GEPA artifact. See [`examples/README.md`](examples/README.md) and each example's `version_comparison.md` for the exact commands and caveats.

## Grounding

Every API claim is grounded in:

- https://dspy.ai/ (official docs, DSPy 3.2.x)
- https://code.claude.com/docs/en/skills.md (Claude Code skill spec)
- https://developers.openai.com/codex/skills (Codex skill spec)

## Development

```bash
# Run validation suite
uv run --with pytest python -m pytest tests/ -v

# Smoke-test every example offline (no API key needed)
for f in skills/*/example_*.py; do uv run --with dspy python "$f" --dry-run; done

# Validate the current DSPy API surface used by these skills
env -u UV_EXCLUDE_NEWER uv run --with dspy==3.2.1 python scripts/check_dspy_surface.py

# Live GEPA run (requires OPENAI_API_KEY)
cd skills/dspy-advanced-workflow
OPENAI_API_KEY=... uv run --with dspy python example_pipeline.py --auto light
```

If `uv run --with dspy` resolves an older DSPy release instead of the current `3.2.1` wheel, check whether `UV_EXCLUDE_NEWER` or a stale package mirror is hiding the new release. The exact 3.2.1 override we validated for this repo is:

```bash
env -u UV_EXCLUDE_NEWER uv run --with dspy==3.2.1 python -c 'import dspy; print(dspy.__version__)'
```

## Compatibility

- **DSPy**: 3.2.x (tested against 3.2.1; committed example artifacts remain explicitly labeled by the DSPy version that produced them)
- **Claude Code**: current (skill spec as of 2026-04-17)
- **Codex CLI**: current Agent Skills format
- **Python**: 3.10+
- **Deno**: required only for `dspy.RLM` examples (Pyodide sandbox)

## Layout

```
dspy-agent-skills/
├── .claude-plugin/
│   ├── plugin.json
│   └── marketplace.json
├── skills/
│   ├── dspy-fundamentals/{SKILL.md, reference.md, example_qa.py}
│   ├── dspy-evaluation-harness/{SKILL.md, reference.md, example_metric.py}
│   ├── dspy-gepa-optimizer/{SKILL.md, reference.md, example_gepa.py}
│   ├── dspy-rlm-module/{SKILL.md, reference.md, example_rlm.py}
│   └── dspy-advanced-workflow/{SKILL.md, reference.md, example_pipeline.py}
├── scripts/install.sh           # dual-target installer
├── tests/                       # spec validators
├── docs/{installation,usage,CHANGELOG}.md
├── README.md  LICENSE  .gitignore
```

## Version

**v0.9.0** • Targets DSPy 3.2.x

## License

MIT — see [LICENSE](LICENSE).

## Credits

Draft contributors: Bryan Young ([@intertwine](https://github.com/intertwine)) with Grok (xAI).
Validation, spec-alignment, and dual-agent packaging: Claude Opus 4.7, April 2026.

### Upstream sources

Several skills are ports or consolidations of prior work, each credited in its own `SKILL.md`:

| Skills here | Ported from | License |
|---|---|---|
| `dspy-optimizer-selection`, `dspy-retrieval`, `dspy-production` | [OmidZamani/dspy-skills](https://github.com/OmidZamani/dspy-skills) | MIT |
| `dspy-refrag` | [dspy-refrag](https://github.com/netzkontrast/dspy-refrag) | MIT |
| `dspy-rlm-hooks` | [dspy-rlm-hooks](https://github.com/netzkontrast/dspy-rlm-hooks) | MIT |
| `dspy-drg-kg` | [drg-kg](https://github.com/netzkontrast/drg-kg) | MIT |
| `dspy-tools-cli` | [dspytools](https://github.com/netzkontrast/dspytools) | see repo |
| `dspy-tara-rag` | [self-corrective-rag](https://github.com/netzkontrast/self-corrective-rag) | README claims MIT; LICENSE file absent |
| `dspy-context-engineering-book` | [context-engineering-dspy-book](https://github.com/netzkontrast/context-engineering-dspy-book) | MIT |
| `dspy-tetraframe` | `tetraframe-dspy` (Hmbown) | MIT |
| `dspy-clarify` | `clarify` (Hmbown) | Apache 2.0 |
| `dspy-autodialectics` | [autodialectics](https://github.com/Hmbown/autodialectics) | MIT |
| `dspy-deep-refine` | DeepRefine (arXiv:2605.10488) | — |
| `dspy-reflect-loop` | `claude-reflect-system` | — |
