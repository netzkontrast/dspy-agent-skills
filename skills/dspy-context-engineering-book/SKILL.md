---
name: dspy-context-engineering-book
description: >-
  Index over the nine per-chapter skills derived from the O'Reilly book Context
  Engineering with DSPy, and a direct route to any of its 57 notebooks. Points
  at the chapter skill that carries each technique: build order, datasets,
  metric recipes, measured optimizer results, modules and adapters, agents and
  MCP, seven application architectures, serving, and optimizing a coding
  agent's own instruction files.
when_to_use: >-
  User asks for a worked DSPy example, "is there a notebook for this", "how
  would I actually build X in DSPy", "show me MIPROv2 end to end", "example of
  multimodal / adapters / MCP / RAG with Qdrant"; a pattern needs a reference
  implementation before being written from scratch.
---

# Context Engineering with DSPy — Notebook Map

Companion code (MIT) for the O'Reilly book *Context Engineering with DSPy*: 57
notebooks across 11 chapters. This skill is a **router**, not a tutorial. The
value is finding the notebook that already solves your problem.

```bash
git clone https://github.com/netzkontrast/context-engineering-dspy-book
cd context-engineering-dspy-book && uv sync      # Python 3.12-3.14
```

`uv sync` creates `.venv` from the lockfile. Do not hand-manage a virtualenv.

## The chapter skills

The book is split into nine skills, one per chapter group. Load the one that
matches your question; each carries that chapter's transferable technique, not
a summary.

| Skill | Chapters | What it gives you |
|---|---|---|
| `dspy-book-eight-steps` | 1–3 | the build order, and why the judge is optimized before the task |
| `dspy-book-datasets` | 4 | conversion, seeded splits, difficulty tiers, synthetic data and its risks |
| `dspy-book-metrics` | 5 | eleven metric recipes and the judge-calibration trust bar |
| `dspy-book-optimizers` | 6 | twelve optimizers measured on one task, with cost and wall time |
| `dspy-book-modules` | 7 | adapters, multimodal inputs, code execution, parallel and voting |
| `dspy-book-agents` | 8 | MCP tools, conversation memory, multi-hop budgets |
| `dspy-book-use-cases` | 9 | seven application architectures, routed by task shape |
| `dspy-book-production` | 10 | serving a compiled program, guardrails, MLflow tracing |
| `dspy-book-coding-agents` | 11 | optimizing a SKILL.md or AGENTS.md as a text artifact |

## Find a notebook directly

| Your question | Notebook |
|---|---|
| First DSPy program | `chapter01/hello-dspy` |
| Tour of the framework | `chapter02/dspy-tour` |
| The canonical build sequence | `chapter03/dspy-in-8-steps` |
| Where training data comes from | `chapter04/hf-datasets`, `synthetic-distillation` |
| Writing a metric | `chapter05/` — five notebooks |
| Any single optimizer, end to end | `chapter06/` — thirteen notebooks |
| Modules, adapters, multimodal | `chapter07/` — eight notebooks |
| Agents, MCP, RAG, memory | `chapter08/` — seven notebooks |
| A complete application to copy | `chapter09/` — seven of them |
| Serving and tracking | `chapter10/` — three notebooks |
| Optimizing a coding agent | `chapter11/` — five notebooks |

## How to use it well

- Read the notebook, port the pattern, do not vendor the notebook. These are teaching artifacts with inline data and hard-coded models.
- Notebooks pin models that may be retired. Expect to swap the `dspy.LM(...)` line.
- The repo's dependency set is large (MLflow, Qdrant, TRL, PEFT, accelerate). Install it in its own environment, not into your project.
- Check the notebook against this pack's skill for the same topic. Where they disagree, the skill was verified against the installed wheel; a notebook was correct when it was written.

## Anti-patterns

- Copying a notebook wholesale into production; that is what chapters 10 and this pack's `dspy-production` are for.
- Treating a notebook's optimizer choice as a recommendation for your task — it is a demonstration of that optimizer.
- Installing the book's requirements into an existing project environment.
- Reading chapter 6 notebooks as benchmarks: they show mechanics, not comparative results on your data.

## Where to go next

- The nine chapter skills above carry the techniques; this page only routes.
- Cross-cutting skills they build on → `dspy-fundamentals`, `dspy-evaluation-harness`, `dspy-optimizer-selection`, `dspy-retrieval`, `dspy-production`
- Full reference (every notebook, per chapter, and the mapping to this pack) → [reference.md](reference.md)
- Runnable, dry-run capable → [example_book_map.py](example_book_map.py)
