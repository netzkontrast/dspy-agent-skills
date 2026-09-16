---
name: dspy-context-engineering-book
description: >-
  Route a DSPy question to the worked notebook that already answers it, in the
  57-notebook companion repo for the O'Reilly book Context Engineering with
  DSPy. Chapter 6 alone holds one runnable notebook per optimizer
  (BootstrapFewShot, random search, KNN, labeled, COPRO, MIPROv2, SIMBA, GEPA,
  BetterTogether, BootstrapFinetune, Ensemble), and chapters 7 to 11 cover
  modules, agents, real use cases, production and coding-agent optimization.
  Use it to find a working reference implementation instead of writing one.
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

## Route by what you are trying to do

| Your question | Notebook |
|---|---|
| First DSPy program | `chapter01/hello-dspy` |
| Tour of the whole framework | `chapter02/dspy-tour` |
| The canonical build sequence | `chapter03/dspy-in-8-steps` |
| Where do I get training data | `chapter04/hf-datasets`, `kaggle-imdb` |
| Generate data instead of collecting it | `chapter04/synthetic-distillation`, `pii-synthesizer` |
| Decide what to fix first | `chapter04/error-analysis-router` |
| Write a metric | `chapter05/string-and-regex-metrics`, `semantic-similarity`, `bleu-rouge-f1` |
| LLM-as-judge, done carefully | `chapter05/human-then-llm-judge` |
| Rubric scoring, multiple predictors | `chapter05/rubric-and-multipredictor` |
| **Any single optimizer, end to end** | `chapter06/` — see below |
| Module composition patterns | `chapter07/modules-tour`, `multi-stage-patterns` |
| Tools and ReAct | `chapter07/react-and-tools` |
| Code execution | `chapter07/program-of-thought`, `codeact-and-rlm` |
| Concurrency and voting | `chapter07/parallel-and-majority` |
| Images, audio, files | `chapter07/multimodal` |
| Output format control | `chapter07/adapters` |
| Agent basics and framework comparison | `chapter08/react-basics`, `framework-comparison` |
| MCP tools in an agent | `chapter08/mcp-integration` |
| RAG, in memory then real vector DB | `chapter08/rag-inmemory`, `rag-qdrant` |
| Multi-hop retrieval with web search | `chapter08/web-search-and-multihop` |
| Conversation history and memory | `chapter08/history-mem0-rlm` |
| A complete use case to copy | `chapter09/` — 7 of them |
| Experiment tracking | `chapter10/mlflow-tracking` |
| Serve it behind an API | `chapter10/fastapi-invoice-api` |
| Give it a UI | `chapter10/dspyui-gradio` |
| Optimize a coding agent's skills | `chapter11/landing-page-skill-optimizer`, `image-cli-optimizer` |
| Discover skills with RLM | `chapter11/skill-discovery-rlm` |
| Test an AGENTS.md | `chapter11/test-agents-md` |
| Persona engineering | `chapter11/clawsona-dspy` |

## Chapter 6 is the optimizer reference

One self-contained notebook per optimizer, which makes it the practical
companion to `dspy-optimizer-selection`: pick the optimizer there, then read
its notebook here.

`labeled-few-shot` · `bootstrap-few-shot` · `bootstrap-random-search` ·
`knn-few-shot` · `copro` · `miprov2` · `simba` · `gepa` ·
`gepa-expanded-dataset-experiment` · `better-together` · `bootstrap-finetune` ·
`ensemble` · `quickstart-ai-detector` (the baseline all of them improve on).

The `gepa-expanded-dataset-experiment` notebook is the one worth reading even
if you have chosen a different optimizer: it shows what changing the dataset,
rather than the optimizer, does to the result.

## Chapter 9 use cases

`invoice-extraction` (the one with a committed benchmark and tests),
`customer-service-rag`, `financial-analyst`, `news-researcher`, `blog-writer`,
`sentiment-classifier`, `video-generator`. Start from the one closest to your
shape rather than from a blank notebook.

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

- Choosing which chapter-6 notebook applies → `dspy-optimizer-selection`
- Metric design behind chapter 5 → `dspy-evaluation-harness`
- RAG behind chapter 8 → `dspy-retrieval`
- Production behind chapter 10 → `dspy-production`
- Coding-agent optimization behind chapter 11 → `dspy-reflect-loop`
- Full reference (every notebook, per chapter) → [reference.md](reference.md)
- Runnable, dry-run capable → [example_book_map.py](example_book_map.py)
