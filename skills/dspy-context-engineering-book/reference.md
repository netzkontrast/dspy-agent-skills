# Context Engineering with DSPy — Notebook Reference

Source: `context-engineering-dspy-book` (MIT), the companion repo for the
O'Reilly book *Context Engineering with DSPy*. Counted from the tree: 57
notebooks in 11 chapters.

Setup: Python 3.12–3.14, `uv sync`. `requirements.txt` is a uv export of the
lockfile, so prefer `uv sync` over `pip install -r`.

## Full index

| Chapter | Topic | Notebooks |
|---|---|---|
| 1 | Introduction to context engineering | `hello-dspy` |
| 2 | Introduction to DSPy | `dspy-tour` |
| 3 | DSPy in 8 steps | `dspy-in-8-steps`, `humanize-quickstart` |
| 4 | Collecting datasets | `error-analysis-router`, `hf-datasets`, `kaggle-imdb`, `pii-synthesizer`, `synthetic-distillation` |
| 5 | Formalizing evaluation metrics | `string-and-regex-metrics`, `semantic-similarity`, `bleu-rouge-f1`, `human-then-llm-judge`, `rubric-and-multipredictor` |
| 6 | Prompt optimizers | `labeled-few-shot`, `bootstrap-few-shot`, `bootstrap-random-search`, `knn-few-shot`, `copro`, `miprov2`, `simba`, `gepa`, `gepa-expanded-dataset-experiment`, `better-together`, `bootstrap-finetune`, `ensemble`, `quickstart-ai-detector` |
| 7 | Customizing DSPy programs | `modules-tour`, `react-and-tools`, `program-of-thought`, `codeact-and-rlm`, `multi-stage-patterns`, `parallel-and-majority`, `multimodal`, `adapters` |
| 8 | Building AI agents | `react-basics`, `framework-comparison`, `mcp-integration`, `rag-inmemory`, `rag-qdrant`, `web-search-and-multihop`, `history-mem0-rlm` |
| 9 | Real-world use cases | `invoice-extraction`, `customer-service-rag`, `financial-analyst`, `news-researcher`, `blog-writer`, `sentiment-classifier`, `video-generator` |
| 10 | Production | `mlflow-tracking`, `fastapi-invoice-api`, `dspyui-gradio` |
| 11 | Optimizing coding agents | `landing-page-skill-optimizer`, `image-cli-optimizer`, `skill-discovery-rlm`, `test-agents-md`, `clawsona-dspy` |

## Extra assets

- `chapter09/tests/test_invoice_extraction_benchmark.py` with committed results in `chapter09/results/invoice_extraction_benchmark.json` — the only chapter shipping a test and a benchmark artifact.
- `chapter11/assets/compacted-dspy-docs.md` — a condensed DSPy reference used as context inside the chapter-11 notebooks.

## Mapping to this pack

| Book chapter | Skill here | Relationship |
|---|---|---|
| 2, 3, 7 | `dspy-fundamentals` | the skill is the API contract; the notebooks are worked examples |
| 4 | `dspy-evaluation-harness` | devset construction |
| 5 | `dspy-evaluation-harness` | metric design; the skill adds the GEPA-compatible five-argument signature |
| 6 | `dspy-optimizer-selection`, `dspy-gepa-optimizer` | the skill chooses, the notebook demonstrates |
| 7 (`codeact-and-rlm`) | `dspy-rlm-module`, `dspy-rlm-workflow` | |
| 8 (`rag-*`) | `dspy-retrieval` | the skill teaches the injected-retriever shape; `rag-qdrant` shows an external vector DB |
| 10 | `dspy-production` | the skill adds cache hardening, save formats and callbacks the notebooks do not cover |
| 11 | `dspy-reflect-loop`, `dspy-clarify` | optimizing agent skills from feedback |

Where a notebook and a skill disagree on an API, prefer the skill: every API
claim in this pack is asserted against the installed wheel by its example's
dry run. A notebook was accurate when it was authored.

## Caveats

- Notebooks hard-code model names that providers retire. The `dspy.LM(...)` line is the usual edit.
- The dependency closure is large: MLflow, Qdrant, TRL, PEFT, accelerate, fal-client. Keep it in its own environment.
- Chapters 6 and 9 make real optimizer and API calls. Read the budget note in a notebook before running it end to end.
- These are teaching artifacts: inline data, no error handling, no persistence. Port the pattern, not the file.
