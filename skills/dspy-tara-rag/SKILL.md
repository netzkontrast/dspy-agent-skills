---
name: dspy-tara-rag
description: >-
  Build or reproduce TARA, a self-corrective RAG agent that replaces a fixed
  refinement loop with a ReAct agent holding seven retrieval tools, and scores
  its own context on four dimensions (relevance, coverage, specificity,
  sufficiency) with a threshold that relaxes on each retry. Covers the tool
  set, the 4D evaluation signature, the five comparable pipeline variants, the
  dataset and index build order, and the cost profile — this design spends tens
  of LLM calls per question, so it earns its place only on multi-hop questions.
when_to_use: >-
  User says "self-corrective RAG", "TARA", "agentic retrieval", "the retriever
  keeps missing", "CRAG", "multi-hop QA", "should the agent decide how to
  retrieve"; a fixed retrieve-then-generate loop is failing; a RAG design needs
  a baseline comparison rather than an anecdote.
---

# TARA — Tool-Augmented Self-Corrective RAG

TARA is a research codebase with a paper, not a library you import. Its
transferable idea: rather than a hard-coded refine loop, give a ReAct agent
tools and let it decide how to fix bad retrieval. The repo's value is that it
ships four comparable baselines, so the design can be judged rather than
believed.

Package root is `agentic_rag/` at the repo top level (flat, not `src/`); the
distribution name is `agentic-self-corrective-rag`.

## The seven tools

The README says six. The registry holds seven — `calculate` is omitted from the
prose.

| Tool | Signature | Purpose |
|---|---|---|
| `search_passages` | `(query, top_k=10)` | hybrid FAISS + BM25 search |
| `decompose_query` | `(question)` | split a multi-hop question into sub-questions |
| `evaluate_passages` | `(question, passage_ids_json, retry_count=0)` | 4D scoring plus refinement feedback |
| `get_passage_detail` | `(passage_id, include_adjacent=False)` | full text, optionally neighbours |
| `list_document_sections` | `(keyword="")` | browse document structure |
| `get_terminology` | `(user_term)` | map user vocabulary to document vocabulary |
| `calculate` | `(expression)` | sandboxed arithmetic for financial questions |

Core four: `search_passages`, `decompose_query`, `evaluate_passages`,
`get_passage_detail`. Domain-adaptive: `list_document_sections` and
`get_terminology`, which the ablation switches off for Wikipedia-style corpora
where they add nothing. `calculate` is the numeric-domain seventh.

Two tools make their own LLM calls (`decompose_query`, `evaluate_passages`), so
the agent's iteration count understates the real cost.

## Four-dimensional quality assessment

The transferable core. Instead of one relevance score, the evaluator emits
four, which localizes *why* the context is bad:

| Dimension | Points | Failing it means |
|---|---|---|
| relevance | 0–30 | wrong passages |
| coverage | 0–25 | right topic, missing pieces |
| specificity | 0–25 | too general to answer |
| sufficiency | 0–20 | cannot answer even if all true |

The signature also emits `action` (`output` / `refine` / `route_to_agent`),
`keywords_to_add`, `keywords_to_remove` and `suggested_query`, so a failing
score carries its own repair instruction.

Retry uses **progressive leniency** — the bar drops rather than looping
forever:

```python
effective_threshold = max(quality_threshold - (retry * 5), 20)   # default 40, max 3 retries
```

Note a real mismatch: the signature docstring states the formula without the
floor, while the code floors at 20. The code wins.

The floor has a consequence worth stating plainly: the loop terminates by
**lowering the bar**, not by escalating. A context scoring 27 out of 100 fails
at retry 0 and is accepted at retry 3, because the threshold has decayed to 25.
Only a context below 20 is ever routed away. If you adopt this pattern, decide
deliberately whether "eventually accept something mediocre" is the behaviour
you want, or set the floor where a genuine escalation still happens.

## Judge it against the baselines, not in isolation

Five pipelines share one retriever and one dataset loader:

| Variant | What it is |
|---|---|
| `naive` | retrieve then generate, no evaluation |
| `crag` | a CRAG replica: binary correct/incorrect/ambiguous, single-pass correction |
| `loop` | fixed-iteration refinement — the ablation baseline |
| `ircot` | interleaved retrieval and chain of thought |
| `agentic` | TARA |

This is the part worth copying even if you never run TARA. A self-corrective
design that is not measured against `naive` and `loop` is unfalsifiable, and
the repo's own results show the gain is real on 2WikiMultiHopQA and
statistically insignificant on HotpotQA and FinanceBench.

## Run order

The index must exist before any run; nothing builds it lazily.

```bash
uv sync
uv run python scripts/prepare_datasets.py --dataset 2wikimultihopqa --sample 500
uv run python scripts/build_index.py --dataset 2wikimultihopqa
uv run python experiments/run.py --config configs/experiment/rq1.yaml \
    --dataset 2wikimultihopqa --sample 200
uv run python scripts/verify_campaign.py <run_dir> --n 200
```

Always pass `--dataset` explicitly: its default is `popqa`, which
`prepare_datasets.py` no longer exposes.

Only the **generator** is compiled, never the ReAct agent —
`experiments/run.py` finds the `generator` attribute and swaps in a compiled
`ChainOfThought`. Compilation only happens when a variant declares
`optimization: bootstrap|mipro`.

## Cost before you start

CRAG issues roughly 40 LLM calls per question against naive's one; TARA runs
five or six tools per question with extra internal calls. A full campaign is
five pipelines across four datasets at 150–200 questions each, times repeats.
Building the FinanceBench index itself costs LLM calls, because section titles
are classified per passage.

For reproducible latency numbers set `DISABLE_LLM_CACHE=true`. Note also that
`RETRIEVAL_MAX_PASSAGES_ALL_PIPELINES` is off by default, so the published runs
gave naive and CRAG more passages than loop and agentic — the "controlled"
comparison is opt-in.

## Licensing caveat

The README carries an MIT badge and links a `LICENSE` file that **is not
present in the repository**. Resolve that with the authors before vendoring any
of it. Treat the design as readable and the code as unlicensed until fixed.

## Anti-patterns

- Adopting agentic refinement without running `naive` and `loop` on your own data; the published gain does not hold on every dataset.
- Reporting a win from a single dataset: three of the four differences in the paper are not significant.
- Forgetting `--dataset`, silently running against the stale default.
- Comparing latency with the LLM cache on.
- Expecting the agent itself to be optimized; only the generator is compiled.
- Feeding untrusted input to `calculate`; it is a restricted `eval`, adequate for benchmarks only.
- Copying code before the license question is settled.

## Where to go next

- The retrieval layer underneath → `dspy-retrieval`
- Fixing the corpus rather than the loop → `dspy-deep-refine`
- Metrics and the comparison discipline → `dspy-evaluation-harness`
- Compiling the generator → `dspy-optimizer-selection`
- Full reference (module map, signatures, configs, datasets) → [reference.md](reference.md)
- Runnable, dry-run capable → [example_tara.py](example_tara.py)
