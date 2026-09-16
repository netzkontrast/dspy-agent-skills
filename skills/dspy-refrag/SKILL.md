---
name: dspy-refrag
description: >-
  Use the dspy-refrag package (REFRAG: retrieval with fragment selection) with
  accurate expectations. Covers the real working path — corpus ingestion, an
  embedder, SimpleRetriever, the heuristic and advanced sensors with their five
  selection strategies — and documents, with file and line evidence, the parts
  that do not work: the fragment selection never shrinks the prompt, the FAISS
  and Pinecone backends raise NotImplementedError, and importing the package
  requires psycopg2. Includes what is worth reusing versus reimplementing.
when_to_use: >-
  User says "REFRAG", "dspy-refrag", "fragment selection", "MMR selection",
  "should we adopt refrag", "context compression for retrieval"; a retrieval
  layer is being chosen or reviewed; someone proposes vendoring this package.
---

# DSPy REFRAG (dspy-refrag 0.1.0)

REFRAG is a real idea: retrieve many chunks, then select a small subset to
expand into the prompt, so context cost falls without recall falling. The
`dspy-refrag` package implements the retrieval and the selection. It does
**not** implement the compression the idea is named for.

This skill exists so you adopt it with open eyes. Every claim below was
verified by reading the source, with line references, not from the README.

## What actually works

| Component | Status |
|---|---|
| `SimpleRetriever` with a real embedder | works |
| `PSQLRetriever` (Postgres + pgvector) | works; vector dimension hardcoded to 768 |
| `WeaviateRetriever` | code is correct but unusable under the declared pin (below) |
| `Sensor` heuristic selection | works |
| `AdvancedSensor` with 5 strategies | works, and is the most reusable part |
| `build_corpus_from_data` (PDF to chunks) | works |
| `FAISSRetriever`, `PineconeRetriever` | **`__init__` raises `NotImplementedError`** |
| Fragment selection reducing prompt size | **not implemented** |
| Serializer layer | works, but operates on a type the retrieval path never produces |

## The headline feature is missing

`REFRAGModule.forward` computes the selection, writes it into metadata, then
builds the prompt from **every** passage and merely annotates each one:

```python
context_str = "\n".join(
    f"Passage {i}: {p['text']} (selected: {p.get('selected', False)})" ...
)
```

So the sensor changes a boolean in the prompt. It does not remove unselected
text. Token cost is identical to passing everything. If you adopted this
package expecting context compression, you would measure no saving and have no
error to explain it.

The selection logic itself is sound. The gap is between selection and prompt
assembly, and it is roughly a ten-line fix in `forward`.

## The working path

```python
from pathlib import Path
from dspy_refrag import REFRAGModule, SimpleRetriever
from dspy_refrag.common import make_ollama_embedder
from dspy_refrag.data_ingest import build_corpus_from_data

embedder = make_ollama_embedder(model="nomic-embed-text:latest")
corpus = build_corpus_from_data(embedder, Path("data"))    # PDFs -> chunks -> Passage
retriever = SimpleRetriever(embedder=embedder, corpus=corpus)
ctx = REFRAGModule(retriever=retriever, k=5, budget=3).forward("your query")
print(ctx.answer)
```

`SimpleRetriever()` **without** an embedder returns random vectors. It is a
test fixture, not a retriever, and it fails silently as one.

## Four traps that cost an afternoon

1. **`import dspy_refrag` fails without psycopg2.** `__init__.py` imports
   `psql_retriever`, which does a top-level `import psycopg2`. Postgres is a
   hard import dependency of the whole package, even for FAISS-only use.

2. **The README quick start crashes as soon as an API key is present.**
   `forward` does `p['text']` unguarded, outside the try block, while
   `SimpleRetriever`'s default corpus metadata has no `text` key. Every passage
   must carry its text in `metadata["text"]`, which `build_corpus_from_data`
   does and hand-built corpora usually do not.

3. **`lm_model=` is not authoritative.** The constructor calls
   `maybe_configure_openrouter_env()`, which overwrites your argument with
   `$OPENROUTER_MODEL` and mutates `os.environ`. The model you passed may not
   be the model that runs.

4. **Weaviate cannot import as pinned.** `pyproject.toml` pins
   `weaviate-client>=3.25,<4.0`; `weaviate_retriever.py` imports
   `weaviate.classes.config` and `weaviate.classes.query`, which are v4-only.
   One of the two must change before that backend runs.

Also: LM errors are swallowed into the answer string as
`"Error calling LM: ..."` rather than raised, so a failed call looks like a
successful one with an odd answer.

## The part worth keeping: selection strategies

`sensor_advanced.py` is the most valuable file in the package and it has no
dependency on the broken parts.

```python
from dspy_refrag.sensor_advanced import AdvancedSensor, SelectionConfig, SelectionStrategy

sensor = AdvancedSensor(SelectionConfig(strategy=SelectionStrategy.MMR,
                                        diversity_lambda=0.5))
module = REFRAGModule(retriever=retriever, sensor=sensor, k=20, budget=5)
```

| Strategy | Selects for | Tune with |
|---|---|---|
| `SIMILARITY` | closest to the query | `min_score` |
| `MMR` | relevance minus redundancy | `diversity_lambda` |
| `UNCERTAINTY` | where the model is least sure | `temperature` |
| `ENSEMBLE` | combination of the above | `ensemble_weights` |
| `ADAPTIVE` | a score-distribution cutoff | `adaptive_percentile` |

`AdvancedSensor` duck-types into `REFRAGModule(sensor=...)`, and `.select()`
returns plain indices, so it lifts cleanly into any retrieval stack. MMR in
particular is what you want when the top-k are near-duplicates.

**Always set `min_score`.** Plain MMR scores a candidate as relevance minus
redundancy, so an irrelevant passage scores zero while a relevant
near-duplicate scores slightly below zero — and the irrelevant one wins.
Measured on a four-passage fixture, unguarded MMR picks the passage with zero
similarity to the query at every diversity setting from 0.5 to 0.8. With a
floor it picks the relevant, distinct passage. `diversity_lambda` controls
duplicate versus distinct and needs roughly 0.6 or above to bite; the floor is
what keeps irrelevant passages out, and no `diversity_lambda` substitutes for
it.

## Adopt, vendor or reimplement

| You want | Do this |
|---|---|
| MMR or adaptive selection over your own retriever | vendor `sensor_advanced.py` (MIT); it is self-contained |
| PDF to chunked, embedded corpus | reuse `data_ingest.build_corpus_from_data` |
| Actual context compression | reimplement prompt assembly; the package does not do it |
| FAISS retrieval | use `dspy.Embeddings` (see `dspy-retrieval`); this package's FAISS class is a stub |
| A production retrieval layer | not this package: no retries, pooling, batching or async anywhere |

## Anti-patterns

- Benchmarking token savings against this package and concluding REFRAG does not work; you measured an unimplemented path.
- Depending on `FAISSRetriever` or `PineconeRetriever` because the README lists them as production-ready.
- Building a corpus without `metadata["text"]`, then debugging a `KeyError` inside `forward`.
- Trusting `lm_model=` when `OPENROUTER_MODEL` is set.
- Passing the serializer layer a `Passage`; it takes `Fragment`, and no converter between them exists.
- Reading `ctx.answer` without checking for the swallowed `"Error calling LM:"` prefix.
- Installing it into an environment where a Postgres client is unwelcome.

## Where to go next

- A retrieval layer that works today → `dspy-retrieval`
- Fixing the corpus rather than the prompt → `dspy-deep-refine`
- Judging whether a retrieval change helped → `dspy-evaluation-harness`
- Porting plan for the Kohärenz Protokoll wiki → [docs/kohaerenz-protokoll-plugin-plan.md](../../docs/kohaerenz-protokoll-plugin-plan.md)
- Full reference (exports, signatures, backends, env vars, evidence) → [reference.md](reference.md)
- Runnable, dry-run capable → [example_refrag.py](example_refrag.py)
