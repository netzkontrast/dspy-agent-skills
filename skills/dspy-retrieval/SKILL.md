---
name: dspy-retrieval
description: >-
  Build the retrieval half of a DSPy program — a corpus, a dspy.Embedder, a
  dspy.Embeddings index that switches to FAISS above 20,000 passages, a
  persisted index, and a plain callable retriever passed into the module rather
  than a global. Covers single-hop and multi-hop RAG shapes, and the rule that
  makes RAG debuggable: score retrieval separately from answers, because
  recall@k and answer accuracy fail for different reasons and a single
  end-to-end number cannot tell you which one broke.
when_to_use: >-
  User says "RAG", "retrieval", "search the corpus", "ground the answer",
  "embeddings", "vector search", "multi-hop", "it cites the wrong passage",
  "why did it not find that"; a program must answer from documents; a
  retriever is needed before refining a knowledge base or auditing citations.
---

# DSPy Retrieval (3.2.x)

Consolidated from `dspy-embedding-retrieval` and `dspy-rag-pipeline` of
`OmidZamani/dspy-skills` (MIT), corrected to the retriever-as-argument shape
that DSPy 3.x programs actually use.

The one idea worth carrying: **retrieval is a separate component with a
separate metric.** A RAG program that scores 0.4 tells you nothing. Recall@k of
0.95 with answer accuracy of 0.42 tells you to fix the generator; the reverse
tells you to fix the corpus or the chunking. Skip this split and every
optimizer run afterwards is guesswork.

## The retriever is an argument, not a global

```python
import dspy

corpus = [
    "DSPy programs are composed from modules.",
    "MIPROv2 optimizes instructions and demonstrations.",
    "RLM explores large contexts with a sandboxed REPL.",
]

embedder = dspy.Embedder("openai/text-embedding-3-small")   # or any callable
search = dspy.Embeddings(corpus=corpus, embedder=embedder, k=3)

class RAG(dspy.Module):
    def __init__(self, retrieve):
        super().__init__()
        self.retrieve = retrieve                # injected: swap it in tests
        self.answer = dspy.ChainOfThought("context: list[str], question -> answer")

    def forward(self, question: str) -> dspy.Prediction:
        passages = self.retrieve(question).passages
        pred = self.answer(context=passages, question=question)
        return dspy.Prediction(context=passages, answer=pred.answer)

rag = RAG(search)
```

Returning `context` alongside `answer` is not decoration — it is what lets a
metric check grounding and what `dspy-autodialectics` scores for unsupported
claims.

`dspy.Retrieve` and `dspy.ColBERTv2` still exist and read a globally configured
`rm=`. Older tutorials use them. Prefer the injected callable: a global
retriever cannot be swapped per test, per tenant, or per evaluation split.

## `dspy.Embeddings` in practice

| Parameter | Default | What it means |
|---|---|---|
| `corpus` | required | list of passages; you own the chunking |
| `embedder` | required | `dspy.Embedder` or any callable `list[str] -> 2D array` |
| `k` | 5 | passages returned per query |
| `brute_force_threshold` | 20000 | at or above this, a FAISS index is built — `pip install faiss-cpu` |
| `normalize` | True | normalize embeddings before comparison |
| `cache` | False | cache query results |

```python
search.save("./retrieval-index")                                   # persist
loaded = dspy.Embeddings.from_saved("./retrieval-index", embedder=embedder)
```

Persist whenever embedding the corpus costs more than loading it — which is
almost always after the first thousand passages. Re-embed only when the corpus
or the embedding model changes, and version the index directory with both.

Use `dspy.EmbeddingsWithScores` when the similarity values matter: thresholding
out weak matches, reranking, or explaining to a user why nothing was found.

A local embedding model needs no hosted API:

```python
from sentence_transformers import SentenceTransformer
embedder = dspy.Embedder(SentenceTransformer("all-MiniLM-L6-v2").encode)
```

## Multi-hop: one query is often not enough

When the answer needs a fact that the first query cannot surface, generate the
next query from what you already have:

```python
class MultiHopRAG(dspy.Module):
    def __init__(self, retrieve, max_hops: int = 2):
        super().__init__()
        self.retrieve, self.max_hops = retrieve, max_hops
        self.next_query = dspy.ChainOfThought("context: list[str], question -> query")
        self.answer = dspy.ChainOfThought("context: list[str], question -> answer")

    def forward(self, question: str) -> dspy.Prediction:
        context, query = [], question
        for _ in range(self.max_hops):
            context = list(dict.fromkeys(context + self.retrieve(query).passages))
            query = self.next_query(context=context, question=question).query
        pred = self.answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=pred.answer)
```

Deduplicate between hops (`dict.fromkeys` preserves order). Without it hop two
re-retrieves hop one's passages, the context grows without new information, and
the redundancy shows up as a slop signal rather than a retrieval bug.

Cap `max_hops` at 2 or 3. Each hop is a full generate-plus-retrieve round trip,
and recall rarely improves after the third.

## Score retrieval on its own

```python
def recall_at_k(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """Fraction of the gold passages that retrieval actually surfaced."""
    got = {p.strip() for p in pred.context}
    want = {p.strip() for p in gold.gold_passages}
    hits = len(got & want)
    score = hits / max(len(want), 1)
    missing = want - got
    feedback = (f"Missed {len(missing)} gold passage(s): {list(missing)[:2]}"
                if missing else "All gold passages retrieved.")
    return dspy.Prediction(score=score, feedback=feedback)
```

Run `dspy.Evaluate` with this metric *and* with your answer metric. The pair
localizes every regression. The five-argument signature and
`dspy.Prediction(score, feedback)` return keep the metric usable as a GEPA
signal — see `dspy-evaluation-harness`.

| Recall@k | Answer accuracy | What is broken |
|---|---|---|
| low | low | corpus, chunking, or the embedding model |
| high | low | the generator, or the context field type |
| low | high | the model is answering from parameters, not the corpus — check for leakage |

## Anti-patterns

- One end-to-end score for a RAG pipeline; you cannot tell retrieval failure from generation failure.
- A global `rm=` retriever, then wondering why a test cannot isolate the corpus.
- Raising `k` to fix low recall: it raises cost and dilutes context. Fix the chunking first.
- Re-embedding the corpus on every run instead of `save` / `from_saved`.
- Multi-hop without deduplication — the same passages, more tokens, no new facts.
- Dropping `context` from the returned `Prediction`, which leaves grounding unverifiable.
- Changing the embedding model without rebuilding the index; the vectors are silently incomparable.

## Where to go next

- The corpus itself keeps failing the question → `dspy-deep-refine`
- Corpus too large to index, or needs reasoning rather than lookup → `dspy-rlm-module`
- Metrics, devsets and `dspy.Evaluate` → `dspy-evaluation-harness`
- Optimizing the pipeline once both metrics exist → `dspy-optimizer-selection`
- Checking that the answer is actually grounded in the retrieved context → `dspy-autodialectics`
- Full reference (Embedder parameters, FAISS, chunking, index versioning) → [reference.md](reference.md)
- Runnable, dry-run capable → [example_retrieval.py](example_retrieval.py)
