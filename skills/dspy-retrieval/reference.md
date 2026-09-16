# DSPy Retrieval — Reference

Source: `dspy-embedding-retrieval` and `dspy-rag-pipeline` of
`OmidZamani/dspy-skills` (MIT), merged and corrected against the installed
wheel. The source pack taught the global `dspy.configure(rm=...)` +
`dspy.Retrieve(k=...)` pattern; that still runs, but this reference teaches the
injected-callable shape because it is testable.

## Verified signatures

```python
dspy.Embedder(model: str | Callable, batch_size=200, caching=True, **kwargs)

dspy.Embeddings(corpus: list[str], embedder, k=5, callbacks=None,
                cache=False, brute_force_threshold=20_000, normalize=True)

dspy.Embeddings.save(path: str)
dspy.Embeddings.from_saved(path: str, embedder)          # staticmethod
```

`dspy.EmbeddingsWithScores` has the same constructor and additionally returns
similarity scores. Both return a `dspy.Prediction` exposing `.passages` and
`.indices`.

## What a retriever must look like

Anything callable that takes a query string and returns an object with
`.passages` works. That is the whole contract, and it is why an injected
retriever is worth the one extra constructor argument:

```python
class StubRetriever:
    """Deterministic test double — no embeddings, no network."""
    def __init__(self, corpus: list[str], k: int = 3):
        self.corpus, self.k = corpus, k

    def __call__(self, query: str) -> dspy.Prediction:
        terms = {w.lower() for w in query.split() if len(w) > 3}
        ranked = sorted(self.corpus,
                        key=lambda p: -len(terms & {w.lower() for w in p.split()}))
        return dspy.Prediction(passages=ranked[: self.k])
```

Swap this in for unit tests, for CI, and for any evaluation where retrieval
should be held constant while the generator changes.

## `dspy.Embedder` models

| Form | Example | Notes |
|---|---|---|
| hosted model string | `dspy.Embedder("openai/text-embedding-3-small")` | LiteLLM naming; any supported provider |
| local callable | `dspy.Embedder(SentenceTransformer(...).encode)` | must accept `list[str]`, return a 2D array |
| batching | `dspy.Embedder(model, batch_size=200)` | lower it if the provider rate-limits |
| caching | `dspy.Embedder(model, caching=True)` | on by default; disable for one-shot corpora |

Embedding a corpus is a one-time cost paid per model. Changing the model
invalidates the index completely — the vectors from two models are not
comparable, and nothing raises an error if you mix them. The failure shows up
as silently poor recall.

## FAISS

At or above `brute_force_threshold` (default `20_000`) passages,
`dspy.Embeddings` builds a FAISS index instead of scanning:

```bash
pip install faiss-cpu
```

Below the threshold, brute force is faster than the index build. Do not lower
the threshold without measuring. Measure memory as well as latency: a FAISS
index over a large corpus is resident for the process lifetime.

## Chunking

DSPy does not chunk for you — the corpus you pass is the retrieval unit, and
almost every disappointing recall number traces back to this list:

| Rule | Why |
|---|---|
| One idea per passage | a chunk spanning three topics matches all three weakly and none strongly |
| Keep chunks 100–500 words | short chunks lose context, long chunks dilute the embedding |
| Overlap 10–20% at boundaries | a fact split across two chunks is retrievable from neither |
| Prefix each chunk with its document title or section | disambiguates otherwise identical passages |
| Chunk deterministically, and version the rule | a re-chunk changes every index and every recall number |

## Index versioning

Persist three things together, or a stale index will silently answer from the
wrong corpus:

```
retrieval-index/
  index files             # written by search.save(...)
  MANIFEST.json           # embedding model id, chunk rule version, corpus hash, built-at
```

On load, compare the manifest against the current corpus hash and embedding
model. Mismatch means rebuild — never "probably fine".

## Evaluating retrieval

```python
retrieval_score = dspy.Evaluate(devset=devset, metric=recall_at_k, num_threads=8)(rag)
answer_score = dspy.Evaluate(devset=devset, metric=answer_metric, num_threads=8)(rag)
```

`dspy.Evaluate` is keyword-only and its result exposes `.score`. Build the
devset so each example carries both the question and the gold passage ids or
texts; without gold passages recall@k cannot be computed, and you are back to a
single uninterpretable number.

Other useful retrieval metrics:

| Metric | When it is the right one |
|---|---|
| recall@k | the generator needs all the gold passages |
| MRR | one correct passage is enough, and its rank matters |
| precision@k | context budget is tight and noise costs accuracy |
| hit rate | a coarse smoke test across a large devset |

## Multi-hop notes

- Cap hops at 2–3; beyond that, added recall rarely offsets the latency and the context growth.
- Deduplicate between hops and preserve order (`dict.fromkeys`).
- Log the generated query per hop. A degenerate second query that merely restates the first is the most common multi-hop failure, and it is invisible in the final answer.
- Optimize the query generator with the retrieval metric, not the answer metric — that is the predictor recall@k actually measures.

## Legacy retrievers

`dspy.Retrieve(k=...)` reads the globally configured `rm`, set with
`dspy.configure(rm=dspy.ColBERTv2(url=...))`. Both are present in current DSPy
and raise no deprecation warning. Use them only when following a tutorial that
assumes them, or when a hosted ColBERTv2 endpoint is genuinely the corpus.
Everything else is better served by a callable.
