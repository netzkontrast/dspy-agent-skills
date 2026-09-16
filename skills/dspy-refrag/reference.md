# DSPy REFRAG — Reference

Source: `dspy-refrag` 0.1.0 (MIT), read from `src/dspy_refrag/` at
`github.com/netzkontrast/dspy-refrag`. Signatures are the real `def` lines.
Claims marked **verified** were confirmed against the named file and line.

Python `>=3.11`. Not on PyPI: `pip install git+https://github.com/netzkontrast/dspy-refrag`.

## Exports

`__init__.py` `__all__`: `REFRAGModule`, `REFRAGContext`, `SimpleRetriever`,
`Sensor`, `VectorAwareSerializer`, `VectorQuantizer`, `Fragment`,
`JSONFragmentSerializer`, `PickleSerializer`, `UnifiedSerializer`,
`ProtobufStyleSerializer`, `FAISSRetriever`, `PineconeRetriever`,
`PSQLRetriever`.

Reachable only by submodule path: `AdvancedSensor`, `SelectionConfig`,
`SelectionStrategy` (`.sensor_advanced`), `WeaviateRetriever`
(`.weaviate_retriever`), `Passage`, `make_ollama_embedder` (`.common`),
`MsgPackSerializer` (`.serializer_msgpack`), `DSPyPayloadSerializer`
(`.serializer_payload`), `Retriever` ABC (`.retriever`),
`build_corpus_from_data` (`.data_ingest`).

## Signatures

```python
REFRAGModule(retriever=None, lm=None, sensor=None, k=5, budget=2,
             lm_model="gpt-3.5-turbo", api_key=None, **kwargs)
REFRAGModule.forward(query: str) -> REFRAGContext

REFRAGContext(query, chunk_vectors=None, chunk_metadata=None, answer=None)

SimpleRetriever(embed_dim=768, corpus=None, embedder=None)
Sensor(mode="heuristic", threshold=None, learned_weights=None)
Sensor.select(query_vec, chunk_vecs, budget=2) -> list[int]

AdvancedSensor(config: SelectionConfig | None = None)
AdvancedSensor.select(query_vec, chunk_vecs, budget=2, metadata=None) -> list[int]

Passage(text, vector: np.ndarray, metadata: dict)          # what retrieval uses
Fragment(text, embedding, metadata, fragment_id, parent_doc_id=None)   # what serializers use

build_corpus_from_data(embedder, data_dir, max_chars=1500, overlap=200) -> list[Passage]
make_ollama_embedder(api_endpoint="http://localhost:11434",
                     model="nomic-embed-text:latest", normalize=True)
```

`Sensor(mode=...)` asserts `mode in ("heuristic", "learned")`. `"learned"`
without `learned_weights` silently falls back to heuristic.

## `SelectionConfig`

| Field | Default | Applies to |
|---|---|---|
| `strategy` | — | which `SelectionStrategy` member |
| `diversity_lambda` | 0.5 | `MMR` — higher favours diversity over relevance |
| `temperature` | 1.0 | `UNCERTAINTY` |
| `ensemble_weights` | None | `ENSEMBLE` |
| `adaptive_percentile` | 0.75 | `ADAPTIVE` |
| `min_score` | None | all — floor below which nothing is selected |

`SelectionStrategy`: `SIMILARITY`, `MMR`, `UNCERTAINTY`, `ENSEMBLE`, `ADAPTIVE`.

## Retriever backends

| Class | State | Needs |
|---|---|---|
| `SimpleRetriever` | works | an `embedder`, or it returns random vectors |
| `PSQLRetriever(db_url, embedder, table_name="passages")` | works | Postgres + pgvector; runs `CREATE EXTENSION`/`CREATE TABLE` in `__init__`; **`vector(768)` hardcoded** |
| `WeaviateRetriever(collection_name="DSPyRefrag", ...)` | code present, unusable as pinned | Weaviate v4 client; pyproject pins `<4.0` |
| `FAISSRetriever(index_path, embedder)` | **stub** | `__init__` raises `NotImplementedError` (faiss_retriever.py:53) |
| `PineconeRetriever(api_key, index_name, embedder)` | **stub** | raises `NotImplementedError` |

`WeaviateRetriever` does not subclass the `Retriever` ABC. Neither `faiss` nor
`pinecone` appears in the dependency list, consistent with both being stubs.

## Verified defects

| # | Claim | Evidence |
|---|---|---|
| 1 | Selection does not reduce prompt size | `refrag.py` builds `context_str` by joining **all** passages with a `(selected: bool)` annotation |
| 2 | FAISS backend unusable | `faiss_retriever.py:53` and `:83` raise `NotImplementedError` |
| 3 | Pinecone backend unusable | same pattern in `pinecone_retriever.py` |
| 4 | Import requires psycopg2 | `__init__.py:15` imports `.psql_retriever`, whose line 10 is `import psycopg2` |
| 5 | Weaviate pin conflicts with its code | `pyproject.toml:23` pins `weaviate-client>=3.25,<4.0`; `weaviate_retriever.py:15-16` imports v4-only `weaviate.classes.*` |
| 6 | `p['text']` unguarded | `forward` indexes `p['text']` outside the try block; default corpus metadata is `{"id": "a"}` |
| 7 | `lm_model` overridden | `__init__` calls `maybe_configure_openrouter_env()`, which applies `$OPENROUTER_MODEL` and mutates `os.environ` |
| 8 | LM errors swallowed | failures return `answer = f"Error calling LM: {e}"` instead of raising |

README statements not supported by the code: `REFRAGModule.add_memory()` does
not exist; the documented default `k=3` is actually `k=5`;
`REFRAGContext.metadata` is really `chunk_metadata`; the "protobuf" serializer
emits plain dicts with no wire format; "horizontal scaling" and "enterprise
grade" correspond to no retry, pooling, batching or async anywhere, and
`PSQLRetriever` opens one raw connection closed in `__del__`.

## Serializers

All take `Fragment`; the retrieval path produces `Passage`; no converter
exists, so the serializer layer is effectively disconnected. `REFRAGModule`
constructs a `VectorAwareSerializer` it never calls.

| Class | Notes |
|---|---|
| `JSONFragmentSerializer(normalize_vectors=True)` | portable, debuggable |
| `MsgPackSerializer()` | raises `RuntimeError` at construction without msgpack |
| `PickleSerializer(protocol=HIGHEST)` | fast, Python-only, unsafe across trust boundaries |
| `UnifiedSerializer(default_format="json")` | dispatcher over json/msgpack/pickle |
| `VectorAwareSerializer` | thin wrapper guaranteeing normalization |
| `ProtobufStyleSerializer` | dicts shaped `{"__type__","__version__","data"}`; not protobuf |
| `VectorQuantizer.quantize()` | explicit no-op placeholder that returns its input |

## Dependencies and environment

All dependencies are mandatory; there are no optional groups. Notably `dspy>=3.0.3`,
`numpy>=2`, `psycopg2-binary`, `weaviate-client>=3.25,<4.0`, `pandas`,
`matplotlib`, `pypdf`, `anthropic`, `openai`, `google-cloud-aiplatform`,
`google-genai`, plus `pyright` and `pytest` as runtime deps.

Env vars: `OPENROUTER_API_KEY`, `OPENROUTER_MODEL`, `OPENROUTER_BASE_URL`,
`OPENAI_API_KEY`, `OPENAI_API_BASE`, `MODEL_NAME`, `OLLAMA_BASE_URL`,
`OLLAMA_EMBED_MODEL`.

## What to lift

`sensor_advanced.py` is self-contained, dependency-light and genuinely useful;
it is the piece worth vendoring under MIT with attribution.
`data_ingest.build_corpus_from_data` is a reasonable PDF-to-chunks helper, with
the caveat that `examples/quickstart.py` duplicates its logic inline rather
than importing it.

Everything else is better served by `dspy.Embeddings` — see `dspy-retrieval`.
