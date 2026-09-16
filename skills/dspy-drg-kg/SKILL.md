---
name: dspy-drg-kg
description: >-
  Extract a typed, explainable knowledge graph from text with DRG (PyPI drg-kg,
  imported as drg) — declare or infer a schema, extract entities and relations
  against it, build an EnhancedKG, then query, validate, version, diff and
  export it deterministically. Covers the schema-first contract, the
  keyword-only builder, the silent-failure modes that return an empty graph,
  the windowed extraction that quietly multiplies LLM calls on long documents,
  and the boundary DRG draws around itself: extraction, not retrieval.
when_to_use: >-
  User says "knowledge graph", "extract entities and relations", "build a KG",
  "triples", "ontology", "graph from documents", "typed extraction";
  a corpus must become a queryable graph; a wiki or canon store needs typed
  entities rather than prose; graph provenance or versioning is required.
---

# DRG Knowledge-Graph Extraction (drg-kg, alpha)

DRG turns text into an inspectable `EnhancedKG` by combining a **declarative
schema** with DSPy-backed extraction. Its own boundary statement matters and is
easy to get wrong: DRG is **not** a GraphRAG, RAG or retrieval stack. It
produces a graph artifact. Serving that graph to a question is a different
job — see `dspy-retrieval` for the retrieval half.

Install is where most first attempts fail:

```bash
pip install "drg-kg[dspy]"     # PyPI name drg-kg, import name drg
```

The base install deliberately does **not** pull DSPy, so `pip install drg-kg`
alone gives you a package that cannot extract anything. The graph, validation
and evaluation layers work without it; extraction does not.

## The schema is the contract

Everything downstream filters against the schema, so a thin schema silently
produces a thin graph.

```python
from drg import EnhancedDRGSchema, EntityType, Relation, RelationGroup

schema = EnhancedDRGSchema(
    entity_types=[
        EntityType(name="Person", description="A named human being."),
        EntityType(name="Organization", description="A company, agency or institution."),
    ],
    relation_groups=[
        RelationGroup(name="affiliation", relations=[
            Relation(name="works_for", src="Person", dst="Organization"),
        ]),
    ],
)
```

`EntityType.description` is **required and non-empty** — an empty one raises
`SchemaError`. A `RelationGroup` must contain at least one relation. Let the
model propose a schema when the domain is unknown:

```python
from drg import generate_schema_from_text
schema = generate_schema_from_text(corpus_sample)     # then review it by hand
```

Treat an inferred schema as a draft. It decides what can ever be extracted.

## Extraction path

```python
from drg import extract_typed
from drg.graph.builders import build_enhanced_kg

entities, triples = extract_typed(text, schema)
kg = build_enhanced_kg(entities_typed=entities, triples=triples,
                       schema=schema, source_text=text)
kg.save_json("graph.json")
```

`build_enhanced_kg` lives in `drg.graph.builders`, is **not** exported at top
level, and is **keyword-only**. For documents past one chunk, use
`extract_from_chunks(chunks, schema)` instead, which adds cross-chunk relations.

Useful `extract_typed` switches: `enable_coreference_resolution` defaults to
**False**, `enable_implicit_relationships` defaults to **True** (it adds
LLM-inferred edges you did not state), `filter_negated=True` drops negated
relations, and `min_confidence` gates the output. Pass `lm=` to scope a model
per call instead of mutating global DSPy config.

## Why your graph came back empty

This is the single most common failure, and by default it is silent:

| Symptom | Cause | Fix |
|---|---|---|
| Empty graph, only a log warning | No DSPy LM configured — extraction returns `([], [])` | Set `DRG_REQUIRE_LM=1` (or `DRG_PRODUCTION=1`/`DRG_STRICT=1`) so it raises instead |
| `ImportError` on `import drg.extract` | DSPy not installed | `pip install "drg-kg[dspy]"` |
| Entities present, relations missing | Triples the schema does not allow are dropped | Widen `relation_groups`; check `is_valid_relation` |
| Nodes disappear | Isolated nodes are pruned by default | `prune_isolated_nodes=False` |
| `ValueError` on a large document | Text over 100,000 chars | Chunk it and use `extract_from_chunks` |

Set the strict env var in any pipeline. A knowledge base that silently ingests
nothing is worse than one that fails.

## Querying, validating, versioning

```python
q = kg.query()                                   # -> drg.query.GraphQuery
q.entity("Marie Curie"); q.neighbors(...); q.find_paths(...)
q.explain(...); q.evidence_for(...)              # provenance, not generation
q.centrality(); q.pagerank(); q.relations_active_at(...)
```

Query is **deterministic graph traversal**, not an LLM call. That is the point:
`explain` and `evidence_for` trace a claim back to its source span.

Validation, versioning and diff are **module-level, not `EnhancedKG` methods** —
a natural assumption that does not hold:

```python
from drg.graph.validation import validate_graph_file      # -> ValidationReport
from drg.graph.versioning import create_snapshot, list_versions, diff_versions, rollback_to_version
from drg.graph.diff import diff_graph_data                # dicts in, SnapshotDiff out
```

Versioning and diff operate on **file paths and JSON dicts**, not live objects.

## Cost surprises

Long documents auto-switch to windowed relation extraction at 6 or more chunks,
or 25 or more entities, and then issue up to 160 candidate pairs across 3
evidence windows. That is a large multiplier arriving without a flag. The knobs
are env vars: `DRG_WINDOWED_RELATION_CHUNK_THRESHOLD`,
`DRG_WINDOWED_RELATION_ENTITY_THRESHOLD`, `DRG_MAX_RELATION_CANDIDATE_PAIRS`,
`DRG_MAX_RELATION_EVIDENCE_WINDOWS`. Measure on one document before a corpus.

## CLI

```bash
drg extract input.txt -o graph.json --auto-schema --output-format enhancedkg
drg extract new.txt --update graph.json --update-strategy union --diff-output d.json
drg validate graph.json
drg versions list graph.json && drg versions diff graph.json v1 v2
drg eval run dataset.json && drg eval compare baseline.json candidate.json
```

## Anti-patterns

- `pip install drg-kg` without `[dspy]`, then reporting that extraction is broken.
- Running without `DRG_REQUIRE_LM=1` in a pipeline, so a missing key yields an empty graph and a green run.
- Accepting `generate_schema_from_text` output unreviewed — it silently bounds everything you can ever extract.
- Calling `kg.validate()` or `kg.diff()`; those live in `drg.graph.*` and take paths or dicts.
- Treating DRG as a RAG system. It builds the graph; retrieval and answering are yours.
- Ingesting a corpus before measuring one long document's windowed-extraction cost.
- Assuming `enable_implicit_relationships=True` is neutral; it adds inferred edges to your canon.

## Where to go next

- Retrieval over the corpus the graph was built from → `dspy-retrieval`
- Repairing a knowledge base that cannot answer a question → `dspy-deep-refine`
- Metrics for extraction quality → `dspy-evaluation-harness`
- Optimizing the extractor (`drg.optimizer`, opt-in) → `dspy-optimizer-selection`
- Full reference (exports, signatures, extras, env vars, stability tiers) → [reference.md](reference.md)
- Runnable, dry-run capable → [example_drg_kg.py](example_drg_kg.py)
