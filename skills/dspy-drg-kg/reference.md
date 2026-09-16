# DRG Knowledge-Graph Extraction — Reference

Source: `drg-kg` (MIT, alpha), verified by importing the package and reading
`inspect.signature`. PyPI name `drg-kg`; import name `drg`. The unrelated PyPI
package `drg` is a Medicare grouper — installing it is a common mistake.

Python `>=3.10,<3.14`.

## Install matrix

| Command | What works |
|---|---|
| `pip install drg-kg` | schema classes, graph, validation, versioning, evaluation |
| `pip install "drg-kg[dspy]"` | the above **plus extraction** |
| `pip install "drg-kg[extract]"` | adds `tiktoken` on top of `[dspy]` |
| extras | `neo4j`, `api`, `mcp`, `openai`, `gemini`, `openrouter`, `local`, `louvain`, `leiden`, `spectral`, `networkx`, `coreference`, `all`, `dev` |

## Exports

`drg/__init__.py` imports `KG` and the schema classes eagerly; everything else
is lazy through `__getattr__`. In `__all__`: `extract_typed`,
`extract_typed_async`, `extract_triples`, `extract_from_chunks`,
`extract_from_chunks_async`, `generate_schema_from_text`, `KGExtractor`,
`create_kgedge_from_triple`, the confidence classes, the event classes, and the
query classes.

`EnhancedKG`, `KGNode`, `KGEdge` and `Cluster` are reachable but **not** in
`__all__`. Import them from `drg.graph.kg_core`. `build_enhanced_kg` is in
`drg.graph.builders` and is not top-level at all.

## Signatures

```python
extract_typed(text, schema, enable_entity_resolution=True,
              enable_coreference_resolution=False,
              enable_implicit_relationships=True, embedding_provider=None,
              return_enriched=False, min_confidence=None, filter_negated=True,
              enable_reverse_relation_fallback=False, lm=None)

extract_from_chunks(chunks, schema, enable_cross_chunk_relationships=True,
                    enable_entity_resolution=True,
                    enable_coreference_resolution=False,
                    enable_implicit_relationships=True,
                    enable_cross_chunk_context_snippets=True,
                    max_cross_chunk_context_chunks=3, two_pass_extraction=True,
                    ..., lm=None)

build_enhanced_kg(*, entities_typed, triples, schema=None, source_text=None,
                  enriched_relations=None, confidence_strategy="default",
                  entity_confidences=None, relation_confidences=None,
                  entity_properties=None, document_id=None, events=None,
                  name_mapping=None, entity_aliases=None,
                  filter_against_schema=True, prune_isolated_nodes=True,
                  filter_redundant_relations=True) -> EnhancedKG

EntityType(name: str, description: str, examples: list[str] = [], properties: dict = {})
```

The `_async` variants are `asyncio.to_thread` shims, not native async I/O.

## `EnhancedKG` methods

`add_node`, `get_node`, `add_edge`, `add_cluster`, `canonicalize_entities`,
`add_entity_embeddings`, `to_dict`, `to_json`, `to_json_ld`,
`to_enriched_format`, `save_json`, `save_json_ld`, `save_enriched_format`,
`from_dict`, `from_enriched_relationships`, `load_json`, `query`.

Note what is absent: no `validate`, no `diff`, no `version`. Those are modules.

## Module-level graph operations

```python
drg.graph.validation.validate_graph_file(path) -> ValidationReport
drg.graph.validation.validate_graph_data(data, path="<memory>")
drg.graph.versioning.create_snapshot(kg, graph_path, *, operation="snapshot",
                                     document_id=None, versions_dir=None) -> GraphVersion
drg.graph.versioning.list_versions(graph_path, *, versions_dir=None)
drg.graph.versioning.diff_versions(graph_path, old_id, new_id, *, versions_dir=None) -> SnapshotDiff
drg.graph.versioning.rollback_to_version(graph_path, version_id, *, versions_dir=None)
drg.graph.diff.diff_graph_data(old: dict, new: dict) -> SnapshotDiff
```

## `GraphQuery`

Structure: `entity`, `find_entities`, `relations`, `neighbors`, `find_paths`,
`shortest_path`, `related_entities`, `community_of`, `search`, `query(text)`.
Provenance: `evidence_for`, `explain`, `events_for`. Graph metrics:
`centrality`, `pagerank`, `influence_scores`. Temporal:
`relations_active_at`, `role_holders_at`, `temporal_query`,
`temporal_timeline`, `changes_between`, `temporal_overlaps`,
`temporal_conflicts`, `entity_transitions`. Also
`GraphQuery.from_json(filepath)` to query a saved graph without rebuilding.

## How DSPy is used

`KGExtractor(dspy.Module)` builds five signatures **dynamically from the
schema**, each wrapped in `dspy.Predict`: `EntityExtraction`,
`RelationExtraction`, `DocumentRelationExtraction`,
`ImplicitRelationExtraction`, `CoreferenceResolution`. Schema generation adds
`SchemaGeneration`, `SchemaReview` and `SchemaCoverageAudit`.

Because signatures are generated from the schema, changing the schema changes
the prompts. A module-level extractor cache is rebuilt only when the schema
fingerprint or the injected `lm` changes.

Optimization is separate and opt-in:

```python
drg.optimizer.optimize_extractor(training_data, *, config=KGOptimizerConfig(...),
                                 extractor=None, schema=None)
```

Default `optimizer_type="bootstrap"` (also `labeled_few_shot`, `mipro`,
`copro`), composite metric weighted entities 0.6 / relations 0.4. Normal
extraction never compiles.

## Environment

LM: `DRG_MODEL` (default `openai/gpt-4o-mini`, LiteLLM-prefixed),
`DRG_BASE_URL`, `DRG_TEMPERATURE` (0.0), `DRG_MAX_TOKENS`, plus the provider
key (`OPENAI_API_KEY`, `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`,
`OPENROUTER_API_KEY`, `PERPLEXITY_API_KEY`).

Failure behaviour: `DRG_REQUIRE_LM`, `DRG_PRODUCTION`, `DRG_STRICT` — set any
to turn the silent empty-result path into a raised `LLMConfigError`.

Chunking: `DRG_CHUNK_SIZE`, `DRG_OVERLAP_RATIO`, `DRG_CHUNKING_STRATEGY`,
`DRG_MAX_TEXT_CHARS` (100000 hard limit).

Windowed relation extraction: `DRG_WINDOWED_RELATION_EXTRACTION` (auto),
`DRG_WINDOWED_RELATION_CHUNK_THRESHOLD` (6),
`DRG_WINDOWED_RELATION_ENTITY_THRESHOLD` (25),
`DRG_MAX_RELATION_CANDIDATE_PAIRS` (160),
`DRG_MAX_RELATION_EVIDENCE_WINDOWS` (3),
`DRG_MAX_IMPLICIT_CANDIDATE_PAIRS` (120).

Schema generation: `DRG_SCHEMA_MAX_TOKENS`, `DRG_SCHEMA_MAX_ENTITY_TYPES`,
`DRG_SCHEMA_MAX_RELATION_GROUPS`, `DRG_SCHEMA_MAX_RELATIONS`,
`DRG_SCHEMA_COVERAGE_PASS`.

## Supporting layers

- `drg.reasoning` — rule-based inference over an existing graph:
  `MultiDocumentReasoner(config=ReasoningConfig(...)).reason(kg, document_id=...)`.
  Rules: `PathBridgeRule`, `InverseRule`, `SymmetricRule`, `TransitiveRule`,
  `CompositionRule`. Inferred edges carry evidence links. Opt-in, pure stdlib.
- `drg.evaluation` — `BenchmarkRunner(...).evaluate(datasets, runner=fn)`,
  `compare_reports`, `render_markdown_report`, `evaluate_ontology`,
  `evaluate_graph_quality`. Declared stable for the alpha series.
- `drg.graph.neo4j_exporter` — `Neo4jConfig`, `Neo4jExporter`,
  `build_neo4j_sync_plan`. Needs the `[neo4j]` extra.
- `drg.graph.visualization_adapter` — `kg_to_cytoscape`, `kg_to_vis_network`,
  `kg_to_d3_json`, `provenance_to_cytoscape`. Pure Python, no extra needed.
- `drg.graph.hub_mitigation.apply_hub_relation_proxy_split` — splits dominant
  hub nodes, which otherwise distort centrality and path queries.
- `drg.graph.incremental` — `merge_graphs`, `GraphMerger`, `MergeStrategy`.

## Stability tiers (from `docs/public_api.md`)

**Stable for alpha:** `extract_typed`/`_async`, `extract_from_chunks`/`_async`,
`extract_triples`, the schema classes, `build_enhanced_kg`, `drg.evaluation`,
and CLI `extract | validate | diff | versions | eval`.

**Experimental:** optimizer internals, confidence calibration and its data
formats, long-document windowing knobs, MCP server internals, clustering
strategy classes, event extraction internals, API and UI server details.

Pin the version for anything reproducible and read `CHANGELOG.md` before
upgrading.
