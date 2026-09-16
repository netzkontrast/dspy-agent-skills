"""dspy-drg-kg — runnable smoke test.

Builds a DRG schema, shows the extraction call shape, and encodes the two
things that most often waste a run: the install matrix (extraction needs the
[dspy] extra) and the silent empty-graph modes.

The dry run works with or without drg-kg installed. Without it the schema
logic and diagnostics still run; with it the real API surface is asserted.

Usage:
    uv run python example_drg_kg.py --dry-run
    uv run --with "drg-kg[dspy]" python example_drg_kg.py --dry-run
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import os

PACKAGE = "drg"
MAX_TEXT_CHARS = 100_000
WINDOWED_CHUNK_THRESHOLD = 6
WINDOWED_ENTITY_THRESHOLD = 25
STRICT_ENV_VARS = ("DRG_REQUIRE_LM", "DRG_PRODUCTION", "DRG_STRICT")

SAMPLE = (
    "Marie Curie worked for the University of Paris. "
    "She and Pierre Curie discovered polonium in 1898."
)


def package_available() -> bool:
    return importlib.util.find_spec(PACKAGE) is not None


def extraction_available() -> bool:
    """Extraction needs DSPy; the base drg-kg install deliberately omits it."""
    return package_available() and importlib.util.find_spec("dspy") is not None


def build_schema():
    """A schema is the contract: nothing outside it survives extraction."""
    from drg import EnhancedDRGSchema, EntityType, Relation, RelationGroup

    return EnhancedDRGSchema(
        entity_types=[
            EntityType(name="Person", description="A named human being."),
            EntityType(name="Organization", description="A company, agency or institution."),
            EntityType(name="Discovery", description="A scientific finding or named substance."),
        ],
        relation_groups=[
            RelationGroup(name="affiliation", relations=[
                Relation(name="works_for", src="Person", dst="Organization"),
            ]),
            RelationGroup(name="achievement", relations=[
                Relation(name="discovered", src="Person", dst="Discovery"),
            ]),
        ],
    )


def extract_to_graph(text: str, schema):
    """The canonical path: text -> (entities, triples) -> EnhancedKG."""
    from drg import extract_typed
    from drg.graph.builders import build_enhanced_kg

    entities, triples = extract_typed(text, schema)
    return build_enhanced_kg(entities_typed=entities, triples=triples,
                             schema=schema, source_text=text)


def diagnose_empty_graph(*, has_dspy: bool, lm_configured: bool,
                         strict_env: bool, text_chars: int,
                         relations_in_schema: int) -> str:
    """Map an empty or thin graph to its actual cause."""
    if not has_dspy:
        return "DSPy missing: install 'drg-kg[dspy]' — the base install cannot extract."
    if text_chars > MAX_TEXT_CHARS:
        return f"Text over {MAX_TEXT_CHARS} chars raises ValueError; chunk and use extract_from_chunks."
    if not lm_configured and not strict_env:
        return ("No LM configured: extraction returns ([], []) with only a log warning. "
                f"Set one of {', '.join(STRICT_ENV_VARS)}=1 so it raises instead.")
    if not lm_configured:
        return "No LM configured, but a strict env var is set, so this raises LLMConfigError."
    if relations_in_schema == 0:
        return "Schema declares no relations; every triple is filtered out as schema-invalid."
    return "Schema and LM look fine; check pruning and negation filters."


def windowed_extraction_expected(n_chunks: int, n_entities: int) -> bool:
    """DRG silently switches to windowed relation extraction, multiplying LLM calls."""
    return n_chunks >= WINDOWED_CHUNK_THRESHOLD or n_entities >= WINDOWED_ENTITY_THRESHOLD


def assert_api_surface() -> None:
    """Assert the surface this skill teaches, when the package is installed."""
    import drg
    from drg import EntityType, extract_typed
    from drg.graph.builders import build_enhanced_kg
    from drg.graph.kg_core import EnhancedKG

    params = inspect.signature(extract_typed).parameters
    assert params["enable_coreference_resolution"].default is False, (
        "coreference default changed; the skill says it is off"
    )
    assert params["enable_implicit_relationships"].default is True, (
        "implicit relationships default changed; the skill says it is on"
    )
    assert params["filter_negated"].default is True
    assert "lm" in params, "per-call lm injection is gone"

    builder = inspect.signature(build_enhanced_kg).parameters
    assert all(p.kind is inspect.Parameter.KEYWORD_ONLY
               for n, p in builder.items()), "build_enhanced_kg is no longer keyword-only"
    assert {"entities_typed", "triples", "schema"} <= set(builder)

    # Validation, versioning and diff are modules, not EnhancedKG methods.
    for absent in ("validate", "diff", "version"):
        assert not hasattr(EnhancedKG, absent), (
            f"EnhancedKG gained .{absent}(); the skill says it lives in drg.graph.*"
        )
    for present in ("query", "to_json", "save_json", "add_node", "add_edge"):
        assert hasattr(EnhancedKG, present), f"EnhancedKG lost .{present}()"

    assert "description" in inspect.signature(EntityType).parameters
    assert hasattr(drg, "generate_schema_from_text")


def dry_run() -> None:
    print(f"install: package={package_available()} extraction_ready={extraction_available()}")

    cases = [
        ("no dspy extra", dict(has_dspy=False, lm_configured=False, strict_env=False,
                               text_chars=500, relations_in_schema=2)),
        ("no LM, lenient", dict(has_dspy=True, lm_configured=False, strict_env=False,
                                text_chars=500, relations_in_schema=2)),
        ("no LM, strict", dict(has_dspy=True, lm_configured=False, strict_env=True,
                               text_chars=500, relations_in_schema=2)),
        ("huge document", dict(has_dspy=True, lm_configured=True, strict_env=True,
                               text_chars=250_000, relations_in_schema=2)),
        ("empty schema", dict(has_dspy=True, lm_configured=True, strict_env=True,
                              text_chars=500, relations_in_schema=0)),
    ]
    for label, kwargs in cases:
        print(f"{label:16s} -> {diagnose_empty_graph(**kwargs)}")
    assert "drg-kg[dspy]" in diagnose_empty_graph(**cases[0][1])
    assert "raises" in diagnose_empty_graph(**cases[2][1])

    for chunks, entities in ((2, 8), (6, 8), (2, 30)):
        flag = windowed_extraction_expected(chunks, entities)
        print(f"chunks={chunks:<3d} entities={entities:<3d} -> windowed={flag}")
    assert not windowed_extraction_expected(2, 8)
    assert windowed_extraction_expected(6, 8) and windowed_extraction_expected(2, 30)

    if package_available():
        assert_api_surface()
        schema = build_schema()
        relations = schema.get_all_relations()
        print(f"package installed: schema with {len(schema.entity_types)} entity types, "
              f"{len(relations)} relations; API surface asserted")
        assert len(relations) == 2
    else:
        print("package not installed: skipped live API assertions "
              "(pip install 'drg-kg[dspy]')")
    print("dry run OK")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Skip all LM calls")
    args = ap.parse_args()
    if args.dry_run:
        dry_run()
        return
    if not any(os.environ.get(v) for v in STRICT_ENV_VARS):
        os.environ["DRG_REQUIRE_LM"] = "1"      # fail loudly instead of returning an empty graph
    schema = build_schema()
    kg = extract_to_graph(SAMPLE, schema)
    print(f"nodes={len(kg.nodes)} edges={len(kg.edges)}")
    kg.save_json("graph.json")
    query = kg.query()
    print("entities:", [n for n in list(kg.nodes)[:5]])
    print("saved graph.json; validate it with: drg validate graph.json")


if __name__ == "__main__":
    main()
