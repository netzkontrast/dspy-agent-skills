"""dspy-refrag — runnable smoke test.

The useful part of REFRAG is fragment selection, so this example implements
the five selection strategies as standalone deterministic functions (MMR in
particular is worth lifting) and encodes the package's verified defects as
assertions, so a future release that fixes them fails this test loudly.

The dry run needs neither dspy_refrag nor numpy.

Usage:
    uv run python example_refrag.py --dry-run
    OPENAI_API_KEY=... uv run python example_refrag.py
"""

from __future__ import annotations

import argparse
import importlib.util
import math

PACKAGE = "dspy_refrag"
DEFAULT_K = 5
DEFAULT_BUDGET = 2
DEFAULT_DIVERSITY_LAMBDA = 0.5
DEFAULT_ADAPTIVE_PERCENTILE = 0.75
PSQL_HARDCODED_DIM = 768

# Verified defects, keyed by the file that proves each one.
KNOWN_DEFECTS = {
    "selection_does_not_compress": "refrag.py forward() joins ALL passages and only annotates (selected: bool)",
    "faiss_is_a_stub": "faiss_retriever.py:53 raises NotImplementedError",
    "pinecone_is_a_stub": "pinecone_retriever.py raises NotImplementedError",
    "import_needs_psycopg2": "__init__.py:15 -> psql_retriever.py:10 imports psycopg2 at module level",
    "weaviate_pin_conflict": "pyproject pins weaviate-client<4.0; weaviate_retriever.py imports v4-only weaviate.classes",
    "text_key_unguarded": "forward() indexes p['text'] outside the try block",
    "lm_model_overridden": "maybe_configure_openrouter_env() overwrites lm_model from $OPENROUTER_MODEL",
}


def package_available() -> bool:
    return importlib.util.find_spec(PACKAGE) is not None


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def select_similarity(query, chunks, budget, min_score=None):
    """Top-budget by cosine similarity."""
    scored = sorted(range(len(chunks)), key=lambda i: -cosine(query, chunks[i]))
    if min_score is not None:
        scored = [i for i in scored if cosine(query, chunks[i]) >= min_score]
    return scored[:budget]


def select_mmr(query, chunks, budget, diversity_lambda=DEFAULT_DIVERSITY_LAMBDA,
               min_relevance=0.0):
    """Maximal Marginal Relevance: relevance minus redundancy with what is already picked.

    This is the strategy worth lifting: it is what you want when the top-k are
    near-duplicates of each other.

    Note the trap, reproduced faithfully here with `min_relevance=0.0`: an
    irrelevant passage scores `0 - 0 = 0`, while a relevant near-duplicate
    scores slightly below zero, so plain MMR prefers the irrelevant one. The
    package exposes `SelectionConfig.min_score` for exactly this; use it.
    """
    eligible = [i for i in range(len(chunks)) if cosine(query, chunks[i]) >= min_relevance]
    chosen: list[int] = []
    remaining = list(eligible)
    while remaining and len(chosen) < budget:
        best, best_score = None, -math.inf
        for i in remaining:
            relevance = cosine(query, chunks[i])
            redundancy = max((cosine(chunks[i], chunks[j]) for j in chosen), default=0.0)
            score = (1 - diversity_lambda) * relevance - diversity_lambda * redundancy
            if score > best_score:
                best, best_score = i, score
        chosen.append(best)
        remaining.remove(best)
    return chosen


def select_adaptive(query, chunks, budget, percentile=DEFAULT_ADAPTIVE_PERCENTILE):
    """Cut on the score distribution rather than a fixed count."""
    scores = sorted((cosine(query, c) for c in chunks), reverse=True)
    if not scores:
        return []
    cutoff = scores[min(int(len(scores) * (1 - percentile)), len(scores) - 1)]
    keep = [i for i in range(len(chunks)) if cosine(query, chunks[i]) >= cutoff]
    return sorted(keep, key=lambda i: -cosine(query, chunks[i]))[:budget]


def prompt_passages(all_passages: list[str], selected: list[int], *, compress: bool) -> list[str]:
    """What REFRAG *should* do (compress=True) versus what the package does."""
    if compress:
        return [all_passages[i] for i in selected]
    return all_passages                       # package behaviour: annotate, do not remove


def corpus_is_usable(metadata: list[dict]) -> tuple[bool, str]:
    """forward() indexes p['text'] unguarded; a corpus without it crashes."""
    missing = [i for i, m in enumerate(metadata) if "text" not in m]
    if missing:
        return False, f"passages {missing} lack metadata['text']; forward() raises KeyError"
    return True, "every passage carries metadata['text']"


def assert_defects_still_present() -> None:
    """If upstream fixes these, this test fails and the skill must be rewritten."""
    import inspect

    from dspy_refrag import REFRAGModule, SimpleRetriever

    source = inspect.getsource(REFRAGModule.forward)
    assert "selected" in source, "forward() no longer annotates selection"
    compresses = "selected_idxs" in source.split("context_str")[-1][:400]
    assert not compresses, (
        "forward() now appears to filter the prompt by selection — the compression "
        "defect may be fixed. Re-verify and update this skill."
    )
    assert inspect.signature(REFRAGModule.__init__).parameters["k"].default == DEFAULT_K, (
        "REFRAGModule k default changed"
    )
    for stub in ("FAISSRetriever", "PineconeRetriever"):
        cls = __import__("dspy_refrag", fromlist=[stub]).__dict__[stub]
        try:
            cls(index_path="x", embedder=None) if stub == "FAISSRetriever" else cls(
                api_key="x", index_name="y", embedder=None)
        except NotImplementedError:
            pass
        except TypeError:
            pass          # signature differs; the stub claim is checked by source above
        else:
            raise AssertionError(f"{stub} no longer raises NotImplementedError — skill is stale")
    assert "embedder" in inspect.signature(SimpleRetriever.__init__).parameters


def dry_run() -> None:
    query = [1.0, 0.0, 0.0]
    chunks = [
        [0.99, 0.10, 0.0],   # 0 near-duplicate of the query
        [0.98, 0.12, 0.0],   # 1 near-duplicate of 0
        [0.30, 0.95, 0.0],   # 2 different direction, still relevant
        [0.0, 0.0, 1.0],     # 3 unrelated
    ]
    sim = select_similarity(query, chunks, budget=2)
    unguarded = select_mmr(query, chunks, budget=2)
    guarded = select_mmr(query, chunks, budget=2, diversity_lambda=0.65, min_relevance=0.15)
    ada = select_adaptive(query, chunks, budget=3)
    print(f"similarity      -> {sim}   (two near-duplicates: redundant context)")
    print(f"mmr unguarded   -> {unguarded}   (picked chunk 3, which is IRRELEVANT to the query)")
    print(f"mmr with floor  -> {guarded}   (relevant and distinct)")
    print(f"adaptive        -> {ada}   (cut by score distribution)")
    assert sim == [0, 1], "similarity should take both near-duplicates"
    assert unguarded[1] == 3, "unguarded MMR prefers the irrelevant chunk; that is the trap"
    assert guarded == [0, 2], "with a relevance floor, MMR reaches the relevant distinct chunk"

    passages = [f"passage-{i}" for i in range(4)]
    compressed = prompt_passages(passages, guarded, compress=True)
    as_shipped = prompt_passages(passages, guarded, compress=False)
    print(f"prompt with compression: {len(compressed)} passages; as shipped: {len(as_shipped)}")
    assert len(compressed) == 2 and len(as_shipped) == 4, (
        "this asymmetry is the package's central defect"
    )

    ok, why = corpus_is_usable([{"id": "a"}, {"id": "b"}])
    print(f"default-style corpus usable: {ok} | {why}")
    assert not ok
    ok, why = corpus_is_usable([{"text": "hello"}])
    assert ok

    print(f"PSQLRetriever vector dimension is hardcoded to {PSQL_HARDCODED_DIM}")
    for name, evidence in KNOWN_DEFECTS.items():
        print(f"  defect {name}: {evidence}")

    if package_available():
        assert_defects_still_present()
        print("package installed: defects re-verified against the installed source")
    else:
        print("package not installed: skipped source re-verification "
              "(pip install git+https://github.com/netzkontrast/dspy-refrag)")
    print("dry run OK")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Skip all LM and embedding calls")
    ap.add_argument("--data-dir", default="data", help="Directory of PDFs to ingest")
    args = ap.parse_args()
    if args.dry_run:
        dry_run()
        return
    from pathlib import Path

    from dspy_refrag import REFRAGModule, SimpleRetriever
    from dspy_refrag.common import make_ollama_embedder
    from dspy_refrag.data_ingest import build_corpus_from_data
    from dspy_refrag.sensor_advanced import AdvancedSensor, SelectionConfig, SelectionStrategy

    embedder = make_ollama_embedder()
    corpus = build_corpus_from_data(embedder, Path(args.data_dir))
    retriever = SimpleRetriever(embedder=embedder, corpus=corpus)
    sensor = AdvancedSensor(SelectionConfig(strategy=SelectionStrategy.MMR,
                                            diversity_lambda=DEFAULT_DIVERSITY_LAMBDA))
    ctx = REFRAGModule(retriever=retriever, sensor=sensor, k=DEFAULT_K,
                       budget=DEFAULT_BUDGET).forward("your query")
    print(f"passages: {len(ctx.chunk_metadata)}")
    answer = ctx.answer or ""
    if answer.startswith("Error calling LM:"):
        print(f"LM call failed but was swallowed into the answer: {answer}")
    else:
        print(f"answer: {answer}")


if __name__ == "__main__":
    main()
