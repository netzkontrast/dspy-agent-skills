"""dspy-context-engineering-book — runnable smoke test.

The skill is a router, so the example is the routing table plus a lookup.
The dry run checks that every chapter is represented, that the optimizer
notebooks cover the optimizer names this pack teaches, and that lookup by
topic returns a real path.

Usage:
    uv run python example_book_map.py --dry-run
    uv run python example_book_map.py --find "mlflow"
"""

from __future__ import annotations

import argparse

TOTAL_NOTEBOOKS = 57
TOTAL_CHAPTERS = 11

NOTEBOOKS: dict[str, list[str]] = {
    "chapter01": ["hello-dspy"],
    "chapter02": ["dspy-tour"],
    "chapter03": ["dspy-in-8-steps", "humanize-quickstart"],
    "chapter04": ["error-analysis-router", "hf-datasets", "kaggle-imdb",
                  "pii-synthesizer", "synthetic-distillation"],
    "chapter05": ["string-and-regex-metrics", "semantic-similarity", "bleu-rouge-f1",
                  "human-then-llm-judge", "rubric-and-multipredictor"],
    "chapter06": ["labeled-few-shot", "bootstrap-few-shot", "bootstrap-random-search",
                  "knn-few-shot", "copro", "miprov2", "simba", "gepa",
                  "gepa-expanded-dataset-experiment", "better-together",
                  "bootstrap-finetune", "ensemble", "quickstart-ai-detector"],
    "chapter07": ["modules-tour", "react-and-tools", "program-of-thought",
                  "codeact-and-rlm", "multi-stage-patterns", "parallel-and-majority",
                  "multimodal", "adapters"],
    "chapter08": ["react-basics", "framework-comparison", "mcp-integration",
                  "rag-inmemory", "rag-qdrant", "web-search-and-multihop",
                  "history-mem0-rlm"],
    "chapter09": ["invoice-extraction", "customer-service-rag", "financial-analyst",
                  "news-researcher", "blog-writer", "sentiment-classifier",
                  "video-generator"],
    "chapter10": ["mlflow-tracking", "fastapi-invoice-api", "dspyui-gradio"],
    "chapter11": ["landing-page-skill-optimizer", "image-cli-optimizer",
                  "skill-discovery-rlm", "test-agents-md", "clawsona-dspy"],
}

# Optimizer name as used in dspy-optimizer-selection -> chapter 6 notebook.
OPTIMIZER_NOTEBOOKS = {
    "LabeledFewShot": "labeled-few-shot",
    "BootstrapFewShot": "bootstrap-few-shot",
    "BootstrapFewShotWithRandomSearch": "bootstrap-random-search",
    "KNNFewShot": "knn-few-shot",
    "COPRO": "copro",
    "MIPROv2": "miprov2",
    "SIMBA": "simba",
    "GEPA": "gepa",
    "BetterTogether": "better-together",
    "BootstrapFinetune": "bootstrap-finetune",
    "Ensemble": "ensemble",
}

TOPIC_ALIASES = {
    "rag": "rag-inmemory", "vector db": "rag-qdrant", "qdrant": "rag-qdrant",
    "mcp": "mcp-integration", "agent": "react-basics", "tools": "react-and-tools",
    "multimodal": "multimodal", "image": "multimodal", "adapter": "adapters",
    "metric": "string-and-regex-metrics", "judge": "human-then-llm-judge",
    "rubric": "rubric-and-multipredictor", "dataset": "hf-datasets",
    "synthetic": "synthetic-distillation", "mlflow": "mlflow-tracking",
    "api": "fastapi-invoice-api", "ui": "dspyui-gradio", "rlm": "codeact-and-rlm",
    "memory": "history-mem0-rlm", "multi-hop": "web-search-and-multihop",
    "skills": "skill-discovery-rlm", "agents.md": "test-agents-md",
    "persona": "clawsona-dspy", "first program": "hello-dspy",
}


def path_of(notebook: str) -> str | None:
    for chapter, names in NOTEBOOKS.items():
        if notebook in names:
            return f"{chapter}/{notebook}.ipynb"
    return None


def find(query: str) -> list[str]:
    """Alias hit first, then substring match across every notebook name."""
    q = query.lower().strip()
    hits: list[str] = []
    if q in TOPIC_ALIASES:
        hits.append(path_of(TOPIC_ALIASES[q]))
    for chapter, names in NOTEBOOKS.items():
        for name in names:
            candidate = f"{chapter}/{name}.ipynb"
            if q in name and candidate not in hits:
                hits.append(candidate)
    return [h for h in hits if h]


def notebook_for_optimizer(optimizer: str) -> str | None:
    name = OPTIMIZER_NOTEBOOKS.get(optimizer)
    return path_of(name) if name else None


def dry_run() -> None:
    total = sum(len(v) for v in NOTEBOOKS.values())
    print(f"chapters={len(NOTEBOOKS)} notebooks={total}")
    assert len(NOTEBOOKS) == TOTAL_CHAPTERS
    assert total == TOTAL_NOTEBOOKS, f"expected {TOTAL_NOTEBOOKS} notebooks, mapped {total}"
    assert all(names for names in NOTEBOOKS.values()), "a chapter has no notebooks"

    for optimizer in ("MIPROv2", "GEPA", "SIMBA", "Ensemble"):
        path = notebook_for_optimizer(optimizer)
        print(f"{optimizer:34s} -> {path}")
        assert path is not None, f"no notebook mapped for {optimizer}"
    missing = [o for o in OPTIMIZER_NOTEBOOKS if notebook_for_optimizer(o) is None]
    assert not missing, f"unmapped optimizers: {missing}"

    for query in ("rag", "mlflow", "gepa", "multimodal"):
        print(f"find({query!r}) -> {find(query)}")
        assert find(query), f"no hit for {query}"
    assert not find("nonexistent-topic-xyz")

    for alias, notebook in TOPIC_ALIASES.items():
        assert path_of(notebook), f"alias {alias!r} points at unknown notebook {notebook!r}"
    print(f"all {len(TOPIC_ALIASES)} topic aliases resolve to real notebooks")
    print("dry run OK")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Verify the routing table")
    ap.add_argument("--find", metavar="TOPIC", help="Find notebooks for a topic")
    ap.add_argument("--optimizer", metavar="NAME", help="Find the notebook for an optimizer")
    args = ap.parse_args()
    if args.find:
        hits = find(args.find)
        print("\n".join(hits) if hits else f"no notebook matches {args.find!r}")
        return
    if args.optimizer:
        print(notebook_for_optimizer(args.optimizer) or f"unknown optimizer {args.optimizer!r}")
        return
    dry_run()


if __name__ == "__main__":
    main()
