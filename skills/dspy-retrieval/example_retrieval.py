"""dspy-retrieval — runnable smoke test.

Builds the injected-retriever RAG shape, a deterministic stub retriever, the
recall@k metric and the retrieval-vs-answer diagnosis table. The dry run
exercises retrieval, deduplicated multi-hop and the metric without embeddings
or any LM call; the live path swaps in a real `dspy.Embeddings` index.

Usage:
    uv run python example_retrieval.py --dry-run
    OPENAI_API_KEY=... uv run python example_retrieval.py
"""

from __future__ import annotations

import argparse
import inspect
import os

DEFAULT_K = 3
MAX_HOPS = 2
MIN_TERM_LEN = 4
FAISS_THRESHOLD = 20_000          # dspy.Embeddings brute_force_threshold default

CORPUS = [
    "DSPy programs are composed from modules that carry signatures.",
    "MIPROv2 optimizes instructions and demonstrations with Bayesian search.",
    "GEPA reflects on textual feedback to rewrite predictor instructions.",
    "RLM explores large contexts with a sandboxed Python interpreter.",
    "The Embeddings retriever switches to a FAISS index above 20000 passages.",
    "BootstrapFewShot generates demonstrations with a teacher language model.",
]


def terms(text: str) -> set[str]:
    return {w.lower().strip(".,;:()") for w in text.split() if len(w) > MIN_TERM_LEN}


class StubRetriever:
    """Deterministic lexical retriever: a test double with the same contract."""

    def __init__(self, corpus: list[str], k: int = DEFAULT_K) -> None:
        self.corpus, self.k = corpus, k

    def __call__(self, query: str):
        import dspy

        wanted = terms(query)
        ranked = sorted(self.corpus, key=lambda p: -len(wanted & terms(p)))
        return dspy.Prediction(passages=ranked[: self.k])


def build_embeddings_retriever(corpus: list[str], k: int = DEFAULT_K):
    """The real retriever: hosted embedder plus an Embeddings index."""
    import dspy

    embedder = dspy.Embedder("openai/text-embedding-3-small")
    return dspy.Embeddings(corpus=corpus, embedder=embedder, k=k)


def build_rag(retrieve):
    import dspy

    class RAG(dspy.Module):
        def __init__(self, retrieve):
            super().__init__()
            self.retrieve = retrieve
            self.answer = dspy.ChainOfThought("context: list[str], question -> answer")

        def forward(self, question: str) -> dspy.Prediction:
            passages = self.retrieve(question).passages
            pred = self.answer(context=passages, question=question)
            return dspy.Prediction(context=passages, answer=pred.answer)

    return RAG(retrieve)


def build_multihop(retrieve, max_hops: int = MAX_HOPS):
    import dspy

    class MultiHopRAG(dspy.Module):
        def __init__(self, retrieve, max_hops):
            super().__init__()
            self.retrieve, self.max_hops = retrieve, max_hops
            self.next_query = dspy.ChainOfThought("context: list[str], question -> query")
            self.answer = dspy.ChainOfThought("context: list[str], question -> answer")

        def forward(self, question: str) -> dspy.Prediction:
            context, query = [], question
            for _ in range(self.max_hops):
                context = dedupe(context + self.retrieve(query).passages)
                query = self.next_query(context=context, question=question).query
            pred = self.answer(context=context, question=question)
            return dspy.Prediction(context=context, answer=pred.answer)

    return MultiHopRAG(retrieve, max_hops)


def dedupe(passages: list[str]) -> list[str]:
    """Order-preserving deduplication between hops."""
    return list(dict.fromkeys(passages))


def recall_at_k(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """Fraction of gold passages retrieval surfaced; GEPA-compatible return."""
    import dspy

    got = {p.strip() for p in pred.context}
    want = {p.strip() for p in gold.gold_passages}
    score = len(got & want) / max(len(want), 1)
    missing = want - got
    feedback = (f"Missed {len(missing)} gold passage(s): {sorted(missing)[:2]}"
                if missing else "All gold passages retrieved.")
    return dspy.Prediction(score=score, feedback=feedback)


def diagnose(recall: float, answer_accuracy: float, threshold: float = 0.6) -> str:
    """The retrieval-vs-answer table, as a function."""
    good_recall, good_answer = recall >= threshold, answer_accuracy >= threshold
    if not good_recall and not good_answer:
        return "corpus, chunking or embedding model"
    if good_recall and not good_answer:
        return "the generator, or the context field type"
    if not good_recall and good_answer:
        return "answering from parameters, not the corpus — check for leakage"
    return "both components healthy"


def assert_api_surface() -> None:
    """Fail loudly if the retrieval API this skill teaches has drifted."""
    import dspy

    params = inspect.signature(dspy.Embeddings.__init__).parameters
    assert params["brute_force_threshold"].default == FAISS_THRESHOLD, (
        f"Embeddings.brute_force_threshold default changed from {FAISS_THRESHOLD}"
    )
    assert params["k"].default == 5 and params["normalize"].default is True
    assert hasattr(dspy.Embeddings, "save") and hasattr(dspy.Embeddings, "from_saved")
    assert hasattr(dspy, "EmbeddingsWithScores")
    assert "model" in inspect.signature(dspy.Embedder.__init__).parameters


def dry_run() -> None:
    import dspy

    assert_api_surface()
    retrieve = StubRetriever(CORPUS, k=DEFAULT_K)

    hits = retrieve("Which optimizer uses Bayesian search over instructions?")
    print(f"retrieved {len(hits.passages)}: {hits.passages[0][:60]}...")
    assert "MIPROv2" in hits.passages[0]

    # Multi-hop deduplication: the same query twice must not grow the context.
    merged = dedupe(retrieve("FAISS index").passages + retrieve("FAISS index").passages)
    assert len(merged) == DEFAULT_K, "dedupe failed; hop two re-added hop one's passages"
    print(f"two identical hops -> {len(merged)} unique passages (no growth)")

    gold = dspy.Example(question="Which optimizer uses Bayesian search?",
                        gold_passages=[CORPUS[1]]).with_inputs("question")
    scored = recall_at_k(gold, dspy.Prediction(context=hits.passages))
    print(f"recall@{DEFAULT_K}={scored.score:.2f} | {scored.feedback}")
    assert scored.score == 1.0

    missed = recall_at_k(gold, dspy.Prediction(context=[CORPUS[3]]))
    assert missed.score == 0.0 and "Missed" in missed.feedback
    print(f"miss case: recall={missed.score:.2f} | {missed.feedback[:60]}...")

    for recall, accuracy in ((0.3, 0.3), (0.9, 0.3), (0.3, 0.9), (0.9, 0.9)):
        print(f"recall={recall} answer={accuracy} -> {diagnose(recall, accuracy)}")
    print("dry run OK")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Skip all LM and embedding calls")
    ap.add_argument("--model", default=os.environ.get("DSPY_MODEL", "openai/gpt-4o-mini"))
    ap.add_argument("--stub-retriever", action="store_true",
                    help="Use the lexical stub instead of real embeddings")
    args = ap.parse_args()
    if args.dry_run:
        dry_run()
        return
    import dspy

    dspy.configure(lm=dspy.LM(args.model), track_usage=True)
    retrieve = StubRetriever(CORPUS) if args.stub_retriever else build_embeddings_retriever(CORPUS)
    question = "Which optimizer uses Bayesian search over instructions?"

    result = build_rag(retrieve)(question=question)
    print(f"answer: {result.answer}")
    print(f"context: {len(result.context)} passage(s)")

    multi = build_multihop(retrieve)(question=question)
    print(f"multi-hop context: {len(multi.context)} unique passage(s)")


if __name__ == "__main__":
    main()
