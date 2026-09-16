"""Scaffolding for the Kohärenz Protokoll `CanonRetriever` seam.

Design note plus scaffolding only: this module is inert. Nothing in Kohärenz
Protokoll imports it, and importing it here changes no behaviour there.

`tools/kpwiki/programs.py` already defines the seam and leaves it empty:

    CanonRetriever = Callable[[list[Claim]], str]

    def no_canon_retrieval(_claims: list[Claim]) -> str:
        return ""

    class SourceIngest(dspy.Module):
        def __init__(self, retrieve_canon: CanonRetriever = no_canon_retrieval):

So `CheckCanonConflict` currently never fires: with nothing retrieved, nothing
can conflict, and every ingest reports zero conflicts. This file sketches what
fills that callable, in the shape the plan's step 3 describes.

Two pieces, deliberately separable:

1. `select_mmr` — Maximal Marginal Relevance selection, the one genuinely
   reusable idea in `dspy-refrag` (MIT, `sensor_advanced.py`). Canon and codex
   passages are dense with near-duplicates, so plain top-k spends the context
   window restating one fact.
2. `CanonIndex` — the retrieval seam, with a deterministic lexical fallback so
   the ingest loop stays testable offline exactly as it is today.

Plan: docs/kohaerenz-protokoll-plugin-plan.md
Skills: dspy-retrieval (the retrieval half), dspy-refrag (why only one file
of that package is worth taking).

Usage once ported into KP:

    from tools.kpwiki.programs import SourceIngest
    from tools.kpwiki.retrieval import CanonIndex

    ingest = SourceIngest(retrieve_canon=CanonIndex.load("Plan/wiki/index/canon"))

Revert by passing `no_canon_retrieval` again.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, Sequence

DEFAULT_K = 12
DEFAULT_BUDGET = 5
# Upstream's default is 0.5. Measured on a near-duplicate fixture, 0.5 still
# returns the duplicate and only lambda >= 0.6 reaches a relevant-but-distinct
# passage. The codex is dense with entries restating one rule, so this pack
# defaults higher than upstream. Lower it if recall drops.
DEFAULT_DIVERSITY_LAMBDA = 0.65
# Relevance floor. Without it, MMR can prefer an irrelevant passage to a
# relevant one, because zero relevance with zero redundancy outscores high
# relevance with high redundancy. See select_mmr.
DEFAULT_MIN_RELEVANCE = 0.15
MIN_TERM_LENGTH = 4
PASSAGE_SEPARATOR = "\n\n---\n\n"


class ClaimLike(Protocol):
    """Structural view of `tools.kpwiki.schema.Claim` — no import needed here."""

    text: str
    entities: list[str]


@dataclass(frozen=True)
class CanonPassage:
    """One retrievable unit of canon, carrying the citation it must keep."""

    text: str
    source_file: str
    section: str = ""
    vector: tuple[float, ...] | None = None

    def cite(self) -> str:
        """Canon passages must arrive citable, or a conflict cannot be adjudicated."""
        where = f"{self.source_file}#{self.section}" if self.section else self.source_file
        return f"[{where}] {self.text}"


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def select_mmr(query_vec: Sequence[float], passage_vecs: Sequence[Sequence[float]],
               budget: int = DEFAULT_BUDGET,
               diversity_lambda: float = DEFAULT_DIVERSITY_LAMBDA,
               min_relevance: float = DEFAULT_MIN_RELEVANCE) -> list[int]:
    """Maximal Marginal Relevance with a relevance floor.

    Ported from dspy-refrag `sensor_advanced.py` (MIT), plus the floor that
    package exposes as `SelectionConfig.min_score`. Vendor this into KP with
    attribution rather than depending on that package — see the plan for why.

    `diversity_lambda` 0.0 is pure relevance (plain top-k); 1.0 is pure
    diversity. 0.5 is the upstream default and a sane start for a codex where
    several entries restate the same rule.

    **The floor is not optional.** Plain MMR scores a candidate
    `(1-λ)·relevance - λ·redundancy`. An irrelevant passage scores
    `0 - 0 = 0`, while a highly relevant near-duplicate scores slightly
    *below* zero. At λ=0.5 the unrelated passage therefore wins, which is the
    opposite of what a canon-conflict check needs: a duplicate merely wastes
    context, an unrelated passage invites a fabricated conflict. Candidates
    below `min_relevance` are excluded before selection.

    Set `min_relevance=0.0` only to reproduce unguarded upstream behaviour.

    Measured on a fixture of two near-duplicates, one relevant-but-distinct
    passage and one unrelated passage:

    | lambda | floor off | floor on |
    |---|---|---|
    | 0.5 | duplicate + **unrelated** | duplicate + duplicate |
    | 0.6-0.8 | duplicate + **unrelated** | duplicate + relevant-distinct |

    The floor changes the outcome at every lambda; lambda alone never fixes it.
    """
    eligible = [i for i in range(len(passage_vecs))
                if cosine(query_vec, passage_vecs[i]) >= min_relevance]
    if not eligible:                                  # nothing clears the floor
        return []
    chosen: list[int] = []
    remaining = list(eligible)
    while remaining and len(chosen) < budget:
        best, best_score = remaining[0], -math.inf
        for i in remaining:
            relevance = cosine(query_vec, passage_vecs[i])
            redundancy = max((cosine(passage_vecs[i], passage_vecs[j]) for j in chosen),
                             default=0.0)
            score = (1 - diversity_lambda) * relevance - diversity_lambda * redundancy
            if score > best_score:
                best, best_score = i, score
        chosen.append(best)
        remaining.remove(best)
    return chosen


def terms(text: str) -> set[str]:
    return {w.lower().strip(".,;:()[]\"'") for w in text.split() if len(w) > MIN_TERM_LENGTH}


def claims_to_query(claims: Sequence[ClaimLike]) -> str:
    """Build one retrieval query from a claim batch.

    Entities carry more retrieval signal than claim prose, because the codex is
    indexed by name. They go first and are repeated once to weight them.
    """
    entities = [e for claim in claims for e in getattr(claim, "entities", [])]
    bodies = [getattr(claim, "text", "") for claim in claims]
    return " ".join([*entities, *entities, *bodies]).strip()


@dataclass
class CanonIndex:
    """The `CanonRetriever` seam: `list[Claim] -> str`.

    Call it directly; an instance is the callable `SourceIngest` expects.

    With `embedder` set, retrieval is semantic and selection is MMR. Without
    one, it falls back to deterministic lexical overlap, so tests and dry runs
    behave exactly as they do today with `no_canon_retrieval` — only now they
    can also assert what was retrieved.
    """

    passages: list[CanonPassage] = field(default_factory=list)
    embedder: Callable[[list[str]], Any] | None = None
    k: int = DEFAULT_K
    budget: int = DEFAULT_BUDGET
    diversity_lambda: float = DEFAULT_DIVERSITY_LAMBDA
    min_relevance: float = DEFAULT_MIN_RELEVANCE

    def __call__(self, claims: Sequence[ClaimLike]) -> str:
        """Satisfies `CanonRetriever = Callable[[list[Claim]], str]`."""
        if not claims or not self.passages:
            return ""
        query = claims_to_query(claims)
        candidates = self._shortlist(query)
        selected = self._select(query, candidates)
        return PASSAGE_SEPARATOR.join(p.cite() for p in selected)

    def _shortlist(self, query: str) -> list[CanonPassage]:
        """Cheap first pass: k candidates, before the expensive selection."""
        wanted = terms(query)
        scored = sorted(self.passages,
                        key=lambda p: -len(wanted & terms(p.text)))
        return scored[: self.k]

    def _select(self, query: str, candidates: list[CanonPassage]) -> list[CanonPassage]:
        vectors = [p.vector for p in candidates]
        if self.embedder is None or any(v is None for v in vectors):
            return candidates[: self.budget]          # lexical fallback, deterministic
        query_vec = self._embed_one(query)
        picks = select_mmr(query_vec, [tuple(v) for v in vectors],
                           budget=self.budget, diversity_lambda=self.diversity_lambda,
                           min_relevance=self.min_relevance)
        return [candidates[i] for i in picks]

    def _embed_one(self, text: str) -> tuple[float, ...]:
        vectors = self.embedder([text])               # type: ignore[misc]
        first = vectors[0]
        return tuple(float(x) for x in first)

    # ── Build and load ──────────────────────────────────────────────────
    #
    # Deliberately unimplemented. Building the index is where the real
    # decisions live, and they are the plan's open questions: which trees to
    # index (Canon/ and Codex/ certainly; Manuscript/ is question 1), how to
    # chunk (per codex entry and per canon section, not per file), and how to
    # version the index against the corpus hash and the embedding model.
    #
    # See dspy-retrieval's reference for the chunking rules and the index
    # manifest, and build this with dspy.Embeddings.

    @classmethod
    def load(cls, index_dir: str, embedder: Callable[[list[str]], Any] | None = None) -> "CanonIndex":
        raise NotImplementedError(
            "Scaffold only. Implement in KP as tools/kpwiki/retrieval.py: read the "
            "persisted dspy.Embeddings index from index_dir, verify its manifest "
            "against the current corpus hash and embedding model, and return a "
            "CanonIndex. Until then, SourceIngest keeps its no_canon_retrieval default. "
            "See docs/kohaerenz-protokoll-plugin-plan.md step 3."
        )

    @classmethod
    def build(cls, canon_root: str, codex_root: str,
              embedder: Callable[[list[str]], Any] | None = None) -> "CanonIndex":
        raise NotImplementedError(
            "Scaffold only. Chunk Canon/ per section and Codex/ per entry, embed with "
            "dspy.Embeddings, and persist alongside a manifest recording the corpus "
            "hash, the chunking rule version and the embedding model id. "
            "See docs/kohaerenz-protokoll-plugin-plan.md step 3."
        )


__all__ = ["CanonIndex", "CanonPassage", "ClaimLike", "claims_to_query", "cosine", "select_mmr"]
