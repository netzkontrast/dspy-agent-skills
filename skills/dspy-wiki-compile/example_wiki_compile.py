"""dspy-wiki-compile — runnable smoke test.

Compile immutable sources into a maintained wiki with DSPy: triage → extract
claims with line-range citations → merge concepts across the whole batch
(two-phase: every source is extracted before any page is written) → decide
per existing page (flag / update / create, with active-page protection) →
knowledge diff → cited answers with named gaps. Patterns from the Karpathy
LLM-wiki (concept table, contradiction block), llm-wiki-agent (post-ingest
validation), llm-wiki-compiler (two-phase compile, citations), synthadoc
(decision rules, active-page protection, truncation flag) and quicky-wiki
(knowledge diff). The dry run exercises the deterministic metric on a
hand-built batch without any LM call.

Usage:
    uv run python example_wiki_compile.py --dry-run
    OPENAI_API_KEY=... uv run python example_wiki_compile.py
"""

from __future__ import annotations

import argparse
import os
import re
from typing import Literal

from pydantic import BaseModel, Field

Tier = Literal["primary", "secondary", "superseded", "duplicate", "out-of-scope"]
Action = Literal["flag", "update", "create"]
PageStatus = Literal["draft", "reviewed", "locked", "contested", "archived"]
ConceptStatus = Literal["high-confidence", "single-source", "tentative", "contradicted"]
Resolution = Literal["pending", "supersedes", "both-valid"]
PROTECTED_STATUSES = ("reviewed", "locked")
WEIGHTS = {"citations": 0.30, "decisions": 0.25, "merge": 0.20, "diffs": 0.15, "links": 0.10}
LANGUAGE_MARKERS = {"de": (" der ", " die ", " das ", " und ", " nicht ", " ist "),
                    "en": (" the ", " and ", " not ", " is ", " of ", " with ")}


class Citation(BaseModel):
    file: str
    start: int = Field(ge=1)
    end: int = Field(ge=1)
    quote: str


class Claim(BaseModel):
    text: str
    citation: Citation
    kind: Literal["fact", "definition", "rule", "event", "opinion"] = "fact"
    entities: list[str] = Field(default_factory=list)


class Triage(BaseModel):
    tier: Tier
    category: str
    language: str
    truncated: bool = False


class Extraction(BaseModel):
    source: str
    triage: Triage
    claims: list[Claim]


class PageState(BaseModel):
    slug: str
    status: PageStatus
    body: str


class Disagreement(BaseModel):
    topic: str
    sources: list[str]
    positions: list[str]
    resolution: Resolution = "pending"


class ConceptDraft(BaseModel):
    slug: str
    definition: str
    sources: list[str]
    citations: list[Citation]
    disagreements: list[Disagreement] = Field(default_factory=list)
    status: ConceptStatus


class IngestDecision(BaseModel):
    slug: str
    action: Action
    rationale: str
    conflicts: list[str] = Field(default_factory=list)


class Diff(BaseModel):
    reinforced: list[str] = Field(default_factory=list)
    challenged: list[str] = Field(default_factory=list)
    new: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)


class Answer(BaseModel):
    text: str
    cited_pages: list[str]
    confidence: Literal["high", "medium", "low"]
    gaps: list[str] = Field(default_factory=list)


class Compiled(BaseModel):
    extractions: list[Extraction]
    concepts: list[ConceptDraft]
    decisions: list[IngestDecision]
    diffs: dict[str, Diff]


# --- deterministic checks -------------------------------------------------------

def number_lines(text: str) -> str:
    return "\n".join(f"{i}: {line}" for i, line in enumerate(text.splitlines(), 1))


def citation_resolves(c: Citation, sources: dict[str, str]) -> bool:
    lines = sources.get(c.file, "").splitlines()
    if not lines or not 1 <= c.start <= c.end <= len(lines):
        return False
    return c.quote.strip() in "\n".join(lines[c.start - 1:c.end])


def decision_legal(d: IngestDecision, page: PageState | None) -> bool:
    """Active-page protection: a reviewed or locked page with conflicts is flagged, never updated."""
    if page is None:
        return d.action == "create"
    if d.action == "create":
        return False
    return not (d.action == "update" and page.status in PROTECTED_STATUSES and d.conflicts)


def _overlap(a: str, b: str) -> bool:
    tokens = lambda t: {w for w in re.findall(r"\w+", t.lower()) if len(w) > 2}  # noqa: E731
    return bool(tokens(a) & tokens(b))


def diff_consistent(diff: Diff, decision: IngestDecision, prior: str) -> list[str]:
    problems = [f"challenged item without a conflict: {x!r}" for x in diff.challenged
                if not any(_overlap(x, c) for c in decision.conflicts)]
    problems += [f"'new' item already on the page: {x!r}" for x in diff.new if x.lower() in prior.lower()]
    return problems


def unlinked_entities(text: str, known: list[str]) -> list[str]:
    return [e for e in known if re.search(rf"(?<!\w){re.escape(e)}(?!\w)", text) and f"[[{e}]]" not in text]


def language_kept(text: str, language: str) -> bool:
    padded = f" {text.lower()} "
    own = sum(padded.count(m) for m in LANGUAGE_MARKERS.get(language, ()))
    other = sum(padded.count(m) for lang, ms in LANGUAGE_MARKERS.items() if lang != language for m in ms)
    return own >= other


def _mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 1.0


def _citation_axis(run: Compiled, sources: dict[str, str], deficits: list[str]) -> float:
    cites = [c.citation for e in run.extractions for c in e.claims] + [c for k in run.concepts for c in k.citations]
    bad = [c for c in cites if not citation_resolves(c, sources)]
    deficits += [f"citation does not resolve: {c.file}:{c.start}-{c.end} {c.quote[:30]!r}" for c in bad]
    return 1.0 - len(bad) / len(cites) if cites else 0.0


def _decision_axis(run: Compiled, pages: dict[str, PageState], deficits: list[str]) -> float:
    scores = []
    for d in run.decisions:
        ok = decision_legal(d, pages.get(d.slug))
        scores.append(1.0 if ok else 0.0)
        if not ok:
            deficits.append(f"illegal decision: {d.action} on {d.slug} "
                            f"(status {pages[d.slug].status if d.slug in pages else 'absent'}, conflicts {d.conflicts})")
    return _mean(scores)


def _merge_axis(run: Compiled, deficits: list[str]) -> float:
    batch = {e.source for e in run.extractions}
    scores = []
    for k in run.concepts:
        checks = [set(k.sources) <= batch and len(k.sources) == len(set(k.sources)), bool(k.citations),
                  all(len(set(d.sources)) >= 2 for d in k.disagreements),
                  (k.status == "contradicted") == any(d.resolution == "pending" for d in k.disagreements)]
        scores.append(_mean(1.0 if c else 0.0 for c in checks))
        if not checks[2]:
            deficits.append(f"{k.slug}: a disagreement needs two distinct sources")
        if not checks[3]:
            deficits.append(f"{k.slug}: status {k.status} does not match its pending disagreements")
        if not checks[0] or not checks[1]:
            deficits.append(f"{k.slug}: sources outside the batch or no citation")
    return _mean(scores)


def _diff_axis(run: Compiled, pages: dict[str, PageState], deficits: list[str]) -> float:
    decisions = {d.slug: d for d in run.decisions}
    scores = []
    for slug, diff in run.diffs.items():
        problems = diff_consistent(diff, decisions[slug], pages[slug].body) if slug in decisions and slug in pages else []
        scores.append(0.0 if problems else 1.0)
        deficits += [f"{slug}: {p}" for p in problems]
    return _mean(scores)


def _link_axis(run: Compiled, known: list[str], language: str, deficits: list[str]) -> float:
    scores = []
    for k in run.concepts:
        missing = unlinked_entities(k.definition, known)
        lang_ok = language_kept(k.definition, language)
        scores.append(_mean([1.0 if not missing else 0.0, 1.0 if lang_ok else 0.0]))
        if missing:
            deficits.append(f"{k.slug}: known entities not linked: {missing}")
        if not lang_ok:
            deficits.append(f"{k.slug}: definition is not in the source language ({language})")
    return _mean(scores)


def compile_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """GEPA-compatible metric: every axis is deterministic and names its blame."""
    import dspy

    run: Compiled = pred.compiled
    deficits: list[str] = []
    axes = {"citations": _citation_axis(run, gold.sources, deficits),
            "decisions": _decision_axis(run, gold.pages, deficits),
            "merge": _merge_axis(run, deficits),
            "diffs": _diff_axis(run, gold.pages, deficits),
            "links": _link_axis(run, gold.known_entities, gold.language, deficits)}
    score = sum(WEIGHTS[a] * v for a, v in axes.items())
    return dspy.Prediction(score=round(score, 3),
                           feedback="; ".join(deficits) or "cited, legally decided, merged across sources, consistent diff")


# --- program -------------------------------------------------------------------

def build():
    import dspy

    class TriageSource(dspy.Signature):
        """Classify a source: tier, category, language; truncated when the body was capped."""

        source_name: str = dspy.InputField()
        text: str = dspy.InputField()
        triage: Triage = dspy.OutputField()

    class ExtractClaims(dspy.Signature):
        """Extract atomic claims; each cites the numbered lines it comes from with a verbatim quote.
        Never paraphrase into the quote; never add what the source does not say."""

        source_name: str = dspy.InputField()
        numbered_text: str = dspy.InputField()
        known_entities: list[str] = dspy.InputField()
        claims: list[Claim] = dspy.OutputField()

    class MergeConcepts(dspy.Signature):
        """Merge all extractions of the batch into concept drafts. One draft per concept, sources listed,
        every definition sentence backed by a citation, disagreements listed with all sources and
        resolution 'pending' unless one source explicitly supersedes; known entities as [[links]];
        keep the source language."""

        extractions: list[Extraction] = dspy.InputField()
        known_entities: list[str] = dspy.InputField()
        concepts: list[ConceptDraft] = dspy.OutputField()

    class DecideIngest(dspy.Signature):
        """RULE 1: a source that disputes the page → flag. RULE 1b: a reviewed or locked page is
        authoritative; conflicting content is flagged, never applied. RULE 2: new material without
        dispute → update. List every conflict verbatim."""

        page: PageState = dspy.InputField()
        concept: ConceptDraft = dspy.InputField()
        decision: IngestDecision = dspy.OutputField()

    class KnowledgeDiff(dspy.Signature):
        """What the batch changes for this page: reinforced, challenged (only what conflicts),
        new (only what the page lacks), gaps the sources leave open."""

        page: PageState = dspy.InputField()
        concept: ConceptDraft = dspy.InputField()
        diff: Diff = dspy.OutputField()

    class AnswerQuery(dspy.Signature):
        """Answer from the given pages only, cite pages as [[slug]], name gaps when the pages are thin."""

        question: str = dspy.InputField()
        pages: list[PageState] = dspy.InputField()
        answer: Answer = dspy.OutputField()

    class BatchCompile(dspy.Module):
        def __init__(self):
            super().__init__()
            self.triage = dspy.ChainOfThought(TriageSource)
            self.extract = dspy.ChainOfThought(ExtractClaims)
            self.merge = dspy.ChainOfThought(MergeConcepts)
            self.decide = dspy.ChainOfThought(DecideIngest)
            self.diff = dspy.ChainOfThought(KnowledgeDiff)
            self.answer = dspy.ChainOfThought(AnswerQuery)

        def forward(self, sources: dict[str, str], pages: dict[str, PageState], known_entities: list[str]):
            extractions = [Extraction(source=name, triage=self.triage(source_name=name, text=text).triage,
                                      claims=self.extract(source_name=name, numbered_text=number_lines(text),
                                                          known_entities=known_entities).claims)
                           for name, text in sources.items()]                       # phase 1: all sources
            concepts = self.merge(extractions=extractions, known_entities=known_entities).concepts  # phase 2
            decisions, diffs = [], {}
            for concept in concepts:                                                  # phase 3: per page
                page = pages.get(concept.slug)
                if page is None:                                                      # create is decided in code
                    decisions.append(IngestDecision(slug=concept.slug, action="create", rationale="no page yet"))
                    continue
                decisions.append(self.decide(page=page, concept=concept).decision)
                diffs[concept.slug] = self.diff(page=page, concept=concept).diff
            return dspy.Prediction(compiled=Compiled(extractions=extractions, concepts=concepts,
                                                     decisions=decisions, diffs=diffs))

    return BatchCompile


# --- fixture -------------------------------------------------------------------

SOURCES = {"a.md": "Juna lebt in KW2.\nDer Schleier hält bis Kapitel 13.",
           "b.md": "Juna lebt in KW3.\nDer Schleier ist ein Ritual."}
PAGES = {"juna": PageState(slug="juna", status="reviewed", body="Juna lebt in KW2.")}
KNOWN = ["Juna", "Schleier"]


def fixture() -> Compiled:
    cite = lambda f, n, q: Citation(file=f, start=n, end=n, quote=q)  # noqa: E731
    ext = [Extraction(source="a.md", triage=Triage(tier="primary", category="charaktere", language="de"),
                      claims=[Claim(text="Juna lebt in KW2.", citation=cite("a.md", 1, "Juna lebt in KW2."), entities=["Juna"]),
                              Claim(text="Der Schleier hält bis Kapitel 13.", citation=cite("a.md", 2, "hält bis Kapitel 13"), kind="rule")]),
           Extraction(source="b.md", triage=Triage(tier="secondary", category="charaktere", language="de"),
                      claims=[Claim(text="Juna lebt in KW3.", citation=cite("b.md", 1, "Juna lebt in KW3."), entities=["Juna"])])]
    juna = ConceptDraft(slug="juna", definition="[[Juna]] lebt laut a.md in KW2 und laut b.md in KW3.",
                        sources=["a.md", "b.md"], citations=[cite("a.md", 1, "Juna lebt in KW2."), cite("b.md", 1, "Juna lebt in KW3.")],
                        disagreements=[Disagreement(topic="Wohnort", sources=["a.md", "b.md"], positions=["KW2", "KW3"])],
                        status="contradicted")
    schleier = ConceptDraft(slug="schleier", definition="Der [[Schleier]] hält bis Kapitel 13 und ist ein Ritual.",
                            sources=["a.md", "b.md"], citations=[cite("a.md", 2, "hält bis Kapitel 13"), cite("b.md", 2, "ist ein Ritual")],
                            status="tentative")
    decisions = [IngestDecision(slug="juna", action="flag", rationale="b.md disputes the reviewed page", conflicts=["Wohnort: KW2 vs KW3"]),
                 IngestDecision(slug="schleier", action="create", rationale="no page yet")]
    diffs = {"juna": Diff(challenged=["Wohnort KW2"], gaps=["Zeitpunkt eines Umzugs"])}
    return Compiled(extractions=ext, concepts=[juna, schleier], decisions=decisions, diffs=diffs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Skip all LM calls")
    ap.add_argument("--model", default=os.getenv("DSPY_MODEL", "openai/gpt-4o"))
    args = ap.parse_args()

    import dspy

    BatchCompile = build()
    program = BatchCompile()
    gold = dspy.Example(sources=SOURCES, pages=PAGES, known_entities=KNOWN, language="de").with_inputs("sources", "pages", "known_entities")

    if args.dry_run:
        good = compile_metric(gold, dspy.Prediction(compiled=fixture()))
        assert good.score >= 0.95, good
        bad = fixture()
        bad.decisions[0] = IngestDecision(slug="juna", action="update", rationale="apply b.md", conflicts=["Wohnort: KW2 vs KW3"])
        bad.concepts[0].citations[0].quote = "Juna wohnt in KW2."
        bad.concepts[0].disagreements[0].sources = ["b.md"]
        bad.diffs["juna"].new = ["Juna lebt in KW2."]
        worse = compile_metric(gold, dspy.Prediction(compiled=bad))
        assert worse.score < good.score
        for needle in ("illegal decision: update on juna", "citation does not resolve", "two distinct sources", "already on the page"):
            assert needle in worse.feedback, (needle, worse.feedback)
        print(f"OK: BatchCompile constructed with predictors {[n for n, _ in program.named_predictors()]}")
        print(f"    fixture metric: {good.score} · broken fixture: {worse.score}")
        print(f"    broken-fixture feedback: {worse.feedback[:160]}…")
        return 0

    dspy.configure(lm=dspy.LM(args.model))
    result = program(sources=SOURCES, pages=PAGES, known_entities=KNOWN)
    print(compile_metric(gold, result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
