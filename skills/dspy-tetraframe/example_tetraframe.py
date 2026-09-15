"""dspy-tetraframe — runnable smoke test.

Compact port of Hmbown/tetraframe-dspy (MIT): a decision seed is distilled
to a predicate, four corners are generated in isolation (P, not-P, both,
neither), their relations are mapped, a non-averaging frame P* is produced
with dspy.BestOfN, and a deterministic verification suite scores the run.
The dry run exercises the guards and the verification on a hand-built run
without any LM call.

Usage:
    uv run python example_tetraframe.py --dry-run
    OPENAI_API_KEY=... uv run python example_tetraframe.py --seed "..."
"""

from __future__ import annotations

import argparse
import os
import re
from itertools import combinations
from typing import Literal

from pydantic import BaseModel, Field

Mode = Literal["P", "not-P", "both", "neither"]
MODES = ("P", "not-P", "both", "neither")
BOTH_BASES = ("temporal_split", "scale_split", "role_split", "ontology_split", "context_split",
              "layered_causality", "admissible_paradox")
NEITHER_FAILURES = ("category_error", "false_binary", "overloaded_predicate", "missing_latent_variable",
                    "bad_ontology", "ill_posed_objective", "frame_collapse_under_scrutiny")
INCOMPATIBLE = (("P", "not-P"), ("P", "neither"), ("not-P", "neither"))
COMPROMISE = ("middle ground", "balanced approach", "split the difference", "on the one hand", "on the other hand")
MUSH = {"balanced", "nuanced", "important", "helpful", "complex", "thoughtful", "consider", "various", "multiple"}
CONTRACTS = {
    "P": "Strongest clean affirmation of the predicate. No compromise, no mention of other corners.",
    "not-P": "Strongest clean rejection, inversion or dismantling of the predicate. No mere surface negation.",
    "both": "Valid only when P and not-P co-hold under a typed split or an admissible paradox.",
    "neither": "Valid only when the predicate is misframed and replaced by a better predicate or frame.",
}
THRESHOLDS = {"branch_independence": 0.90, "rigor_of_both": 0.78, "rigor_of_neither": 0.78,
              "contradiction_honesty": 0.75, "transformation_quality": 0.82, "fake_novelty_risk": 0.70, "slop_risk": 0.70}


class Distilled(BaseModel):
    normalized_seed: str
    stakes: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    hidden_assumptions: list[str] = Field(default_factory=list)
    candidate_predicates: list[str] = Field(default_factory=list)
    frame_risk_score: float = Field(ge=0, le=1, default=0.5)
    evaluation_criteria: list[str] = Field(default_factory=list)


class Selection(BaseModel):
    primary_predicate: str
    rejected: list[str] = Field(default_factory=list)
    rationale: str = ""


class CornerView(BaseModel):
    """The only input a corner generator sees."""

    normalized_seed: str
    stakes: list[str]
    constraints: list[str]
    hidden_assumptions: list[str]
    primary_predicate: str
    evaluation_criteria: list[str]
    corner_contract: str


class Corner(BaseModel):
    mode: Mode
    core_claim: str
    strongest_case: str
    scope_conditions: list[str] = Field(default_factory=list)
    evidence_needs: list[str] = Field(default_factory=list)
    unique_signal: str = ""
    basis_label: str = ""
    basis_explanation: str = ""
    replacement_predicate: str = ""
    patched_claim: str = ""
    minimal_falsifiers: list[str] = Field(default_factory=list)
    confidence_score: float = Field(ge=0, le=1, default=0.5)


class Cartography(BaseModel):
    contradiction_map: list[str] = Field(default_factory=list)
    complementarity_map: list[str] = Field(default_factory=list)
    discriminators: list[str] = Field(default_factory=list)
    invariants: list[str] = Field(default_factory=list)
    arbiter_notes: str = ""


class Frame(BaseModel):
    transformed_predicate: str
    transformed_frame: str
    survivors_from_p: list[str] = Field(default_factory=list)
    survivors_from_not_p: list[str] = Field(default_factory=list)
    hidden_structure_from_both: list[str] = Field(default_factory=list)
    dissolved_false_frame_from_neither: list[str] = Field(default_factory=list)
    operational_tests: list[str] = Field(default_factory=list)


class Run(BaseModel):
    seed: str
    distilled: Distilled
    selection: Selection
    corners: dict[str, Corner]
    cartography: Cartography
    frame: Frame


# --- guards and heuristics (deterministic) ------------------------------------

def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-zA-Z_]+", (text or "").lower()) if len(t) > 2}


def residual(text: str, seed: str) -> str:
    return " ".join(sorted(_tokens(text) - _tokens(seed)))


def similarity(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta and not tb:
        return 1.0
    return len(ta & tb) / len(ta | tb) if ta and tb else 0.0


def assert_isolation(view: CornerView) -> None:
    leaked = set(view.model_dump()) - set(CornerView.model_fields)
    if leaked:
        raise ValueError(f"corner view leaked fields: {sorted(leaked)}")


def near_duplicates(corners: dict[str, Corner], seed: str, threshold: float = 0.78) -> list[tuple[str, str]]:
    return [(a, b) for a, b in combinations(corners, 2)
            if similarity(residual(corners[a].core_claim, seed), residual(corners[b].core_claim, seed)) >= threshold]


def _mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def falsifier_quality(c: Corner) -> float:
    if not c.minimal_falsifiers:
        return 0.0
    return _mean(1.0 if len(f.split()) >= 5 else 0.5 for f in c.minimal_falsifiers)


def both_rigor(c: Corner) -> float:
    basis = 1.0 if c.basis_label in BOTH_BASES else 0.0
    penalty = 0.4 if any(m in c.strongest_case.lower() for m in COMPROMISE) else 0.0
    co_hold = 1.0 if any(w in c.basis_explanation.lower() for w in ("both", "co-hold", "simult", "split")) else 0.5
    return max(0.0, _mean([basis, co_hold, 1.0 if c.scope_conditions else 0.5, falsifier_quality(c)]) - penalty)


def neither_rigor(c: Corner) -> float:
    mode = 1.0 if c.basis_label in NEITHER_FAILURES else 0.0
    replacement = 1.0 if c.replacement_predicate.strip() else 0.0
    diagnosis = 1.0 if len(c.basis_explanation.split()) >= 8 else 0.5
    evasion = 0.3 if "it depends" in c.strongest_case.lower() else 0.0
    return max(0.0, _mean([mode, replacement, diagnosis, falsifier_quality(c)]) - evasion)


def branch_independence(run: Run) -> float:
    seed = run.distilled.normalized_seed
    pair = [max(0.0, similarity(residual(run.corners[a].core_claim, seed), residual(run.corners[b].core_claim, seed)) - 0.35)
            for a, b in INCOMPATIBLE]
    return max(0.0, 1.0 - min(1.0, 0.6 * _mean(pair)))


def transformation_quality(frame: Frame, corners: dict[str, Corner]) -> float:
    text = (frame.transformed_frame + " " + frame.transformed_predicate).lower()
    required = [frame.survivors_from_p, frame.survivors_from_not_p, frame.hidden_structure_from_both,
                frame.dissolved_false_frame_from_neither, frame.operational_tests]
    overlap = similarity(frame.transformed_predicate, corners["P"].patched_claim + " " + corners["not-P"].patched_claim)
    score = min(1.0, _mean(1.0 if r else 0.0 for r in required) + 0.3 * (1.0 - overlap))  # cap BEFORE the penalty
    return max(0.0, score - (0.4 if any(m in text for m in COMPROMISE) else 0.0))


def contradiction_honesty(c: Cartography) -> float:
    if not c.contradiction_map:
        return 0.35 if c.complementarity_map else 0.25
    return min(1.0, 0.4 + 0.1 * len(c.contradiction_map) + 0.05 * len(c.discriminators))


def fake_novelty(run: Run) -> float:
    source = " ".join([run.selection.primary_predicate, run.distilled.normalized_seed,
                       *(c.patched_claim + " " + c.strongest_case + " " + c.replacement_predicate for c in run.corners.values()),
                       *run.cartography.invariants, *run.frame.survivors_from_p, *run.frame.survivors_from_not_p,
                       *run.frame.hidden_structure_from_both, *run.frame.dissolved_false_frame_from_neither])
    src = _tokens(source)
    unsupported = [t for t in _tokens(run.frame.transformed_predicate) if len(t) > 4
                   and t not in src and not any(len(s) > 3 and s in t for s in src)]
    return max(0.0, 1.0 - min(0.6, 0.08 * len(unsupported)))


def slop(run: Run) -> float:
    tokens = [t for text in (run.frame.transformed_frame, *(c.patched_claim for c in run.corners.values()))
              for t in re.findall(r"[a-zA-Z]+", text.lower())]
    ratio = sum(t in MUSH for t in tokens) / len(tokens) if tokens else 0.0
    return max(0.0, 1.0 - 3.0 * ratio)


def verify(run: Run) -> dict[str, float]:
    return {
        "branch_independence": branch_independence(run),
        "rigor_of_both": both_rigor(run.corners["both"]),
        "rigor_of_neither": neither_rigor(run.corners["neither"]),
        "contradiction_honesty": contradiction_honesty(run.cartography),
        "transformation_quality": transformation_quality(run.frame, run.corners),
        "fake_novelty_risk": fake_novelty(run),
        "slop_risk": slop(run),
    }


def transform_reward(args: dict, pred) -> float:
    f: Frame = pred.frame
    penalty = 0.4 if any(m in (f.transformed_predicate + f.transformed_frame).lower() for m in COMPROMISE) else 0.0
    parts = [f.survivors_from_p, f.survivors_from_not_p, f.hidden_structure_from_both, f.dissolved_false_frame_from_neither]
    return max(0.0, _mean(1.0 if p else 0.0 for p in parts) - penalty)


def tetraframe_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    import dspy

    run: Run = pred.run
    scores = verify(run)
    deficits = [f"{k} {v:.2f} < {THRESHOLDS[k]:.2f}" for k, v in scores.items() if v < THRESHOLDS[k]]
    banned = [b for b in getattr(gold, "banned_transformed_phrases", []) or [] if b in run.frame.transformed_frame.lower()]
    if banned:
        deficits.append(f"P* uses banned phrases {banned}")
    score = _mean(scores.values()) * (0.5 if banned else 1.0)
    return dspy.Prediction(score=round(score, 3), feedback="; ".join(deficits) or "independent corners, rigorous both/neither, transformed P*")


# --- program -------------------------------------------------------------------

def build():
    import dspy

    class DistillSeed(dspy.Signature):
        """Distill a decision seed into stakes, constraints, hidden assumptions, candidate predicates
        and evaluation criteria; high frame risk when the seed bundles objectives."""

        seed: str = dspy.InputField()
        distilled: Distilled = dspy.OutputField()

    class SelectPredicate(dspy.Signature):
        """Choose one operational, falsifiable primary predicate; reject malformed candidates with reasons."""

        distilled: Distilled = dspy.InputField()
        selection: Selection = dspy.OutputField()

    class GenerateCorner(dspy.Signature):
        """Generate and harden one corner from the view alone; never mention other corners."""

        view: CornerView = dspy.InputField()
        corner: Corner = dspy.OutputField()

    class CornerP(GenerateCorner):
        """Strongest clean affirmation of the predicate; basis_label must be 'affirmation'."""

    class CornerNotP(GenerateCorner):
        """Strongest clean rejection or dismantling of the predicate's substance; basis_label 'rejection'."""

    class CornerBoth(GenerateCorner):
        """Both, not compromise: pick one basis (temporal_split, scale_split, role_split, ontology_split,
        context_split, layered_causality, admissible_paradox) as basis_label and show why both hold under it."""

    class CornerNeither(GenerateCorner):
        """Neither, not evasion: name the failure mode (category_error, false_binary, overloaded_predicate,
        missing_latent_variable, bad_ontology, ill_posed_objective, frame_collapse_under_scrutiny) as
        basis_label and propose a replacement_predicate."""

    class MapCorners(dspy.Signature):
        """Map contradictions, complementarities, evidence discriminators and invariants across the corners
        without adding premises."""

        corners: list[Corner] = dspy.InputField()
        cartography: Cartography = dspy.OutputField()

    class Transform(dspy.Signature):
        """Produce P*: not an average; keep survivors from P and not-P, the hidden structure from both,
        dissolve the false frame from neither, give operational tests."""

        primary_predicate: str = dspy.InputField()
        corners: list[Corner] = dspy.InputField()
        cartography: Cartography = dspy.InputField()
        frame: Frame = dspy.OutputField()

    class TetraFrame(dspy.Module):
        def __init__(self):
            super().__init__()
            self.distill = dspy.ChainOfThought(DistillSeed)
            self.select = dspy.ChainOfThought(SelectPredicate)
            self.generators = {"P": dspy.ChainOfThought(CornerP), "not-P": dspy.ChainOfThought(CornerNotP),
                               "both": dspy.ChainOfThought(CornerBoth), "neither": dspy.ChainOfThought(CornerNeither)}
            self.corner_p, self.corner_not_p = self.generators["P"], self.generators["not-P"]
            self.corner_both, self.corner_neither = self.generators["both"], self.generators["neither"]
            self.map = dspy.ChainOfThought(MapCorners)
            self.transform = dspy.BestOfN(module=dspy.ChainOfThought(Transform), N=3,
                                          reward_fn=transform_reward, threshold=0.84)

        def forward(self, seed: str):
            d = self.distill(seed=seed).distilled
            s = self.select(distilled=d).selection
            corners = {}
            for i, mode in enumerate(MODES):
                view = CornerView(normalized_seed=d.normalized_seed, stakes=d.stakes, constraints=d.constraints,
                                  hidden_assumptions=d.hidden_assumptions, primary_predicate=s.primary_predicate,
                                  evaluation_criteria=d.evaluation_criteria, corner_contract=CONTRACTS[mode])
                assert_isolation(view)
                lm = dspy.settings.lm
                with dspy.context(lm=lm.copy(rollout_id=i, temperature=0.9 if mode in ("both", "neither") else 0.7)):
                    corner = self.generators[mode](view=view).corner
                corner.mode = mode
                corners[mode] = corner
            cartography = self.map(corners=list(corners.values())).cartography
            frame = self.transform(primary_predicate=s.primary_predicate, corners=list(corners.values()),
                                   cartography=cartography).frame
            return dspy.Prediction(run=Run(seed=seed, distilled=d, selection=s, corners=corners,
                                           cartography=cartography, frame=frame))

    return TetraFrame


SEED = "We keep framing the wiki change as 'keep the old page' or 'replace it', but the real choice may be a versioned page with a supersedes link."


def fixture_run() -> Run:
    def corner(mode, claim, **kw):
        base = dict(mode=mode, core_claim=claim, strongest_case=claim, patched_claim=claim, unique_signal=f"signal {mode}",
                    scope_conditions=["wiki concept pages"], minimal_falsifiers=["a reader who needs the old page and cannot find it"],
                    evidence_needs=["log: how often the old page is opened after replacement"], confidence_score=0.8)
        base.update(kw)
        return Corner(**base)
    corners = {
        "P": corner("P", "Keep the old page; history is evidence and links must not break.", basis_label="affirmation"),
        "not-P": corner("not-P", "Replace it; two pages for one concept guarantee drift.", basis_label="rejection"),
        "both": corner("both", "Keep for readers of history, replace for current work.", basis_label="role_split",
                       basis_explanation="Both hold under a role split: archival readers and current authors co-hold different needs simultaneously."),
        "neither": corner("neither", "The choice conflates identity of a concept with identity of a file.", basis_label="false_binary",
                          replacement_predicate="A concept has one current page and a supersedes chain.",
                          basis_explanation="The predicate is a false binary because keep-or-replace hides the third structure: versioned pages with a supersedes link."),
    }
    cartography = Cartography(
        contradiction_map=["P vs not-P on file identity", "P vs neither on whether history needs the old file",
                           "not-P vs both on whether two readerships need two pages"],
        complementarity_map=["both and neither agree on two audiences"],
        discriminators=["do inbound links target the file or the concept?", "is the old page opened after the replacement lands?"],
        invariants=["one current page per concept"])
    frame = Frame(transformed_predicate="One current page per concept with a supersedes chain; old pages stay reachable as history.",
                  transformed_frame="Versioned pages instead of keep-or-replace.", survivors_from_p=["history stays reachable"],
                  survivors_from_not_p=["one current page"], hidden_structure_from_both=["two audiences"],
                  dissolved_false_frame_from_neither=["file identity as concept identity"], operational_tests=["lint: every superseded page links to its successor"])
    return Run(seed=SEED, distilled=Distilled(normalized_seed=SEED), selection=Selection(primary_predicate="Replacing a wiki page is better than keeping it"),
               corners=corners, cartography=cartography, frame=frame)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Skip all LM calls")
    ap.add_argument("--model", default=os.getenv("DSPY_MODEL", "openai/gpt-4o"))
    ap.add_argument("--seed", default=SEED)
    args = ap.parse_args()

    import dspy

    TetraFrame = build()
    program = TetraFrame()

    if args.dry_run:
        run = fixture_run()
        scores = verify(run)
        assert all(v >= THRESHOLDS[k] for k, v in scores.items()), scores
        assert near_duplicates({"P": run.corners["P"], "x": run.corners["P"]}, run.seed) == [("P", "x")]
        run.frame.transformed_frame = "On the one hand keep, on the other hand replace: a balanced approach."
        assert transformation_quality(run.frame, run.corners) < THRESHOLDS["transformation_quality"]
        metric = tetraframe_metric(dspy.Example(seed=SEED, banned_transformed_phrases=["balanced approach"]), dspy.Prediction(run=run))
        assert metric.score < 0.6 and "banned" in metric.feedback
        names = [n for n, _ in program.named_predictors()]
        print("OK: TetraFrame constructed with predictors", names)
        print("    fixture verification:", {k: round(v, 2) for k, v in scores.items()})
        print(f"    compromise P* flagged; metric feedback: {metric.feedback}")
        return 0

    dspy.configure(lm=dspy.LM(args.model), track_usage=True)
    run = program(seed=args.seed).run
    for mode, c in run.corners.items():
        print(f"[{mode}] {c.patched_claim or c.core_claim}")
    print("P*:", run.frame.transformed_predicate)
    print("verification:", {k: round(v, 2) for k, v in verify(run).items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
