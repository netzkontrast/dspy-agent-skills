"""dspy-autodialectics — runnable smoke test.

Ships the deterministic core of the Autodialectics harness as importable
functions (contract compiler, per-criterion checks, 12-dimension slop score,
rubric run score, gate, promotion rule, canary check) and builds the DSPy
`Dialectic` module around a toy executor. The dry run scores a sloppy output
and an honest output for the same contract without any LM call.

Usage:
    uv run python example_autodialectics.py --dry-run
    OPENAI_API_KEY=... uv run python example_autodialectics.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from typing import Literal

from pydantic import BaseModel, Field

Domain = Literal["code", "research", "writing", "experiment", "analysis", "generic"]

COMMON_SHORTCUTS = [
    "Do not claim completion without verification evidence.",
    "Do not invent citations, logs, tests, files, or benchmark results.",
    "Do not silently rewrite objectives or constraints during the run.",
    "Do not suppress uncertainty when evidence is weak or conflicting.",
    "Do not treat scratchpad notes as canonical requirements.",
]
DOMAIN_CRITERIA: dict[str, list[str]] = {
    "code": ["All tests pass on the reference interpreter/platform.", "No regressions in existing functionality."],
    "research": ["Every factual claim cites a verifiable source.", "Contradictory evidence is acknowledged and discussed."],
    "writing": ["Tone and style are consistent with the brief.", "No factual errors introduced during revision."],
    "experiment": ["Experimental procedure is fully specified and reproducible.", "Results include confidence intervals or significance tests."],
    "analysis": ["Analysis considers multiple interpretations of the data.", "Conclusions follow from the evidence presented."],
    "generic": ["Deliverable satisfies all stated objectives.", "Constraints are respected throughout."],
}
DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "code": ["code", "bug", "function", "test", "implement", "refactor", "repo"],
    "research": ["research", "literature", "sources", "cite", "synthesize", "papers"],
    "writing": ["draft", "essay", "argument", "revise", "prose", "tone"],
    "experiment": ["experiment", "hypothesis", "protocol", "ablation", "measure"],
    "analysis": ["analyze", "analysis", "dataset", "trend", "interpret"],
}
SLOP_WEIGHTS = {
    "verbosity_without_gain": 0.12, "repetition_without_progress": 0.10, "unsupported_claims": 0.15,
    "requirement_drift": 0.10, "fake_completion": 0.15, "self_verification_bias": 0.08,
    "benchmark_gaming": 0.05, "shallow_novelty": 0.05, "context_contamination": 0.05,
    "refusal_to_surface_uncertainty": 0.05, "tool_abuse": 0.05, "synthesis_ignores_objections": 0.05,
}
BASE_RUBRIC = {
    "task_success": 0.30, "groundedness": 0.20, "objection_coverage": 0.10, "unsupported_assertion_rate": 0.05,
    "redundancy_rate": 0.05, "novelty_usefulness": 0.10, "requirement_fidelity": 0.10, "verification_quality": 0.10,
}
RUBRIC_OVERRIDES = {
    "code": {"task_success": 0.35, "verification_quality": 0.15, "groundedness": 0.15, "novelty_usefulness": 0.05},
    "research": {"groundedness": 0.30, "task_success": 0.20, "objection_coverage": 0.15},
    "experiment": {"verification_quality": 0.20, "groundedness": 0.20, "task_success": 0.25},
}
ACCEPT_MIN_SCORE, ACCEPT_MAX_SLOP, REJECT_SLOP, REJECT_CONFIDENCE = 0.6, 0.4, 0.7, 0.3
FEEDBACK_THRESHOLD, SERIOUS_SEVERITY, MIN_VERBOSE_WORDS = 0.3, 0.5, 200
NEGATION = ("did not", "didn't", "does not", "cannot", "unable", "without", "missing", "fail", "not ", "no ")


class Contract(BaseModel):
    source_hash: str
    title: str
    domain: Domain
    objectives: list[str]
    constraints: list[str] = Field(default_factory=list)
    acceptance_criteria: list[str]
    forbidden_shortcuts: list[str]
    rubric: dict[str, float]


class Objection(BaseModel):
    claim: str
    objection: str
    severity: float = Field(ge=0.0, le=1.0)


class Disposition(BaseModel):
    objection_index: int
    accepted: bool
    how: str


class Output(BaseModel):
    """What the executor returns; `summary` and `uncertainties` feed the slop score."""
    text: str
    summary: str = ""
    uncertainties: list[str] = Field(default_factory=list)
    test_results: list[str] = Field(default_factory=list)
    patches: list[str] = Field(default_factory=list)
    tool_log: list[str] = Field(default_factory=list)


# ── deterministic core ─────────────────────────────────────────────────

def keywords(text: str) -> set[str]:
    return {w.lower().strip(".,;:()") for w in text.split() if len(w) > 3}


def _clamp(value: float) -> float:
    return max(0.0, min(float(value), 1.0))


def infer_domain(task: dict) -> str:
    if task.get("domain"):
        return task["domain"]
    text = keywords(f"{task.get('title', '')} {task.get('description', '')}")
    hits = {d: len(text & set(k)) for d, k in DOMAIN_KEYWORDS.items()}
    best = max(hits, key=hits.get)
    return best if hits[best] > 0 else "generic"


def compile_contract(task: dict) -> Contract:
    """Immutable contract: user items first, domain defaults appended, sha256 of the task."""
    domain = infer_domain(task)
    rubric = {**BASE_RUBRIC, **RUBRIC_OVERRIDES.get(domain, {})}
    dedupe = lambda items: list(dict.fromkeys(i for i in items if i))
    return Contract(
        source_hash=hashlib.sha256(json.dumps(task, sort_keys=True).encode()).hexdigest(),
        title=task["title"], domain=domain,
        objectives=dedupe(task.get("objectives") or [task.get("description", task["title"])]),
        constraints=dedupe(task.get("constraints", [])),
        acceptance_criteria=dedupe(task.get("acceptance_criteria", []) + DOMAIN_CRITERIA[domain]),
        forbidden_shortcuts=dedupe(task.get("forbidden_shortcuts", []) + COMMON_SHORTCUTS),
        rubric=rubric,
    )


def _negated_support(text: str, terms: set[str]) -> bool:
    """True when every sentence that mentions the criterion also carries a negation marker."""
    windows = [w for w in re.split(r"(?<=[.!?])\s+|\n+", text) if len(terms & keywords(w)) / max(len(terms), 1) >= 0.5]
    return bool(windows) and all(any(n in f" {w.lower()} " for n in NEGATION) for w in windows)


def criterion_checks(contract: Contract, text: str) -> list[tuple[str, bool, str]]:
    """Deterministic backstop: keyword overlap per criterion; a criterion mentioned only in negated sentences fails."""
    checks = []
    for criterion in contract.acceptance_criteria:
        terms = keywords(criterion)
        present = terms & keywords(text)
        negated = _negated_support(text, terms)
        passed = len(present) >= max(1, len(terms) // 2) and not negated
        checks.append((criterion, passed, f"{len(present)}/{len(terms)} keywords" + (", negated" if negated else "")))
    return checks


def _repetition(text: str) -> float:
    sentences = [s.strip().lower() for s in re.split(r"[.!?]+", text) if s.strip()]
    tokens = text.lower().split()
    sent = 1 - len(set(sentences)) / len(sentences) if sentences else 0.0
    trigrams = [tuple(tokens[i:i + 3]) for i in range(len(tokens) - 2)]
    tri = 1 - len(set(trigrams)) / len(trigrams) if trigrams else 0.0
    return (sent + tri) / 2


def _count(patterns: list[str], text: str) -> int:
    return sum(len(re.findall(p, text, re.IGNORECASE)) for p in patterns)


def slop_dimensions(contract: Contract, out: Output, objections: list[Objection],
                    excerpts: list[str]) -> dict[str, float]:
    text, n_words = out.text, len(out.text.split())
    claims = _count([r"(?:is|are) (?:the )?(?:best|only|proven)", r"studies show|it is known",
                     r"clearly|obviously|certainly", r"it follows that|this means that"], text)
    supported = min(sum(1 for e in excerpts if e[:100] in text), claims)
    completion = _count([r"\b(?:done|complete|finished|implemented|resolved)\b"], text)
    fake = [bool(completion) and not (out.test_results or out.patches),
            bool(contract.constraints) and not out.uncertainties]
    self_verify = _count([r"i (?:have )?verif", r"tests? (?:pass|passed)", r"works? as (?:expected|intended)"], text)
    independent = sum(1 for e in out.tool_log if "test" in e.lower() or "verify" in e.lower())
    hedges = _count([r"\b(?:uncertain|unclear|ambiguous|may|might|could|possibly)\b"], text)
    certain = _count([r"\b(?:definitely|certainly|absolutely|guaranteed|unquestionably)\b"], text)
    novelty = _count([r"\b(?:novel|innovative|breakthrough|unique)\b"], text)
    novelty_in_summary = _count([r"\b(?:novel|innovative|breakthrough|unique)\b"], out.summary)
    obj_kw, cons_kw, text_kw = keywords(" ".join(contract.objectives)), keywords(" ".join(contract.constraints)), keywords(text)
    drift_parts = [(len(obj_kw & text_kw) / len(obj_kw), 0.6)] if obj_kw else []
    drift_parts += [(len(cons_kw & text_kw) / len(cons_kw), 0.4)] if cons_kw else []
    serious = [o for o in objections if o.severity > SERIOUS_SEVERITY]
    ignored = sum(1 for o in serious if not (keywords(o.objection) & text_kw))
    jaccard = [len(keywords(e) & text_kw) / max(len(keywords(e) | text_kw), 1) for e in excerpts]
    refusal = 0.0 if not certain and not out.uncertainties else 1 - hedges / max(certain + hedges, 1)
    return {k: _clamp(v) for k, v in {
        "verbosity_without_gain": 0.0 if n_words < MIN_VERBOSE_WORDS else min(n_words / 5000, 1) * (1 - len(out.summary.split()) / max(n_words, 1)),
        "repetition_without_progress": _repetition(text),
        "unsupported_claims": 0.0 if not claims else (claims - supported) / claims * (1 - len(out.uncertainties) / claims),
        "requirement_drift": 1 - sum(v * w for v, w in drift_parts) / sum(w for _, w in drift_parts) if drift_parts else 0.0,
        "fake_completion": sum(fake) / 2,
        "self_verification_bias": 0.0 if not self_verify else max(self_verify - independent, 0) / self_verify,
        "benchmark_gaming": 0.3 * _count([r"hard-?coded", r"overfit", r"optimized (?:specifically|just) to pass"], text),
        "shallow_novelty": 0.0 if not novelty else max(novelty - novelty_in_summary, 0) / novelty,
        "context_contamination": (sum(jaccard) / len(jaccard) - 0.3) / 0.5 if jaccard else 0.0,
        "refusal_to_surface_uncertainty": refusal * (0.3 if out.uncertainties else 1.0),
        "tool_abuse": sum("redundant" in e.lower() or "duplicate" in e.lower() for e in out.tool_log) / max(len(out.tool_log), 1),
        "synthesis_ignores_objections": ignored / len(serious) if serious else 0.0,
    }.items()}


def slop_score(contract: Contract, out: Output, objections: list[Objection] | None = None,
               excerpts: list[str] | None = None):
    """GEPA-compatible: score = 1 - composite; feedback names the worst dimensions."""
    import dspy

    dims = slop_dimensions(contract, out, objections or [], excerpts or [])
    composite = sum(dims[k] * w for k, w in SLOP_WEIGHTS.items()) / sum(SLOP_WEIGHTS.values())
    worst = sorted(((v, k) for k, v in dims.items() if v > FEEDBACK_THRESHOLD), reverse=True)
    feedback = "; ".join(f"{k} {v:.2f}" for v, k in worst) or "No slop dimension above threshold."
    return dspy.Prediction(score=_clamp(1 - composite), feedback=feedback, composite=composite, dims=dims)


def run_score(contract: Contract, checks: list[tuple[str, bool, str]], dims: dict[str, float],
              objections: list[Objection], dispositions: list[Disposition]) -> float:
    """Rubric-weighted run score (the gate's `score`)."""
    passed = sum(1 for _, ok, _ in checks if ok) / max(len(checks), 1)
    coverage = len({d.objection_index for d in dispositions}) / len(objections) if objections else 1.0
    components = {
        "task_success": passed, "groundedness": 1 - dims["unsupported_claims"], "objection_coverage": coverage,
        "unsupported_assertion_rate": 1 - dims["unsupported_claims"], "redundancy_rate": 1 - dims["repetition_without_progress"],
        "novelty_usefulness": 1 - (dims["shallow_novelty"] + dims["benchmark_gaming"]) / 2,
        "requirement_fidelity": 1 - dims["requirement_drift"], "verification_quality": passed,
    }
    return _clamp(sum(components[k] * w for k, w in contract.rubric.items()) / sum(contract.rubric.values()))


def gate(verdict_pass: bool, confidence: float, score: float, slop: float) -> str:
    if not verdict_pass and confidence < REJECT_CONFIDENCE:
        return "reject"
    if slop > REJECT_SLOP:
        return "reject"
    if verdict_pass and score >= ACCEPT_MIN_SCORE and slop < ACCEPT_MAX_SLOP:
        return "accept"
    return "revise"


def canary_passes(text: str, must_include: list[str], must_not_include: list[str], slop: float,
                  groundedness: float, max_slop: float = 0.6, min_groundedness: float = 0.2) -> bool:
    lowered = text.lower()
    return (all(t in lowered for t in must_include) and not any(t in lowered for t in must_not_include)
            and slop <= max_slop and groundedness >= min_groundedness)


def promote(champion: tuple[float, float], challenger: tuple[float, float], canaries_passed: bool) -> tuple[bool, str]:
    """(score, slop) pairs; all three conditions are required."""
    if not canaries_passed:
        return False, "canary failed"
    if challenger[1] > champion[1]:
        return False, f"challenger slop {challenger[1]:.2f} > champion {champion[1]:.2f}"
    if challenger[0] <= champion[0]:
        return False, f"challenger score {challenger[0]:.2f} <= champion {champion[0]:.2f}"
    return True, "score up, slop not up, canaries pass"


# ── DSPy module ────────────────────────────────────────────────────────

def build():
    import dspy

    class Thesis(dspy.Signature):
        """Plan the task as numbered, concrete steps that satisfy every objective under every
        constraint. Include verification steps; never plan around a forbidden shortcut."""
        contract: Contract = dspy.InputField()
        evidence_summary: str = dspy.InputField()
        steps: list[str] = dspy.OutputField()

    class Antithesis(dspy.Signature):
        """Find the real flaws, risks and gaps in the plan. Be adversarial but concrete; do not
        invent problems. Severity 1.0 = the plan fails the contract, 0.3 = nit."""
        contract: Contract = dspy.InputField()
        steps: list[str] = dspy.InputField()
        objections: list[Objection] = dspy.OutputField()

    class Synthesis(dspy.Signature):
        """Revise the plan. For EVERY objection state accepted or rejected and how the revised
        steps reflect it. Unaddressed objections with severity > 0.8 are failures."""
        contract: Contract = dspy.InputField()
        steps: list[str] = dspy.InputField()
        objections: list[Objection] = dspy.InputField()
        dispositions: list[Disposition] = dspy.OutputField()
        revised_steps: list[str] = dspy.OutputField()

    class Execute(dspy.Signature):
        """Carry out the plan. Report what was actually done, a one-paragraph summary, and every
        uncertainty; never claim tests or files you did not produce."""
        contract: Contract = dspy.InputField()
        plan: list[str] = dspy.InputField()
        output: Output = dspy.OutputField()

    class Verify(dspy.Signature):
        """Independent verification: you see only the contract and the output, not the plan.
        Decide pass/fail per acceptance criterion with a one-line reason."""
        contract: Contract = dspy.InputField()
        output: str = dspy.InputField()
        checks: list[tuple[str, bool, str]] = dspy.OutputField()

    class Dialectic(dspy.Module):
        def __init__(self):
            super().__init__()
            self.thesis, self.antithesis = dspy.ChainOfThought(Thesis), dspy.ChainOfThought(Antithesis)
            self.synthesis, self.execute = dspy.ChainOfThought(Synthesis), dspy.ChainOfThought(Execute)
            self.verify = dspy.Predict(Verify)

        def forward(self, contract: Contract, evidence_summary: str = "") -> dspy.Prediction:
            t = self.thesis(contract=contract, evidence_summary=evidence_summary)
            a = self.antithesis(contract=contract, steps=t.steps)
            s = self.synthesis(contract=contract, steps=t.steps, objections=a.objections)
            out = self.execute(contract=contract, plan=s.revised_steps).output
            v = self.verify(contract=contract, output=out.text)
            return dspy.Prediction(plan=s.revised_steps, objections=a.objections,
                                   dispositions=s.dispositions, output=out, checks=v.checks)

    return Dialectic()


def harness_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """End-to-end GEPA metric: verified criteria + slop feedback, one Prediction."""
    import dspy

    slop = slop_score(gold.contract, pred.output, pred.objections)
    score = run_score(gold.contract, pred.checks, slop.dims, pred.objections, pred.dispositions)
    decision = gate(all(ok for _, ok, _ in pred.checks), sum(ok for _, ok, _ in pred.checks) / max(len(pred.checks), 1),
                    score, slop.composite)
    return dspy.Prediction(score=score * (0.5 if decision == "reject" else 1.0),
                           feedback=f"gate={decision}; {slop.feedback}")


# ── demo ───────────────────────────────────────────────────────────────

TASK = {"title": "Fix calculator division by zero",
        "description": "The divide function crashes on zero input. Fix it and keep the tests green.",
        "constraints": ["Do not change the public signature of divide()."]}
SLOPPY = Output(text="Done. I verified the fix and all tests pass; it clearly works as expected and is definitely "
                     "the best solution. The implementation is complete and resolved.",
                summary="Fixed.")
HONEST = Output(text="Changed divide() to raise ZeroDivisionError with a message instead of crashing; the public "
                     "signature is unchanged. All 4 tests pass on the reference interpreter. Existing functionality "
                     "shows zero regressions. It is unclear whether callers expect None instead of an exception.",
                summary="divide() now raises on zero; tests green; caller expectation uncertain.",
                uncertainties=["Callers may expect None rather than an exception."],
                test_results=["4 passed"], tool_log=["run tests: 4 passed"])
OBJECTIONS = [Objection(claim="raise on zero", objection="callers may expect None instead of an exception", severity=0.7)]


def dry_run() -> None:
    contract = compile_contract(TASK)
    print(f"contract: domain={contract.domain} hash={contract.source_hash[:12]} criteria={len(contract.acceptance_criteria)}")
    for label, out in (("sloppy", SLOPPY), ("honest", HONEST)):
        checks = criterion_checks(contract, out.text)
        slop = slop_score(contract, out, OBJECTIONS)
        score = run_score(contract, checks, slop.dims, OBJECTIONS, [Disposition(objection_index=0, accepted=True, how="documented")])
        verdict = all(ok for _, ok, _ in checks)
        decision = gate(verdict, sum(ok for _, ok, _ in checks) / len(checks), score, slop.composite)
        print(f"{label:6s} score={score:.2f} slop={slop.composite:.2f} gate={decision} | {slop.feedback}")
    assert gate(True, 1.0, 0.8, 0.1) == "accept" and gate(False, 0.0, 0.8, 0.1) == "reject"
    assert promote((0.6, 0.3), (0.7, 0.2), True)[0] and not promote((0.6, 0.3), (0.7, 0.4), True)[0]
    assert canary_passes("this is ambiguous and contradictory, so uncertain", ["ambiguous", "uncertain"], ["guaranteed"], 0.2, 0.5)
    print("built Dialectic:", type(build()).__name__)
    print("dry run OK")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Skip all LM calls")
    ap.add_argument("--model", default=os.environ.get("DSPY_MODEL", "openai/gpt-4o-mini"))
    args = ap.parse_args()
    if args.dry_run:
        dry_run()
        return
    import dspy

    dspy.configure(lm=dspy.LM(args.model))
    contract = compile_contract(TASK)
    pred = build()(contract=contract, evidence_summary="calculator.py: divide(a, b) returns a / b")
    slop = slop_score(contract, pred.output, pred.objections)
    score = run_score(contract, pred.checks, slop.dims, pred.objections, pred.dispositions)
    verdict = all(ok for _, ok, _ in pred.checks)
    print(f"gate={gate(verdict, 1.0 if verdict else 0.0, score, slop.composite)} score={score:.2f} slop={slop.composite:.2f}")
    print(slop.feedback)


if __name__ == "__main__":
    main()
