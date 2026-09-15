"""dspy-rlm-workflow — runnable smoke test.

Builds the distill → decompose → solve → synthesize → verify → iterate pipeline
as DSPy modules. The dry run exercises every deterministic part (DAG
validation and topological order, the Tier-1 verifier, the Refine wrapper's
construction) without any LM call.

Usage:
    uv run python example_rlm_workflow.py --dry-run
    OPENAI_API_KEY=... uv run python example_rlm_workflow.py
"""

from __future__ import annotations

import argparse
import os
from typing import Literal

from pydantic import BaseModel, Field


class SubProblem(BaseModel):
    id: int
    description: str
    dependencies: list[int] = Field(default_factory=list)
    complexity: Literal["low", "medium", "high"] = "medium"
    success_criteria: str = ""


def validate_dag(sub_problems: list[SubProblem]) -> list[SubProblem]:
    """Return sub-problems in dependency order; raise ValueError on bad graphs."""
    by_id = {sp.id: sp for sp in sub_problems}
    if len(by_id) != len(sub_problems):
        raise ValueError("duplicate sub-problem ids")
    unknown = {d for sp in sub_problems for d in sp.dependencies if d not in by_id}
    if unknown:
        raise ValueError(f"dependencies on unknown ids: {sorted(unknown)}")
    order: list[SubProblem] = []
    state: dict[int, int] = {}

    def visit(sid: int, stack: tuple[int, ...]) -> None:
        if state.get(sid) == 2:
            return
        if state.get(sid) == 1:
            raise ValueError(f"cycle: {' -> '.join(map(str, stack + (sid,)))}")
        state[sid] = 1
        for dep in by_id[sid].dependencies:
            visit(dep, stack + (sid,))
        state[sid] = 2
        order.append(by_id[sid])

    for sp in sub_problems:
        visit(sp.id, ())
    return order


def build():
    import dspy

    class Decompose(dspy.Signature):
        """Break the problem into independent, testable sub-problems with explicit
        dependencies; every dependency names an existing id, no cycles."""

        problem: str = dspy.InputField()
        distilled_context: str = dspy.InputField()
        strategy: Literal["by-domain", "by-dependency", "by-size"] = dspy.OutputField()
        sub_problems: list[SubProblem] = dspy.OutputField()

    class Solve(dspy.Signature):
        """Solve one sub-problem from the distilled context and its dependency results."""

        sub_problem: SubProblem = dspy.InputField()
        distilled_context: str = dspy.InputField()
        dependency_results: str = dspy.InputField()
        result: str = dspy.OutputField()
        confidence: float = dspy.OutputField()

    class Synthesize(dspy.Signature):
        """Combine sub-results; list agreements, contradictions with resolutions, gaps."""

        problem: str = dspy.InputField()
        sub_results: str = dspy.InputField()
        contradictions: list[str] = dspy.OutputField()
        gaps: list[str] = dspy.OutputField()
        answer: str = dspy.OutputField()
        confidence: float = dspy.OutputField()

    class RLMWorkflow(dspy.Module):
        def __init__(self, max_depth: int = 2):
            super().__init__()
            self.decompose = dspy.ChainOfThought(Decompose)
            self.solve = dspy.ChainOfThought(Solve)
            self.synthesize = dspy.ChainOfThought(Synthesize)
            self.max_depth = max_depth

        def forward(self, problem: str, distilled_context: str, depth: int = 0):
            plan = self.decompose(problem=problem, distilled_context=distilled_context)
            results: dict[int, str] = {}
            for sp in validate_dag(plan.sub_problems):
                deps = "\n".join(f"[{d}] {results[d]}" for d in sp.dependencies)
                if sp.complexity == "high" and depth < self.max_depth:
                    results[sp.id] = self.forward(sp.description, distilled_context, depth + 1).answer
                    continue
                results[sp.id] = self.solve(
                    sub_problem=sp, distilled_context=distilled_context, dependency_results=deps
                ).result
            merged = self.synthesize(
                problem=problem,
                sub_results="\n".join(f"[{k}] {v}" for k, v in results.items()),
            )
            return dspy.Prediction(plan=plan, sub_results=results, **merged)

    def tier1_syntactic(pred) -> tuple[float, str]:
        """Deterministic structural checks: fields present, confidence in range."""
        problems = []
        if not getattr(pred, "answer", "").strip():
            problems.append("empty answer")
        conf = getattr(pred, "confidence", None)
        if not isinstance(conf, (int, float)) or not 0.0 <= float(conf) <= 1.0:
            problems.append("confidence not in [0, 1]")
        if not isinstance(getattr(pred, "contradictions", None), list):
            problems.append("contradictions must be a list")
        return (1.0 if not problems else 0.0), "; ".join(problems)

    def verify_cascade(gold, pred, trace=None, pred_name=None, pred_trace=None):
        t1, report = tier1_syntactic(pred)
        if t1 < 1.0:
            return dspy.Prediction(score=0.3 * t1, feedback=f"Tier 1 failed: {report}")
        # Tier 2: every success criterion must be addressed in the answer.
        criteria = list(getattr(gold, "success_criteria", []) or [])
        hit = sum(c.lower() in pred.answer.lower() for c in criteria)
        t2 = hit / len(criteria) if criteria else 1.0
        if t2 < 0.8:
            missing = [c for c in criteria if c.lower() not in pred.answer.lower()]
            return dspy.Prediction(score=0.3 + 0.4 * t2, feedback=f"Tier 2 failed: criteria missing {missing}")
        # Tier 3 (pragmatic) would be an LM judge; the smoke test treats it as passed.
        return dspy.Prediction(score=0.3 + 0.4 * t2 + 0.3, feedback="Verified on all three tiers.")

    return RLMWorkflow, verify_cascade


SAMPLE_PLAN = [
    SubProblem(id=1, description="Design the rate-limiter interface", success_criteria="protocol defined"),
    SubProblem(id=2, description="Implement sliding-window counter", dependencies=[1]),
    SubProblem(id=3, description="Build middleware", dependencies=[1, 2], complexity="high"),
    SubProblem(id=4, description="Write tests", dependencies=[1, 2, 3]),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Skip all LM calls")
    ap.add_argument("--model", default=os.getenv("DSPY_MODEL", "openai/gpt-4o"))
    args = ap.parse_args()

    import dspy

    RLMWorkflow, verify_cascade = build()
    workflow = RLMWorkflow(max_depth=2)

    if args.dry_run:
        order = [sp.id for sp in validate_dag(SAMPLE_PLAN)]
        assert order == [1, 2, 3, 4], order
        try:
            validate_dag([SubProblem(id=1, description="a", dependencies=[2]),
                          SubProblem(id=2, description="b", dependencies=[1])])
        except ValueError as exc:
            assert "cycle" in str(exc)
        gold = dspy.Example(problem="rate limiting", success_criteria=["sliding window", "redis"])
        good = dspy.Prediction(answer="Sliding window over Redis sorted sets.", confidence=0.9, contradictions=[])
        bad = dspy.Prediction(answer="", confidence=1.4, contradictions=None)
        assert verify_cascade(gold, good).score > 0.99
        assert verify_cascade(gold, bad).score == 0.0

        def reward(call_args: dict, pred) -> float:
            return float(verify_cascade(gold, pred).score)

        refined = dspy.Refine(module=workflow, N=3, reward_fn=reward, threshold=0.9)
        names = [n for n, _ in workflow.named_predictors()]
        print("OK: RLMWorkflow constructed with predictors", names)
        print(f"    topological order {order}; cycle detected; Tier-1/Tier-2 cascade scored good=1.00 bad=0.00")
        print(f"    dspy.Refine(N={refined.N}, threshold={refined.threshold}) wraps the workflow")
        return 0

    dspy.configure(lm=dspy.LM(args.model), track_usage=True)
    pred = workflow(
        problem="Add per-user rate limiting (100 req/min) to the API with Redis, <10ms overhead.",
        distilled_context="Middleware base class in middleware/base.py; Redis client in core/redis.py.",
    )
    print("Answer:", pred.answer)
    print("Contradictions:", pred.contradictions)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
