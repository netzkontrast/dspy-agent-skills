---
name: dspy-adversarial-review
description: >-
  Add an independent judge to a DSPy pipeline that cannot rewrite what it
  judges — a reviewer LM that is asserted to differ from the writer scores an
  artifact against its evidence (overstated claims with the evidence they
  would need, unsupported claims, strengths, weaknesses, a 1–10 score), a
  citation-support check grades each cited span, a demotion rule turns too
  many overstated claims into a status change, dspy.Refine re-runs the
  writer until the review score meets a target, and judgements are cached
  by content hash. A precision/recall metric on the judge itself keeps GEPA
  from merely making it harsher. Use before a page, claim or draft gains
  authority.
when_to_use: >-
  User says "review this", "red team", "adversarial review", "second
  opinion", "is this overstated", "check the citations", "refine until it
  passes"; a page is about to be marked reviewed or high-confidence; a
  draft must reach a target quality score; a pipeline's verifier is the same
  model that wrote the output and must be replaced by an independent one.
---

# DSPy Adversarial Review (3.3.x)

Four repos converge on one pattern: a second model reads what the first
wrote and is only allowed to object. `synthadoc` runs an adversarial lint
pass that demotes a page when it flags too many overstated claims; `AutoSci`
runs `/review` with an independent Review LLM at three difficulties and
`/refine` in rounds until a target score; `quicky-wiki` red-teams
high-confidence claims; `llm-wiki-compiler` judges whether a cited span
supports its claim and caches the verdict. This skill is that pattern as
DSPy constructs, with the independence rule enforced in code.

`dspy-autodialectics` keeps a *program run* honest (contract, thesis /
antithesis / synthesis, fake completion). This skill reviews an *artifact*
against *evidence* with a *different model*. Use both when a program writes
knowledge that people will rely on.

## Constructs

| Construct | What it does | Deterministic |
|---|---|---|
| `assert_independent(reviewer, writer)` | refuses the same object, or the same model string with the same kwargs | yes |
| `ReviewArtifact` | artifact + evidence + `difficulty ∈ {standard, hard, adversarial}` → `Review{score 1–10, overstated[{quote, why, evidence_needed}], unsupported[], strengths[], weaknesses[]}` | — |
| `CitationSupport` | claim + cited span → `supported \| partial \| unsupported` with a reason | — |
| `demotion(review, k)` | `contested` when `len(overstated) ≥ k`; a status change, never an edit | yes |
| cache | `sha256(artifact, evidence, difficulty, reviewer model)` → judgement | yes |
| `review_refine(writer, reviewer, evidence)` | `dspy.Refine(module=writer, N, reward_fn=review score / 10, threshold=0.8)` | rule |
| `judge_metric(gold, pred)` | F1 of `overstated` and `unsupported` against a labelled set, feedback names missed and spurious items | yes |

## Canonical program

```python
class AdversarialReviewer(dspy.Module):
    def __init__(self, reviewer_lm, demote_at: int = 3):
        super().__init__()
        self.review = dspy.ChainOfThought(ReviewArtifact)
        self.support = dspy.Predict(CitationSupport)
        self.reviewer_lm, self.demote_at, self.cache = reviewer_lm, demote_at, {}

    def forward(self, artifact: str, evidence: str, difficulty: Difficulty = "standard"):
        assert_independent(self.reviewer_lm, dspy.settings.lm)      # the judge is not the author
        key = digest(artifact, evidence, difficulty, str(self.reviewer_lm.model))
        if key in self.cache:
            return self.cache[key]
        with dspy.context(lm=self.reviewer_lm):
            review = self.review(artifact=artifact, evidence=evidence, difficulty=difficulty).review
        self.cache[key] = dspy.Prediction(review=review, verdict=demotion(review, self.demote_at), cache_key=key)
        return self.cache[key]
```

The writer stays under `dspy.configure(lm=writer)`; the reviewer runs inside
`dspy.context(lm=reviewer_lm)`. Difficulty changes the instruction (`hard`
demands a citation per factual sentence, `adversarial` also attacks framing
and omissions), never the threshold.

## Refine to a target

```python
refined = review_refine(writer=Drafter(), reviewer=reviewer, evidence=evidence, rounds=3)
best = refined(question=q)          # re-runs Drafter until the reviewer scores ≥ 0.8 or rounds are spent
```

`dspy.Refine` receives the review score as its reward, so the loop is
"write → independent review → write again", never "write → self-check".

## The judge has its own metric

`judge_metric` scores the reviewer against a labelled set of overstated and
unsupported claims and names what it missed and what it invented. Optimising
the reviewer with GEPA on this metric improves precision *and* recall; a
reviewer that flags everything scores low on precision, one that flags
nothing scores low on recall. Without this metric an "adversarial" reviewer
optimises toward harshness.

## Rules

1. **Reviewer ≠ writer**, asserted at call time. Same vendor with a different
   model is acceptable; a different vendor is stronger.
2. **The reviewer never proposes text.** Its output is quotes, reasons and
   the evidence that would be needed.
3. **Demotion changes status, not content.** A demoted page keeps its body
   and its history.
4. **Cache by content.** The same artifact against the same evidence is judged once.
5. **Evidence is the only ground.** What the reviewer knows from elsewhere is not a finding.

## Anti-patterns

- Passing the writer's LM as the reviewer "to save a config line"; the guard exists because it happens.
- Letting the reviewer rewrite the artifact; that makes it a second writer with no reviewer.
- Raising the threshold instead of the difficulty when a page "feels" weak.
- Skipping the judge metric and calling the reviewer optimised because it finds more.

## Where to go next

- Keeping the program's own run honest (contract, dialectic, fake completion) → `dspy-autodialectics`
- Fixing a page the reviewer demoted → `dspy-wiki-compile` (re-compile) or `dspy-deep-refine`
- Metric contract → `dspy-evaluation-harness`; optimising the reviewer → `dspy-gepa-optimizer`
- Full reference (models, enums, feedback strings, difficulty contracts) → [reference.md](reference.md)
- Runnable example → [example_adversarial_review.py](example_adversarial_review.py)
