---
name: dspy-tetraframe
description: Critically assess a contested decision with the TetraFrame method in DSPy — distill the seed into one falsifiable predicate, generate four corners in strict isolation (P, not-P, both under a typed split, neither with a replacement predicate), map their contradictions and evidence discriminators, then produce a transformed frame P* with dspy.BestOfN that keeps the survivors of every corner instead of averaging them. A deterministic verification suite (branch independence, rigor of both/neither, contradiction honesty, transformation quality, fake novelty, slop) blocks collapsed runs and doubles as the GEPA metric. Use before any decision that changes authority or is hard to reverse.
when_to_use: >-
  User says "tetraframe", "assess this decision", "steelman both sides",
  "are we asking the wrong question", "before we decide"; a knowledge base
  or wiki is about to be changed in a way that merges, supersedes or deletes
  pages; a research claim contradicts the canonical store; a design or
  storyform decision has two camps; or a question page must become a
  recorded decision.
---

# DSPy TetraFrame (3.2.x)

Port of `tetraframe-dspy` (Hmbown, MIT) as a compact DSPy program. The
method refuses the two lazy outcomes of a debate: picking a side and
splitting the difference. It forces four *independent* positions on one
predicate, maps how they relate, and asks a transformer for a frame that
survives all four. Every stage is typed, every anti-collapse rule is code,
and the verification suite returns scores that a human reads before deciding.

The output is never a decision. It is the strongest possible material for
one: four hardened positions, their contradictions, the evidence that would
discriminate between them, and a candidate reframing with operational tests.

## Where it plugs in

| Stage | Seed | After the run |
|---|---|---|
| question → decision (research wiki, ADR, D-xx record) | the open question with its candidate answers | the decision record cites the run file; the question page lists P/not-P/both/neither as candidate answers |
| knowledge-base change (`dspy-deep-refine`, wiki merge/supersede/delete) | "page A should replace page B" | the change proposal carries P* and its tests; the human approves or rejects it |
| promotion with conflict (`dspy-clarify` verdict `needs-author`, relation *contradicts*) | the clarified claim vs. the canonical statement | the conflict page lists the discriminating evidence to fetch next |
| context-heavy plan (`dspy-rlm-workflow` Initialize) | a plan whose objective bundles two goals | the plan is decomposed from P*, not from the bundled seed |

Run `dspy-clarify` on the seed first when it is vague; TetraFrame on a
vague predicate produces four vague corners.

## Canonical program

```python
import dspy
from pydantic import BaseModel, Field
from typing import Literal

Mode = Literal["P", "not-P", "both", "neither"]
BOTH_BASES = ("temporal_split", "scale_split", "role_split", "ontology_split",
              "context_split", "layered_causality", "admissible_paradox")
NEITHER_FAILURES = ("category_error", "false_binary", "overloaded_predicate",
                    "missing_latent_variable", "bad_ontology", "ill_posed_objective",
                    "frame_collapse_under_scrutiny")

class CornerView(BaseModel):          # the ONLY input a corner generator sees
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
    basis_label: str = ""             # both: one of BOTH_BASES; neither: one of NEITHER_FAILURES
    basis_explanation: str = ""
    replacement_predicate: str = ""   # neither only
    patched_claim: str = ""           # after the internal attack
    minimal_falsifiers: list[str] = Field(default_factory=list)
    confidence_score: float = Field(ge=0, le=1, default=0.5)

class GenerateCorner(dspy.Signature):
    """Generate and harden one corner from the view alone; never mention other corners."""
    view: CornerView = dspy.InputField()
    corner: Corner = dspy.OutputField()

class CornerBoth(GenerateCorner):
    """Both, not compromise: pick one basis from the typed splits as basis_label and show why P and not-P hold under it."""

class CornerNeither(GenerateCorner):
    """Neither, not evasion: name the failure mode as basis_label and propose a replacement_predicate."""

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
        self.map = dspy.ChainOfThought(MapCorners)
        self.transform = dspy.BestOfN(module=dspy.ChainOfThought(Transform), N=3,
                                      reward_fn=transform_reward, threshold=0.84)

    def forward(self, seed: str):
        d = self.distill(seed=seed).distilled
        s = self.select(distilled=d).selection
        corners = {}
        for i, mode in enumerate(MODES):
            view = CornerView(..., corner_contract=CONTRACTS[mode])
            assert_isolation(view)                                   # no cross-corner leakage
            lm = dspy.settings.lm
            with dspy.context(lm=lm.copy(rollout_id=i, temperature=0.9 if mode in ("both", "neither") else 0.7)):
                corners[mode] = self.generators[mode](view=view).corner
        cartography = self.map(corners=list(corners.values())).cartography
        frame = self.transform(primary_predicate=s.primary_predicate,
                               corners=list(corners.values()), cartography=cartography).frame
        return dspy.Prediction(run=Run(seed=seed, distilled=d, selection=s,
                                       corners=corners, cartography=cartography, frame=frame))
```

Full models, signatures and heuristics: [example_tetraframe.py](example_tetraframe.py).

## The anti-collapse rules are code, not prompts

| Rule | Where it lives |
|---|---|
| a corner sees only its `CornerView` | `assert_isolation` raises on any extra field; the four generators never receive each other's output |
| corners are sampled independently | `lm.copy(rollout_id=i, temperature=…)` per corner; both/neither run hotter |
| *both* is a typed split, not a compromise | `basis_label ∈ BOTH_BASES`; compromise phrases in `strongest_case` cost 0.4 |
| *neither* names a failure mode and a replacement | `basis_label ∈ NEITHER_FAILURES`; empty `replacement_predicate` or "it depends" is penalised |
| near-duplicate corners are regenerated | `near_duplicates` compares residual tokens (seed removed) at Jaccard ≥ 0.78 |
| P* is not an average | `transform_reward` for `dspy.BestOfN`: all four survivor lists present, compromise phrases −0.4 |

## Verification suite (thresholds from upstream)

| Check | Threshold | What it measures |
|---|---|---|
| `branch_independence` | 0.90 | residual similarity of the incompatible pairs (P/not-P, P/neither, not-P/neither) |
| `rigor_of_both` | 0.78 | typed basis, co-holding explained, scope conditions, falsifier quality |
| `rigor_of_neither` | 0.78 | named failure mode, replacement predicate, diagnosis length, no evasion |
| `contradiction_honesty` | 0.75 | contradictions named and evidence discriminators listed, not smoothed into complementarity |
| `transformation_quality` | 0.82 | survivors from every corner, tests present, P* not a rephrase of P or not-P, no compromise language |
| `fake_novelty_risk` | 0.70 | tokens in P* that appear in no corner, invariant or survivor list |
| `slop_risk` | 0.70 | density of filler words (*balanced, nuanced, important, various* …) |

`verify(run)` is deterministic; `tetraframe_metric(gold, pred)` averages it
into `dspy.Prediction(score, feedback)` and halves the score when P* uses a
phrase listed in `gold.banned_transformed_phrases`. The feedback names every
check under threshold, so GEPA gets concrete blame.

## Rules the method enforces on the pipeline

1. **The run is not the decision.** Present the four corners, the
   contradiction map and P* to the human; record their choice separately and
   cite the run.
2. **A failed check blocks the run, not the decision.** Re-run with a
   sharper seed or stronger anti-collapse hints; never edit a corner by hand
   to pass verification.
3. **One predicate per run.** Bundled seeds get a high `frame_risk_score`
   from distillation; split them and run twice.
4. **P* must be testable.** `operational_tests` and `minimal_falsifiers`
   are what the decision record inherits; a P* without them is a slogan.
5. **Dry-run first.** The program returns a `Run`; writing it (decision
   record, question page, change proposal) is the caller's separate step.

## Gold set and optimization

Ten to twenty seeds with hand-labelled expectations — the expected primary
predicate's key phrase, the admissible `both` bases, the expected `neither`
failure modes, banned P* phrases — are enough for `dspy.GEPA(auto="light")`
over the corner generators and the transformer. Include seeds whose right
answer is *neither* (the question was wrong) so the optimizer does not learn
that every debate has a winner.

## Anti-patterns

- Passing the other corners "for context" — the isolation guard exists
  because the model will otherwise write four paragraphs of one essay.
- Treating *both* as "a bit of each": if no typed split applies, the honest
  *both* corner has low confidence and says so.
- Lowering the thresholds to make a run pass; the thresholds are the method.
- Running on a seed that is a plan or a request rather than a claim about
  the world — distill it, or clarify it first.
- Letting P* introduce a new noun that no corner produced (`fake_novelty_risk`).

## Where to go next

- Sharpen the seed → `dspy-clarify`
- Metric contract → `dspy-evaluation-harness`
- Optimizing the corner generators → `dspy-gepa-optimizer`
- Knowledge-base changes that this method gates → `dspy-deep-refine`
- Full reference (models, verification heuristics, feedback strings, extension points) → [reference.md](reference.md)
- Runnable example → [example_tetraframe.py](example_tetraframe.py)
