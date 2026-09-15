# DSPy TetraFrame — Reference

Source: `Hmbown/tetraframe-dspy` (MIT). The upstream package has ten
stages, a `CornerInputView` with a blocked-field list, four corner
signatures, pairwise plus global cartography, a transformer wrapped in
`dspy.BestOfN`, and a `VerificationSuite` of eight heuristics with fixed
thresholds. This skill keeps the method and the thresholds and folds the
stages into six predictors so the program fits one file.

## Stage mapping (upstream → this skill)

| Upstream stage | Here | Notes |
|---|---|---|
| SeedDistill | `DistillSeed` → `Distilled` | stakes, constraints, hidden assumptions, candidate predicates, `frame_risk_score`, evaluation criteria |
| SplitPredicate + ChoosePredicate | `SelectPredicate` → `Selection` | one primary predicate, rejected candidates with reasons |
| GenerateCorner{P,NotP,Both,Neither} + Harden | `CornerP` … `CornerNeither` (subclasses of `GenerateCorner`) | generation and hardening in one pass; `patched_claim` is the hardened claim |
| PairwiseRelation + GlobalCartography + Arbiter | `MapCorners` → `Cartography` | contradictions, complementarities, discriminators, invariants, arbiter notes |
| TransformFrame (BestOfN N=3, threshold 0.84) | `Transform` under `dspy.BestOfN` | same reward shape, same threshold |
| VerifyRun | `verify(run)` | seven checks; upstream's `robustness` (re-run stability) is a harness concern, run twice and compare |

## Models

```python
class Distilled(BaseModel):
    normalized_seed: str
    stakes: list[str]; constraints: list[str]; hidden_assumptions: list[str]
    candidate_predicates: list[str]
    frame_risk_score: float          # 0..1, high when the seed bundles objectives
    evaluation_criteria: list[str]

class Selection(BaseModel):
    primary_predicate: str
    rejected: list[str]; rationale: str

class CornerView(BaseModel):         # isolation boundary
    normalized_seed: str; stakes: list[str]; constraints: list[str]
    hidden_assumptions: list[str]; primary_predicate: str
    evaluation_criteria: list[str]; corner_contract: str

class Corner(BaseModel):
    mode: Literal["P", "not-P", "both", "neither"]
    core_claim: str; strongest_case: str
    scope_conditions: list[str]; evidence_needs: list[str]; unique_signal: str
    basis_label: str; basis_explanation: str; replacement_predicate: str
    patched_claim: str; minimal_falsifiers: list[str]; confidence_score: float

class Cartography(BaseModel):
    contradiction_map: list[str]; complementarity_map: list[str]
    discriminators: list[str]; invariants: list[str]; arbiter_notes: str

class Frame(BaseModel):              # P*
    transformed_predicate: str; transformed_frame: str
    survivors_from_p: list[str]; survivors_from_not_p: list[str]
    hidden_structure_from_both: list[str]; dissolved_false_frame_from_neither: list[str]
    operational_tests: list[str]

class Run(BaseModel):
    seed: str; distilled: Distilled; selection: Selection
    corners: dict[str, Corner]; cartography: Cartography; frame: Frame
```

## Closed vocabularies

| Field | Values |
|---|---|
| `Corner.basis_label` for **both** | `temporal_split`, `scale_split`, `role_split`, `ontology_split`, `context_split`, `layered_causality`, `admissible_paradox` |
| `Corner.basis_label` for **neither** | `category_error`, `false_binary`, `overloaded_predicate`, `missing_latent_variable`, `bad_ontology`, `ill_posed_objective`, `frame_collapse_under_scrutiny` |
| `Corner.basis_label` for P / not-P | `affirmation` / `rejection` |
| incompatible pairs (independence check) | (P, not-P), (P, neither), (not-P, neither) |
| compromise phrases (penalised) | *middle ground, balanced approach, split the difference, on the one hand, on the other hand* |
| mush words (slop check) | *balanced, nuanced, important, helpful, complex, thoughtful, consider, various, multiple* |

Upstream cartography also carries eight relation types per pair
(`support`, `contradiction`, `complementarity`, `paradox`, `category_error`,
`frame_dependency`, `evidence_discriminator`, `scale_dependency`). Add a
`PairwiseRelation` model with `relation_type: Literal[...]` when you need the
pair matrix; the global map is enough for gating a decision.

## Guards

```python
def assert_isolation(view: CornerView) -> None
    # raises ValueError if the dumped view has any field outside CornerView.model_fields

def near_duplicates(corners: dict[str, Corner], seed: str, threshold=0.78) -> list[tuple[str, str]]
    # Jaccard over residual tokens (seed tokens removed) of core_claim; pairs at/above threshold
```

Corner sampling: `dspy.settings.lm.copy(rollout_id=i, temperature=t)` with
`t = 0.7` for P / not-P and `0.9` for both / neither. Providers that ignore
`rollout_id` still get distinct temperatures; providers that strip both
(e.g. a CLI-backed LM) rely on the contract docstrings alone — verify
`branch_independence` more strictly there.

## Verification heuristics

| Check | Formula (deterministic) |
|---|---|
| `branch_independence` | `1 − 0.6 · mean(max(0, sim(residual_a, residual_b) − 0.35))` over the incompatible pairs |
| `rigor_of_both` | mean(basis ∈ BOTH_BASES, explanation mentions co-holding, scope conditions present, falsifier quality) − 0.4 if `strongest_case` contains a compromise phrase |
| `rigor_of_neither` | mean(failure ∈ NEITHER_FAILURES, replacement predicate present, diagnosis ≥ 8 words, falsifier quality) − 0.3 if "it depends" |
| `contradiction_honesty` | `0.4 + 0.1·|contradictions| + 0.05·|discriminators|`, capped at 1.0; 0.35 / 0.25 when no contradiction is named |
| `transformation_quality` | `min(1, mean(survivor lists + tests present) + 0.3·(1 − overlap(P*, P ∪ not-P)))` then − 0.4 for compromise phrases |
| `fake_novelty_risk` | `1 − min(0.6, 0.08 · unsupported P* tokens)` (tokens > 4 chars absent from every corner, invariant and survivor list) |
| `slop_risk` | `1 − 3 · mush-word ratio` over P* and the patched claims |

`falsifier quality` = mean over `minimal_falsifiers` of 1.0 for ≥ 5 words,
0.5 otherwise, 0.0 when empty.

Deviation from upstream, on purpose: `transformation_quality` caps at 1.0
*before* the compromise penalty, so a compromise P* with all lists filled
scores 0.60 (< 0.82) instead of slipping through at 0.88.

## Thresholds

```python
THRESHOLDS = {"branch_independence": 0.90, "rigor_of_both": 0.78, "rigor_of_neither": 0.78,
              "contradiction_honesty": 0.75, "transformation_quality": 0.82,
              "fake_novelty_risk": 0.70, "slop_risk": 0.70}
```

A run passes when every check meets its threshold. Report the table to the
human either way; a failed `rigor_of_neither` on a seed that really has a
winner is information, not noise.

## Reward and metric

```python
def transform_reward(args: dict, pred) -> float
    # BestOfN reward: mean(four survivor lists non-empty) − 0.4 for compromise phrases in P*

def tetraframe_metric(gold, pred, trace=None, pred_name=None, pred_trace=None) -> dspy.Prediction
    # score = mean(verify(run).values()) × 0.5 if any gold.banned_transformed_phrases occurs in P*
    # feedback lists every check under threshold and the banned phrases found
```

Gold fields the metric reads (all optional): `banned_transformed_phrases`.
Extend with `expected_primary_predicate_contains`,
`allowed_both_basis`, `expected_neither_failure_modes` when you have labels;
each adds a named deficit to `feedback`.

## Feedback strings

| Situation | Feedback |
|---|---|
| all checks pass | `independent corners, rigorous both/neither, transformed P*` |
| a check fails | `<check> <score> < <threshold>` (joined with `; `) |
| banned phrase | `P* uses banned phrases ['…']` |

## Extension points

- **LM judge on top of the heuristics.** Add a `JudgeRun` signature that
  scores `contradiction_honesty` and `transformation_quality` semantically
  and take the minimum of judge and heuristic; keep the heuristics as the
  floor so optimization cannot talk its way past them.
- **Domain context.** Give `DistillSeed` an extra `context: str` input
  (glossary, canonical passages) and copy it into `CornerView` — it is
  shared, seed-level context, not another corner's output.
- **Persisting runs.** Dump `Run.model_dump_json()` next to the decision
  record; the decision cites the file, never the chat.
- **Robustness.** Run the same seed twice with different `rollout_id`
  offsets and compare `selection.primary_predicate` and the P* survivor
  lists; upstream's threshold for agreement is 0.70.
