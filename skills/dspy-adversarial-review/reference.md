# DSPy Adversarial Review — Reference

Sources of the pattern:

| Repo | Contribution here |
|---|---|
| `synthadoc` (AGPL, patterns only) | adversarial review pass in lint; threshold demotion of a page to `contradicted`; the page body is never rewritten by the pass |
| `AutoSci` (MIT) | `/review` with an independent Review LLM at `standard · hard · adversarial`; structured score 1–10 with strengths and weaknesses; `/refine` loop to a target score with a score trajectory |
| `quicky-wiki` (MIT) | red-team of high-confidence claims as a periodic "metabolism" pass |
| `llm-wiki-compiler` | citation-support judge (`supported · partial · unsupported`) with cached judgements |

## Models

```python
class Overstated(BaseModel):  quote: str; why: str; evidence_needed: str
class Review(BaseModel):      score: int (1..10); overstated: list[Overstated]; unsupported: list[str]
                              strengths: list[str]; weaknesses: list[str]
class Support(BaseModel):     verdict: Literal["supported", "partial", "unsupported"]; reason: str
Difficulty = Literal["standard", "hard", "adversarial"]
```

`quote` is verbatim from the artifact so the caller can locate it; the
metric matches quotes with substring tolerance in both directions.

## Difficulty contracts (instruction, not threshold)

| Difficulty | The reviewer must |
|---|---|
| `standard` | flag claims the evidence does not support and superlatives without comparison |
| `hard` | additionally demand a citation for every factual sentence |
| `adversarial` | additionally attack the framing (what the artifact implies) and the omissions (what the evidence says that the artifact leaves out) |

## Independence guard

```python
def same_lm(a, b) -> bool:  a is b or (a.model == b.model and a.kwargs == b.kwargs)
def assert_independent(reviewer, writer) -> None   # ValueError when writer is None or same_lm
```

Called at the top of `forward`, against `dspy.settings.lm`, so the guard
holds whatever the caller configured. Two `dspy.LM` objects with the same
model string but different `temperature` pass the guard; that is a weak
independence and the skill says so.

## Demotion and cache

```python
demotion(review, demote_at=3) -> "contested" | "keep"      # len(overstated) >= demote_at
digest(*parts) -> 16-hex sha256                             # (artifact, evidence, difficulty, reviewer model)
```

The cache lives on the module instance; persist it as JSON keyed by the
digest when reviews are expensive.

## Refine

```python
review_reward(reviewer, evidence, difficulty) -> reward(args, pred) -> float   # pred.text reviewed, score / 10
review_refine(writer, reviewer, evidence, rounds=3) -> dspy.Refine(module=writer, N=rounds, reward_fn=…, threshold=0.8)
```

`dspy.Refine` re-runs the writer with the previous attempt's feedback in
context; the reward is the independent review, so the writer learns from the
judge, not from itself. The writer's prediction must expose `.text`; adapt
the reward for other field names.

## Judge metric

```python
judge_metric(gold, pred, trace=None, pred_name=None, pred_trace=None) -> dspy.Prediction(score, feedback)
```

`gold.overstated: list[str]` and `gold.unsupported: list[str]` are the
labelled quotes. Score is the mean of the two F1s. Feedback strings:

| Situation | Feedback |
|---|---|
| all found, none invented | `every overstated and unsupported claim found, none invented` |
| recall miss | `missed overstated claims: [...]` / `missed unsupported claims: [...]` |
| precision miss | `flagged supported claims as overstated: [...]` / `flagged cited claims as unsupported: [...]` |

A labelled set of forty claims, half overstated by construction, is enough
for `dspy.GEPA(auto="light")` on `review.predict`.

## Extension points

- **Citation pass.** Run `check_citation` for every `(claim, span)` pair of a
  page and demote on `unsupported` count as well as on `overstated`.
- **Second vendor.** `reviewer_lm = dspy.LM("openai/…")` under an Anthropic
  writer, or the reverse; the guard is satisfied either way.
- **Trajectory.** Wrap `review_refine` to record `(round, score)` pairs and
  stop early on a plateau, as AutoSci's `/refine` reports.
- **Milestone red-team.** Sample `confidence: high` pages and review them at
  `adversarial` before a release; demotions go to the human, never to auto-fix.
