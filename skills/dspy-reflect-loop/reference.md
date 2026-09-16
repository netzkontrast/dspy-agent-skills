# DSPy Reflect Loop — Reference

Source: `claude-reflect-system` v1.3.0 (`reflect/scripts/extract_signals.py`,
`semantic_detector.py`, `learning_ledger.py`, `scope_analyzer.py`,
`promote_learning.py`, `meta_learning.py`, `present_review.py`,
`references/signal-patterns.md`) mapped onto DSPy 3.3.1.

## Signal taxonomy (from `signal-patterns.md`)

| Confidence | Kind | Original patterns | Action in DSPy terms |
|---|---|---|---|
| HIGH | correction | "no, don't use X, use Y", "actually … is …", "instead of X … Y", "never/always …", DE: "nein …", "verwende/benutze X statt Y", "immer/niemals …" | gold example with `forbidden_behavior` + `expected_behavior`; metric emits the correction as feedback |
| HIGH | explicit | "remember: …", "merk dir: …" | same, without a failing input (instruction-level learning) |
| MEDIUM | approval | "yes, perfect/exactly/correct", "works perfectly", "good job on …" | positive demo for bootstrap stages |
| LOW | observation | "have you considered …", "why not try …", "what about …" | logged; surfaced as a review item, no optimization signal |

False-positive rules kept from the original: only user turns; approvals must
follow an assistant turn; ignore messages under 10 characters; attribute a
learning only to programs/skills actually used in the session.

## Models

```python
from pydantic import BaseModel, Field
from typing import Literal

class LearningSignal(BaseModel):
    is_learning: bool
    kind: Literal["correction", "approval", "observation", "explicit"] | None = None
    confidence: Literal["HIGH", "MEDIUM", "LOW"] | None = None
    old_behavior: str = ""
    new_behavior: str = ""
    learning: str = ""
    scope_hint: Literal["program", "project", "global"] = "program"
    reasoning: str = ""

class Learning(BaseModel):                       # ledger row
    fingerprint: str                            # sha256(normalized learning)[:16]
    learning: str
    kind: str
    confidence: str
    program: str
    contexts: list[str] = Field(default_factory=list)   # repo/program ids where seen
    count: int = 1
    status: Literal["pending", "accepted", "skipped", "promoted"] = "pending"
    first_seen: str
    last_seen: str

class ReviewDecision(BaseModel):
    fingerprint: str
    decision: Literal["accept", "modify", "skip", "quit"]
    modification: str = ""
```

`kind → confidence` mapping is deterministic (`correction`/`explicit` → HIGH,
`approval` → MEDIUM, `observation` → LOW); the Signature may lower but never
raise it.

## Pre-filter

A regex over user turns, EN + DE, tuned for recall. It gates which turns reach
the `ExtractLearningSignal` predictor (the original's `--semantic` mode was
opt-in because it cost ~2–3 s per message; here the pre-filter keeps the LM
calls to the handful of turns that could carry a learning). Extend the pattern
per language; test on ten real transcripts (target from the original: >80 %
precision, >60 % recall at the signal level).

## Ledger

JSONL (`.reflect/learnings.jsonl`) or SQLite; one row per fingerprint.

| Operation | Semantics |
|---|---|
| `record(signal, context_id)` | upsert by fingerprint; append `context_id` if new; `count += 1`; keep max confidence |
| `pending()` | rows with `status == "pending"` |
| `eligible_for_promotion(threshold=2)` | `len(contexts) >= threshold and status != "promoted"` |
| `mark(fingerprint, status)` | after review / promotion |

Context id: `sha256(git remote origin url)[:12]`, fallback `sha256(cwd)`;
for programs, `program_name`.

## Scope

`scope_hint` comes from the Signature; the ledger decides promotion. Heuristics
from `scope_analyzer.py` worth keeping as a deterministic tie-break: project
indicators (paths like `src/components/`, `docker-compose`, specific hosts) vs
global indicators ("run tests", "commit message", "never commit secrets", tool
preferences like `uv`, `pytest`, `ruff`; DE "immer", "niemals", "verwende").

## Review and apply contract

```python
def review(signals: list[LearningSignal]) -> list[ReviewDecision]: ...   # human; A/M/S/Q per item
def apply(decisions, *, gold_path, backup_dir) -> ApplyReport: ...        # only accepted/modified
```

`apply` does, in order: backup the target (`<file>.<timestamp>.bak`), write,
validate (JSON lines parse; Pydantic for examples), roll back on error, log
each decision to `meta/feedback-log.jsonl`, and print the git command. It never
commits or pushes. A lock file plus `last-reflection.timestamp` prevent double
runs; auto-mode (Stop hook) is opt-in and detached.

## Conversion to optimization signal

| Signal | Becomes | Consumed by |
|---|---|---|
| HIGH correction with a failing input | `dspy.Example(inputs…, expected_behavior, forbidden_behavior, feedback)` appended to the program's trainset | `dspy.GEPA` via `corrections_metric` folded into the program metric |
| HIGH explicit (no input) | entry in the family's shared instruction prefix / a synthetic example | GEPA |
| MEDIUM approval | positive demo (input + the approved output) | `BootstrapFewShot`, `BetterTogether(bootstrap=…)` |
| LOW observation | review item only | — |

`corrections_metric` weights: violation → 0.0; expected present → 1.0; neither →
0.5. Fold it into the program's real metric with a weight (e.g. 0.3) so learned
corrections cannot be traded away for other axes.

## Meta-learning (reflect-on-reflect)

Original: `feedback-log.jsonl` → per-pattern acceptance rates → statuses
`insufficient_data (<5)`, `deprecated (<0.20)`, `needs_review (<0.5)`,
`healthy`, `excellent (≥0.80)`; opt-in `--use-meta` adjusts confidence
(+0.1 / −0.15…−0.3). Here the same log is gold for the extractor:

- gold: `accepted = [{learning, confidence}]`, `skipped = [learning]` per transcript;
- `reflector_metric` rewards recovered accepted learnings and penalizes
  proposals the user skipped before;
- `dspy.GEPA(metric=reflector_metric, auto="light")` on `ReflectLoop` after
  ≥ 5 reviewed sessions (the original's `MIN_SAMPLES`).

Keep the deterministic statistics report (`/reflect-meta`) — it is the
human-readable view of the same log.

## Failure modes

| Symptom | Cause | Fix |
|---|---|---|
| same learning proposed every session | ledger not consulted | `record` before proposing; skip fingerprints with `status in {accepted, skipped, promoted}` |
| corrections do not stick after GEPA | metric weight too low or example lacks the failing input | weight ≥ 0.3; store the input that triggered the correction |
| extractor flags task instructions | "is_learning" criterion drifted | add skipped examples to the reflector gold; GEPA learns the boundary |
| promotion pollutes unrelated programs | threshold 1 or no review | threshold ≥ 2 contexts and human approval |
| transcript too long | no pre-filter | pre-filter first; extract with a cheap model |
