---
name: dspy-reflect-loop
description: Turn human corrections, approvals and suggestions from a session into durable learning for DSPy programs — extract typed learning signals (HIGH correction, MEDIUM approval, LOW observation) with a deterministic pre-filter plus a Signature, record them in a fingerprinted ledger, convert them into dspy.Example gold plus metric feedback so GEPA rewrites the instructions instead of hand-editing prompts, promote learnings seen in several projects to the shared instruction layer, and meta-learn from accept/modify/skip decisions. Dry-run first, backups, explicit approval.
when_to_use: >-
  User says "reflect", "learn from this", "correct once never again", "remember
  that", "why did it repeat the mistake", or corrects the program repeatedly;
  after a session with corrections; when a skill or program should be updated
  from feedback; when building the human-feedback source of a self-optimizing loop.
---

# DSPy Reflect Loop (3.3.x)

Port of `claude-reflect-system` (Reflect v1.3, "correct once, never again":
signal extraction → review → safe skill update → git commit, plus a
cross-repo learning ledger, promotion to global rules, and meta-learning on
the reviewer's decisions) into DSPy. The shift: a correction is not a
markdown section to append — it is a **gold example plus feedback**. GEPA
consumes exactly that, so the program's instructions improve through
optimization, and the ledger/meta-learning become metrics on the reflector
itself. This skill is the human-feedback source of the self-optimizing stack
(`dspy-reflect-loop` → GEPA → `dspy-deep-refine` → `dspy-rlm-workflow`).

## Reflect step → DSPy construct

| Reflect (original) | DSPy construct | Deterministic |
|---|---|---|
| regex signal patterns (EN/DE) | `prefilter(message)` — cheap candidate detector | yes |
| `claude -p` semantic detector | `ExtractLearningSignal` signature → `LearningSignal` | — |
| HIGH / MEDIUM / LOW | `confidence: Literal["HIGH","MEDIUM","LOW"]`, mapped from `kind` | yes |
| "Critical Corrections / Best Practices / Considerations" sections | `Learning` → `dspy.Example(...)` + feedback string; instruction text is produced by GEPA, never templated | yes (conversion) |
| interactive review A/M/S/Q | `review()` returning decisions; nothing applied without them | human |
| backups + YAML validation + git commit | `apply()` writes artifacts (`program.json`, gold `.jsonl`) with backup and validation; commit is the user's | yes |
| learning ledger (SQLite, fingerprint, repo ids, threshold 2) | `Ledger` — fingerprint = sha256(normalized text)[:16], `contexts[]`, `status` | yes |
| scope analyzer (project vs global) | `ScopeHint` from the signature + ledger count | mixed |
| promotion to `~/.claude/CLAUDE.md` | promotion to the *shared* instruction set / shared demos of the program family | yes (gate) |
| meta-learning (accept/modify/skip per pattern) | `reflector_metric` — the reviewer's decisions are gold for the extractor; GEPA on the extractor | yes |

## Canonical program

```python
import dspy, hashlib, re
from pydantic import BaseModel, Field
from typing import Literal

class LearningSignal(BaseModel):
    is_learning: bool
    kind: Literal["correction", "approval", "observation", "explicit"] | None = None
    confidence: Literal["HIGH", "MEDIUM", "LOW"] | None = None
    old_behavior: str = ""            # what the program did (for corrections)
    new_behavior: str = ""            # what it should do
    learning: str = ""                # one actionable sentence, source language kept
    scope_hint: Literal["program", "project", "global"] = "program"
    reasoning: str = ""

class ExtractLearningSignal(dspy.Signature):
    """Decide whether a user message contains a reusable learning for the program
    (not a one-off task instruction). Corrections ("use X instead of Y", "never …")
    are HIGH; approvals of a specific approach are MEDIUM; suggestions and
    "have you considered …" are LOW. Works in any language; keep the learning in
    the user's language. Extract old vs new behaviour when both are stated."""
    message: str = dspy.InputField()
    prior_assistant_turn: str = dspy.InputField(desc="what the program/assistant did right before")
    program_name: str = dspy.InputField()
    signal: LearningSignal = dspy.OutputField()

PREFILTER = re.compile(r"(?i)\b(instead of|statt|don't|do not|never|niemals|always|immer|actually|"
                       r"nein,|no,|use \w+ instead|verwende|benutze|remember:|merk dir|"
                       r"perfect|exactly|genau so|works? (perfectly|great)|have you considered|why not)\b")

class ReflectLoop(dspy.Module):
    def __init__(self):
        super().__init__()
        self.extract = dspy.ChainOfThought(ExtractLearningSignal)

    def forward(self, transcript: list[dict], program_name: str) -> dspy.Prediction:
        signals = []
        for i, turn in enumerate(transcript):
            if turn["role"] != "user" or len(turn["content"]) < 10 or not PREFILTER.search(turn["content"]):
                continue                                   # deterministic pre-filter: cheap, high recall
            prior = next((t["content"] for t in reversed(transcript[:i]) if t["role"] == "assistant"), "")
            sig = self.extract(message=turn["content"], prior_assistant_turn=prior,
                               program_name=program_name).signal
            if sig.is_learning:
                signals.append(sig)
        return dspy.Prediction(signals=signals)
```

## From signal to GEPA signal

A HIGH correction about a program `P` becomes a gold example for `P` and a
feedback sentence the metric emits whenever `P` repeats the old behaviour:

```python
def learning_to_example(sig: LearningSignal, failing_input: dict) -> dspy.Example:
    return dspy.Example(**failing_input, expected_behavior=sig.new_behavior,
                        forbidden_behavior=sig.old_behavior,
                        feedback=f"User corrected this: {sig.learning}").with_inputs(*failing_input)

def corrections_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    text = str(pred.toDict()).lower()
    satisfied = not gold.expected_behavior or gold.expected_behavior.lower() in text
    violated = bool(gold.forbidden_behavior) and gold.forbidden_behavior.lower() in text and not satisfied
    score = 0.0 if violated else (1.0 if satisfied else 0.5)   # old form present without the corrected one
    fb = gold.feedback if (violated or not satisfied) else "Follows the learned correction."
    return dspy.Prediction(score=score, feedback=fb)
```

Fold `corrections_metric` into the program's real metric (weighted), append the
examples to its trainset, and run `dspy.GEPA(auto="light")`. The optimizer
rewrites `P`'s instructions so the correction sticks — the "Critical
Corrections" section of the original, written by reflection rather than by a
template. MEDIUM approvals become positive demos (`BootstrapFewShot` /
`BetterTogether`'s bootstrap stage); LOW observations are logged only.

## Ledger, scope and promotion

```python
class Ledger:                                   # JSONL or SQLite; deterministic
    def record(self, sig: LearningSignal, context_id: str) -> str: ...   # fingerprint
    def eligible_for_promotion(self, threshold: int = 2) -> list[Learning]: ...
```

- `fingerprint = sha256(" ".join(learning.lower().split()))[:16]`; the same
  learning from another project/program increments `contexts`.
- A learning seen in `≥ threshold` contexts is a promotion candidate: it moves
  from one program's trainset to the **shared** gold set (or the family-level
  instruction prefix) so every program in the family is optimized against it.
- Promotion is a gate: `preview` → user approval → `apply` with a backup of the
  target file; never automatic.

## Meta-learning = a metric on the reflector

Every review decision (`accept` / `modify` / `skip`) is logged with the signal's
kind and confidence. That log is gold for `ReflectLoop` itself:

```python
def reflector_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    got = {(s.learning.lower(), s.confidence) for s in pred.signals}
    want = {(s["learning"].lower(), s["confidence"]) for s in gold.accepted}
    noise = [s for s in pred.signals if s.learning.lower() in gold.skipped]
    tp = len(got & want); fn = len(want - got)
    score = (tp / max(1, len(want))) * (1 - len(noise) / max(1, len(pred.signals)))
    fb = []
    if fn: fb.append(f"Missed {fn} accepted learning(s), e.g. {next(iter(want - got))[0]!r}.")
    if noise: fb.append(f"{len(noise)} proposal(s) the user skipped before — not reusable learnings.")
    return dspy.Prediction(score=score, feedback=" ".join(fb) or "All accepted learnings found, no skipped noise.")
```

Patterns the user keeps skipping lose weight, patterns they keep accepting gain
it — the original's `--use-meta` confidence adjustment, done by GEPA on the
extractor's instruction instead of by hand-tuned deltas.

## Safety (kept from the original)

- Dry-run by default: `ReflectLoop` returns signals; `review()` shows them;
  `apply()` runs only after the user's explicit choice per signal.
- Backups before every write, validation after (JSON for gold sets, schema for
  artifacts), rollback on error.
- Never auto-commit or push; print the commit command.
- A lock/timestamp prevents double-reflecting the same transcript; auto-mode
  (session-end hook) is opt-in and runs in the background.
- Learnings stay in the user's language; nothing leaves the machine.

## Anti-patterns

- Appending correction text to the instruction docstring by hand — GEPA will overwrite or contradict it; give it the example + feedback instead.
- Treating every "yes" as an approval — the pre-filter is high recall, the Signature decides, the reviewer confirms.
- Promoting on first sight — threshold ≥ 2 contexts, then a human.
- Skipping the ledger — without fingerprints the same learning is re-proposed every session.
- Running the extractor over the whole transcript with the task LM — pre-filter first, extract with a cheap model.

## Where to go next

- Rich metrics and gold sets → `dspy-evaluation-harness`
- Optimizing with the learned examples → `dspy-gepa-optimizer`
- Refining the knowledge base from unanswerable queries → `dspy-deep-refine`
- Verified execution of context-heavy work → `dspy-rlm-workflow`
- Full reference (models, ledger schema, review/apply contract, metrics) → [reference.md](reference.md)
- Runnable example → [example_reflect_loop.py](example_reflect_loop.py)
