---
name: dspy-book-coding-agents
description: >-
  Optimize a text artifact that is not a DSPy program — a SKILL.md, an
  AGENTS.md, a persona file, a cursor rule — using GEPA's optimize_anything
  entry point, where the candidate is the file's text, the dataset is a handful
  of hand-captured real failures, and the evaluator returns a score plus the
  reason. Covers turning a convention document into a 20-case adversarial
  benchmark with binary verdicts, mixing noise-free mechanical checks with
  judge signal, and reading the regression list the optimizer leaves behind.
when_to_use: >-
  User says "optimize this skill", "improve AGENTS.md", "my CLAUDE.md is not
  working", "test a prompt file", "the agent keeps doing X despite the rules",
  "persona", "system prompt optimization"; a markdown instruction file must
  get measurably better rather than be rewritten by hand.
---

# Optimizing Coding Agents (book chapter 11)

From the O'Reilly companion repo (MIT). This chapter is the one directly
applicable to this pack: a skill file is a text artifact loaded as a system
prompt, and it can be measured and optimized like anything else.

The chapter is one move repeated five times. Learn the move.

## The move: optimize_anything

When the artifact is prose rather than a DSPy program, there is nothing to
compile, so GEPA's standalone entry point takes the file's text as the
candidate:

```python
from gepa.optimize_anything import optimize_anything, GEPAConfig, EngineConfig, ReflectionConfig

def evaluator(candidate: str, example: dict) -> tuple[float, dict]:
    response = TASK_LM(messages=[{"role": "system", "content": candidate},
                                 {"role": "user", "content": example["prompt"]}])[0]
    score, verdict = judge(response, example)
    return score, {"Prompt": example["prompt"], "Response": response,
                   "JudgeReasoning": verdict}          # the reflector's only window into why

result = optimize_anything(
    seed_candidate=open("SKILL.md").read(),
    evaluator=evaluator,
    dataset=CASES,
    objective="Rewrite the skill so ... Output ONLY the SKILL.md text. No preamble, no markdown fences.",
    config=GEPAConfig(engine=EngineConfig(max_metric_calls=80),
                      reflection=ReflectionConfig(reflection_lm=REFLECTION_MODEL)),
)
print(result.best_candidate)      # the new file text
```

Four rules the chapter states and one silent trap:

- **The dataset is 5 to 20 hand-captured real failures**, not synthetic volume. You are encoding what actually goes wrong.
- **The side-info dict is load-bearing.** The reflection LM sees only the score and that dict. Always include the judge's reasoning, or it is optimizing blind.
- **End the objective with an output constraint.** Every notebook closes with "Output ONLY the <file> text. No preamble, no markdown fences."
- **Pareto selection runs per example**, so a candidate cannot win by collapsing to one register and abandoning the rest.
- **The trap:** `optimize_anything` introspects the evaluator's signature, and the second parameter **must be named `example`**. Rename it to `task` or `item` and GEPA silently drops the data, then crashes several layers deep with a missing argument.

`optimize_anything` comes from the **`gepa` package**, not `dspy`. If you do
have signatures, use `dspy.GEPA` instead so each predictor gets its own
instruction slot — the chapter's image-CLI notebook is exactly that
counter-example.

## Making a convention document testable

The most transferable notebook turns an AGENTS.md into a **20-case adversarial
benchmark**. Each case is three fields:

| Field | What it is |
|---|---|
| `task` | a terse, deliberately under-specified request — "that's when the model's defaults leak" |
| `antipattern` | the specific habit the task is designed to elicit |
| `tell` | the concrete observable the judge looks for |

Under-specification is the design principle. A fully specified task tests
whether the model can follow instructions; an under-specified one tests what
it does when the instructions run out, which is what a conventions file is for.

Real cases from the chapter: defensive try/except around a happy-path
`json.load`; `# type: ignore` instead of fixing the Optional; comments that
restate the code; a full Args/Returns/Raises docstring on a one-liner; a new
`utils.py` when a fitting module exists; class-wrapping a single function; a
factory for one implementation; a back-compat shim for code with no external
users; reformatting unrelated code inside a bugfix diff; re-implementing an
existing helper; re-wrapping an exception and losing the traceback; a test that
asserts only on the mock; a TODO while claiming the work is done; a
self-congratulatory checkmark summary.

**The verdict is binary.** The chapter's reason: *"carried over from running
tests to checking conventions — binary, no Likert scale to drift on."* An
unparseable verdict raises rather than scoring zero, so a broken judge cannot
quietly look like a failing candidate.

## Mix noise-free signal with judge signal

The landing-page notebook weights its metric deliberately:

```python
score = 0.5 * mechanical + 0.25 * visual_win + 0.25 * voice_win
```

`mechanical` is the mean of five checks with no model in the loop: the page
renders, every brand colour appears in the CSS, the brand font is referenced,
nothing overflows at 375px, the section count matches the brief. The stated
rationale is worth memorizing: *"noise-free signal is worth more per unit than
judge signal."*

The two judges are **pairwise against a gold artifact with randomized
position**, which removes the position bias a single-sample judge carries.

Whenever part of your quality bar can be checked deterministically, check it
deterministically and give it the larger weight.

## Read the regressions

```python
regressed = [c['antipattern'] for c, b, o in zip(CASES, baseline_scores, optimized_scores) if b > o]
```

GEPA drops rules that did not fire during the evaluation. That is correct
behaviour for the benchmark and wrong for your repository, because your file
also encodes rules no case exercised. So: measure the baseline before
optimizing, re-score every case after, print the regression list, and hand-restore
what the optimizer discarded. Then diff against your current file, because the
optimizer never knew about the team rules that were not in the benchmark.

Keep the old artifact. The chapter's line: *"the artifact is the only way to
diagnose regressions."*

## Discovering a skill instead of writing one

Point an RLM at a directory of past agent transcripts, summarize each session,
cluster the summaries, and write a SKILL.md for the largest cluster — skipping
groups smaller than two sessions. The host reads the files and passes the
**texts** as an input value, because the sandboxed interpreter cannot read host
files. Output is a candidate for human review before install, which then feeds
the optimization loop above.

## Honest expectations

The chapter's own numbers are modest and it says so. One reported persona run
reached 0.962 composite rubric adherence at 50 metric calls in about 19
minutes. The landing-page loop used 200 metric calls, roughly 30 minutes.
These are small budgets producing real but bounded gains, on benchmarks the
authors built. Measure your own baseline; do not import theirs.

## Anti-patterns

- Naming the evaluator's second parameter anything but `example`. Silent data loss.
- Writing synthetic cases instead of capturing real failures; you will optimize against imagined problems.
- Returning a bare score with no side info, leaving the reflector nothing to reason from.
- Using a Likert scale where a binary verdict works; the scale drifts between runs.
- Shipping the optimized file without reading the regression list.
- Deleting the previous artifact, leaving no way to diagnose what changed.
- Reaching for `optimize_anything` when you have a DSPy program; use `dspy.GEPA` and get per-predictor instruction slots.
- Copying the notebooks' model identifiers. They are forward-dated placeholders that do not resolve.

## Where to go next

- GEPA on an actual DSPy program → `dspy-gepa-optimizer`
- Whether GEPA is even the right optimizer → `dspy-optimizer-selection`
- Metric shapes and the judge-calibration bar → `dspy-book-metrics`
- Turning session corrections into training signal → `dspy-reflect-loop`
- Full reference (all five notebooks, budgets, the RLM discovery signature) → [reference.md](reference.md)
- Runnable, dry-run capable → [example_artifact_optimizer.py](example_artifact_optimizer.py)
