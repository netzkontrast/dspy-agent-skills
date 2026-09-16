# Usage Guide

## The thirty-two skills at a glance

| Skill | Invoke when | Depends on |
|---|---|---|
| `dspy-fundamentals` | Any new DSPy code | — |
| `dspy-evaluation-harness` | Writing metrics, splitting devset/valset, debugging eval | `dspy-fundamentals` |
| `dspy-gepa-optimizer` | Optimizing/compiling a DSPy program | `dspy-evaluation-harness` |
| `dspy-rlm-module` | Context >100k tokens, codebase/doc exploration | `dspy-fundamentals` |
| `dspy-rlm-workflow` | Context-heavy multi-step work that must be verified; runtime iteration with `dspy.Refine` | `dspy-evaluation-harness`, `dspy-rlm-module` |
| `dspy-deep-refine` | A retrieval base that cannot answer questions; reviewed graph/wiki edits | `dspy-evaluation-harness` |
| `dspy-reflect-loop` | Learning from user corrections; ledger, promotion, meta-learning | `dspy-evaluation-harness`, `dspy-gepa-optimizer` |
| `dspy-clarify` | A claim, task or query must be made precise before it changes authority (promotion, decomposition, refinement) | `dspy-evaluation-harness` |
| `dspy-tetraframe` | A contested or hard-to-reverse decision must be assessed before it is recorded (wiki merge/supersede, promotion conflict, design/storyform choice) | `dspy-evaluation-harness`, `dspy-clarify` |
| `dspy-autodialectics` | A program that drifts, fakes completion or self-certifies; anti-slop gate and GEPA slop metric; champion/challenger promotion | `dspy-evaluation-harness`, `dspy-gepa-optimizer` |
| `dspy-wiki-compile` | Raw documents must become a cited, maintained wiki; a reviewed page must not be overwritten; the same concept in several sources | `dspy-evaluation-harness`, `dspy-clarify` |
| `dspy-adversarial-review` | A page, claim or draft is about to gain authority and needs a second model's objections; refine to a target score | `dspy-evaluation-harness`, `dspy-autodialectics` |
| `dspy-local-runtime` | No API key, `claude` on PATH; DSPy and GEPA on the Claude Code subscription | `dspy-fundamentals` |
| `dspy-optimizer-selection` | Choosing among the optimizer families before compiling; a compile run is too slow or too expensive | `dspy-evaluation-harness` |
| `dspy-retrieval` | RAG, embeddings, vector search, multi-hop; retrieval quality must be separable from answer quality | `dspy-evaluation-harness` |
| `dspy-production` | Deploying a compiled artifact; cost, latency, tracing, streaming, async, batch | `dspy-evaluation-harness` |
| `dspy-refrag` | Evaluating or vendoring REFRAG; fragment selection over near-duplicate passages | `dspy-retrieval` |
| `dspy-rlm-hooks` | Instrumenting an RLM run, or overlapping tool latency with generation | `dspy-rlm-module` |
| `dspy-drg-kg` | Turning documents into a typed, queryable knowledge graph | `dspy-retrieval` |
| `dspy-tara-rag` | Self-corrective retrieval; scoring context on four dimensions | `dspy-retrieval`, `dspy-evaluation-harness` |
| `dspy-tools-cli` | Managing DSPy programs from the shell rather than inline | `dspy-optimizer-selection` |
| `dspy-context-engineering-book` | Finding a worked notebook, or the right chapter skill | — |
| `dspy-book-eight-steps` | The eight-step build order, and optimizing the judge before the task | book chapter 1-3 |
| `dspy-book-datasets` | Where trainsets come from: conversion, seeded splits, difficulty tiers, synthetic data | book chapter 4 |
| `dspy-book-metrics` | Eleven metric recipes, and when an LLM judge may be trusted | book chapter 5 |
| `dspy-book-optimizers` | Twelve optimizers measured on one task, with cost and wall time | book chapter 6 |
| `dspy-book-modules` | Adapters, multimodal inputs, code execution, parallel and majority voting | book chapter 7 |
| `dspy-book-agents` | MCP tools, conversation memory, and multi-hop budget control | book chapter 8 |
| `dspy-book-use-cases` | Seven complete applications as a pattern library | book chapter 9 |
| `dspy-book-production` | Serving a compiled program, optimizable guardrails, MLflow tracing | book chapter 10 |
| `dspy-book-coding-agents` | Optimizing a SKILL.md or AGENTS.md as a text artifact | book chapter 11 |


| `dspy-advanced-workflow` | Full greenfield DSPy build and the self-optimizing loop | all others |

Claude Code / Codex auto-select skills by matching the `description` field. You don't need to invoke them manually in most cases.

## Typical conversation shapes

### Greenfield pipeline

> "Build a DSPy sentiment-classification pipeline on this CSV, optimize it, and save the artifact."

The agent pulls `dspy-advanced-workflow`, which chains the core skills in order: fundamentals (Signature/Module) → evaluation-harness (metric + Evaluate) → gepa-optimizer (compile) → fundamentals (save).

### Debugging an optimizer

> "My GEPA run plateaus after round 2 — why?"

The agent loads `dspy-gepa-optimizer` and `dspy-evaluation-harness` and walks through the top failure modes: thin metric feedback, weak reflection_lm, train=val overlap, insufficient budget.

### Long-document QA

> "Summarize every error class in this 3M-token log."

The agent loads `dspy-rlm-module` and builds an RLM-backed pipeline with a cheap sub-LM.

### Context-heavy work with verification

> "Refactor auth across all services and verify it before you hand it over."

The agent loads `dspy-rlm-workflow`: distill the context, decompose into a dependency-ordered plan, solve, synthesize with explicit contradictions, verify through the three-tier cascade, and iterate with `dspy.Refine`.

### A knowledge base that keeps failing

> "The wiki can't answer where Juna lives — fix the base, not the prompt."

The agent loads `dspy-deep-refine`: judge → widen retrieval → abduce → propose ≤10 edits → grade evidence HIGH/MEDIUM/LOW → stop for approval.

### Learning from corrections

> "I told you twice to use uv, not pip. Make it stick."

The agent loads `dspy-reflect-loop`: the correction becomes a gold example plus metric feedback, GEPA rewrites the instructions, the ledger prevents re-proposing it.

### Precision before promotion

> "This research claim is about to become canon — make sure it says exactly what the source says, and ask me about anything open."

The agent loads `dspy-clarify`: explicit scope only where the source states it, names bound to the glossary, assumptions labelled, every remaining ambiguity as a question; verdict `clear` / `needs-author` / `not-promotable`.

### Assessing a contested decision

> "Should the new research page replace the old concept page, or do we keep both? Steelman it before I decide."

The agent loads `dspy-tetraframe`: the seed is distilled to one predicate, four corners are generated in isolation (P, not-P, both under a typed split, neither with a replacement predicate), contradictions and evidence discriminators are mapped, a non-averaging P* is produced with `dspy.BestOfN`, and the verification table is shown; the human decides and the decision record cites the run.

### Keeping a program honest

> "It keeps saying 'done' with no tests. Gate it."

The agent loads `dspy-autodialectics`, Tier 0 first: compile the contract, run the deterministic slop score and gate on the existing output (no LM calls). Only if the program is rebuilt does it add the typed thesis → antithesis → synthesis plan and the independent verifier; champion/challenger evolution needs a benchmark with canaries.

### Compiling a wiki from sources

> "Ingest these 25 research notes into the wiki without touching the pages I already reviewed."

The agent loads `dspy-wiki-compile`: every source is triaged and extracted with line-range citations, concepts are merged across the batch, each existing page gets a flag / update / create decision with active-page protection, and the knowledge diff is printed before anything is written.

### An independent review

> "Before this page goes to reviewed, have a different model tear it apart."

The agent loads `dspy-adversarial-review`: a reviewer LM that is asserted to differ from the writer quotes every overstated claim with the evidence it would need, lists unsupported claims, scores 1–10, and the demotion rule decides whether the page stays or becomes contested.

### Running without an API key

> "I only have Claude Code here, no key. Can GEPA still run?"

The agent loads `dspy-local-runtime`: a `BaseLM` over `claude -p`, backend selection, the kwargs the CLI strips, and a call budget for evaluation and GEPA on the subscription.

### Explicit invocation

You can also force a skill:

- **Claude Code**: `/dspy-gepa-optimizer` (if `user-invocable` is true, which it is by default).
- **Codex**: `$dspy-gepa-optimizer`.

## Minimal happy-path code sample

```python
import dspy

dspy.configure(lm=dspy.LM("openai/gpt-4o"), track_usage=True)

class QA(dspy.Signature):
    """Answer concisely."""
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

program = dspy.ChainOfThought(QA)

def rich_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    correct = pred.answer.strip().lower() == gold.answer.strip().lower()
    return dspy.Prediction(
        score=1.0 if correct else 0.0,
        feedback="Correct." if correct else f"Expected {gold.answer!r}, got {pred.answer!r}.",
    )

trainset = [dspy.Example(question="2+2?", answer="4").with_inputs("question"), ...]
valset   = [dspy.Example(question="5*6?", answer="30").with_inputs("question"), ...]

evaluator = dspy.Evaluate(devset=valset, metric=rich_metric, num_threads=4,
                          provide_traceback=True)
print("Baseline:", evaluator(program).score)

optimizer = dspy.GEPA(
    metric=rich_metric, auto="medium",
    reflection_lm=dspy.LM("openai/gpt-5", temperature=1.0, max_tokens=32000),
    track_stats=True, log_dir="./gepa_logs",
)
optimized = optimizer.compile(student=program, trainset=trainset, valset=valset)
print("Optimized:", evaluator(optimized).score)

optimized.save("program.json", save_program=False)
```

On DSPy 3.3.x, module calls warn by default when you pass extra input fields or values that don't match the signature's declared types. Treat those warnings as a callsite bug first; only disable them with `dspy.configure(warn_on_type_mismatch=False)` when you intentionally pass pre-serialized values.

## Running the bundled example scripts

Each skill folder contains a runnable `example_*.py` with a `--dry-run` flag that verifies construction without calling an LM:

```bash
cd skills/dspy-fundamentals
uv run python example_qa.py --dry-run

cd ../dspy-evaluation-harness
uv run python example_metric.py --dry-run

cd ../dspy-gepa-optimizer
uv run python example_gepa.py --dry-run
uv run python example_bettertogether.py --dry-run

cd ../dspy-rlm-module
uv run python example_rlm.py --dry-run

cd ../dspy-rlm-workflow
uv run python example_rlm_workflow.py --dry-run

cd ../dspy-deep-refine
uv run python example_deep_refine.py --dry-run

cd ../dspy-reflect-loop
uv run python example_reflect_loop.py --dry-run

cd ../dspy-clarify
uv run python example_clarify.py --dry-run

cd ../dspy-tetraframe
uv run python example_tetraframe.py --dry-run

cd ../dspy-autodialectics
uv run python example_autodialectics.py --dry-run

cd ../dspy-wiki-compile
uv run python example_wiki_compile.py --dry-run

cd ../dspy-adversarial-review
uv run python example_adversarial_review.py --dry-run

cd ../dspy-local-runtime
uv run python example_local_runtime.py --dry-run
cd ../dspy-optimizer-selection
uv run python example_optimizer_selection.py --dry-run

cd ../dspy-retrieval
uv run python example_retrieval.py --dry-run

cd ../dspy-production
uv run python example_production.py --dry-run

cd ../dspy-refrag
uv run python example_refrag.py --dry-run

cd ../dspy-rlm-hooks
uv run python example_rlm_hooks.py --dry-run

cd ../dspy-drg-kg
uv run python example_drg_kg.py --dry-run

cd ../dspy-tara-rag
uv run python example_tara.py --dry-run

cd ../dspy-tools-cli
uv run python example_tools_cli.py --dry-run

cd ../dspy-context-engineering-book
uv run python example_book_map.py --dry-run

cd ../dspy-book-eight-steps
uv run python example_eight_steps.py --dry-run

cd ../dspy-book-datasets
uv run python example_dataset_builder.py --dry-run

cd ../dspy-book-metrics
uv run python example_metric_recipes.py --dry-run

cd ../dspy-book-optimizers
uv run python example_optimizer_results.py --dry-run

cd ../dspy-book-modules
uv run python example_module_choice.py --dry-run

cd ../dspy-book-agents
uv run python example_agent_budget.py --dry-run

cd ../dspy-book-use-cases
uv run python example_use_case_router.py --dry-run

cd ../dspy-book-production
uv run python example_serving_checks.py --dry-run

cd ../dspy-book-coding-agents
uv run python example_artifact_optimizer.py --dry-run

cd ../dspy-advanced-workflow
uv run python example_pipeline.py --dry-run
```

Live runs require `OPENAI_API_KEY` (or equivalent for the chosen `--model`).

## Maintainer API-surface check

Before changing skill guidance for a new DSPy release, verify the live wheel instead of inferring from prose docs:

```bash
env -u UV_EXCLUDE_NEWER uv run --with dspy==3.3.1 python scripts/check_dspy_surface.py
```

## Getting help

- DSPy docs: https://dspy.ai/
- Skill sources: `skills/<name>/reference.md` in this repo
- Issues: https://github.com/intertwine/dspy-agent-skills/issues
