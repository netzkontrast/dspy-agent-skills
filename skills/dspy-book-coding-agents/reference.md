# Optimizing Coding Agents — Reference

Source: chapter 11 of `context-engineering-dspy-book` (MIT). Pins
`dspy==3.3.0`, `gepa==0.1.1`. Five notebooks, all applying one recipe.

## Which optimizer for which artifact

| The artifact | Tool | Why |
|---|---|---|
| a markdown file loaded as a system prompt | `gepa.optimize_anything` | there is no program to compile |
| a DSPy program with signatures | `dspy.GEPA(...).compile(...)` | each predictor gets its own instruction slot |

Getting this backwards is the chapter's implicit lesson: the image-CLI notebook
is a DSPy program and correctly uses `dspy.GEPA`; the other four are text files
and correctly use `optimize_anything`.

## `optimize_anything` contract

```python
from gepa.optimize_anything import optimize_anything, GEPAConfig, EngineConfig, ReflectionConfig

def evaluator(candidate: str, example: dict) -> tuple[float, dict]: ...

optimize_anything(seed_candidate=<file text>, evaluator=evaluator, dataset=CASES,
                  objective=<rewrite instruction>,
                  config=GEPAConfig(engine=EngineConfig(max_metric_calls=80),
                                    reflection=ReflectionConfig(reflection_lm=REFLECTION_MODEL)))
# -> result.best_candidate  (str)
```

| Element | Requirement |
|---|---|
| `seed_candidate` | the current file's text, read from disk |
| `evaluator` | second parameter **must be named `example`** — signature is introspected |
| return value | `(score: float, side_info: dict)`; side info is the reflector's only window |
| `dataset` | 5–20 hand-captured real failures |
| `objective` | the rewrite instruction, ending with an output-format constraint |
| `EngineConfig` | `max_metric_calls`, plus `run_dir`, `track_best_outputs`, `cache_evaluation` |

Every notebook ships a `max_metric_calls=3` smoke cell alongside the
full-budget cell. Do the same: the smoke run catches the `example` trap and the
objective's format constraint before you spend a budget.

## The five notebooks

### landing-page-skill-optimizer

Optimizes `~/.claude/skills/landing-page/SKILL.md`. Rollout runs the candidate
as the system prompt of a page-generation model and returns HTML.

Metric: `0.5 * mechanical + 0.25 * visual_win + 0.25 * voice_win`, where
mechanical is the mean of five binary checks (renders under Playwright; all
brand hex colours present in CSS; brand font referenced; no horizontal overflow
at 375px; section count matches the brief). Both judges compare pairwise
against gold HTML with randomized position:

```python
swap = random.random() < 0.5
candidate_won = (verdict == ("A" if not swap else "B"))
```

Budget: 200 metric calls, roughly 30 minutes, about $0.10 per rollout. Dataset
cases carry `brief`, `brand_guide`, `expected`, `failure_mode`, `notes`,
`gold_html`.

### test-agents-md

Turns an AGENTS.md into 20 adversarial cases of `task` / `antipattern` / `tell`.
Two LM calls per rollout: the agent writes a plan and code under the candidate
file as system prompt, then a judge sees the antipattern and the tell and
returns `VERDICT: <PASS|FAIL>`. An unparseable verdict **raises**.

Seed is deliberately weak: `"You are a careful engineering assistant. Be
helpful and concise."` Baseline is measured before optimizing, all 20 cases are
re-scored after, and the delta is printed with an explicit regression list.

Budget: 80 metric calls in the notebook, 200 suggested for real use.

### clawsona-dspy

Optimizes a persona file (`SOUL.md`) against an eight-rule rubric. Dataset is
10 deliberately heterogeneous prompts — a debugging question, a production
page, a Slack draft, a technology choice, a farewell post, an outage apology.
The heterogeneity is the point: Pareto keeps a candidate that is strong on
technical prompts and weak on social ones, so the persona cannot collapse into
a single register.

Judge emits PASS/FAIL per rule then `SCORE: <0-1>`, parsed with a regex;
unparseable raises. Reported result: 0.962 composite at 50 metric calls, about
19 minutes.

### image-cli-optimizer

The counter-example. The artifact is a DSPy program, so it uses `dspy.GEPA`:

```python
optimizer = dspy.GEPA(metric=cli_metric, reflection_lm=..., max_metric_calls=600,
                      reflection_minibatch_size=8, candidate_selection_strategy="pareto",
                      num_threads=6, track_stats=True, log_dir="./gepa_logs")
optimized = optimizer.compile(student=ImagePopulator(...), trainset=trainset, valset=valset)
```

Three signatures means three separate instruction slots. The metric averages
the module's own per-slot visual judge scores and threads the failure mode into
the feedback string. Data is about 10 real failures plus 50 synthesized, split
50 train / 10 val — expansion is justified there because CLI rollouts are cheap
and the metric scores generated images, so no gold images are needed.

Deployment note worth copying: optimize against the **same** cheap models you
ship, then `optimized.save(...)` and
`production.load_state(optimized.dump_state())`.

### skill-discovery-rlm

```python
class DiscoverSkill(dspy.Signature):
    """Read every session transcript in `session_texts` ... Group similar summaries.
    For the largest group, write a SKILL.md with YAML frontmatter (name, description)
    and a body of concrete rules drawn from how the assistant actually responded.
    Skip groups smaller than 2 sessions. Return only the SKILL.md text."""
    session_texts: list[str] = dspy.InputField()
    skill_md: str = dspy.OutputField()

rlm = dspy.RLM(DiscoverSkill, max_iters=8, max_llm_calls=10, sub_lm=...)
```

The host reads transcripts and passes the texts as an input value, because the
default Deno interpreter cannot read host files. Output goes to disk for human
review before install, then into the optimization loop above. Requires Deno.

## The doc-compaction artifact

`chapter11/assets/compacted-dspy-docs.md` is the entire DSPy documentation site
flattened into one file of roughly 163 KB — each chunk titled with its full
breadcrumb as an H1 and repeated as an H2, so any chunk retrieved in isolation
still carries its path.

No notebook loads it. It is a standalone demonstration of a technique worth
stealing: compact a whole doc site into one grep-able, breadcrumb-titled file
so an agent loads or searches one artifact instead of crawling a site.

Its content is slightly behind the repo's own pins — it still documents
`dspy.Assert` and `dspy.Suggest`, which `dspy.Refine` superseded. If you hand a
compacted doc to an agent as ground truth, date it and re-verify it.

## Applying this to a skill pack

The chapter's method maps onto a pack like this one:

1. Capture real failures: cases where an agent loaded a skill and still did the wrong thing.
2. Write each as `task` / `antipattern` / `tell`, under-specified on purpose.
3. Score binary with a judge that raises on an unparseable verdict.
4. Add every deterministic check you can — a skill's frontmatter, its length, its required sections — and weight them above the judge.
5. Optimize, re-score, read the regressions, restore what the benchmark did not cover.

Steps 1, 2 and 4 are worth doing even if you never run the optimizer: a
convention file with a benchmark attached is falsifiable, and one without is
an opinion.

## Version notes

`optimize_anything` is `gepa`, not `dspy`. All model identifiers in these
notebooks are the book's forward-dated placeholders and do not resolve;
substitute real ones before running anything.
