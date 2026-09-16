---
name: dspy-local-runtime
description: >-
  Run DSPy programs, evaluations and GEPA through the local Claude Code CLI
  on a subscription, with no API key — a dspy.BaseLM subclass that turns each
  request into one `claude -p --output-format json` process, backend
  selection between an API LM and the CLI, and the exact kwargs the CLI
  cannot honour (temperature, max_tokens, rollout_id, n > 1, cache) with what
  each program loses without them. Pattern from Hmbown/dspy-local. Use when
  a machine has the claude CLI but no key, or when optimisation must stay on
  the subscription budget.
when_to_use: >-
  User says "no API key", "run it through claude code", "use my
  subscription", "dspy-local", "claude -p", "can GEPA run without a key";
  `ANTHROPIC_API_KEY` is absent but `claude` is on PATH; a DSPy program must
  run inside a Claude Code session or a CI box that only has the CLI.
---

# DSPy Local Runtime (3.3.x)

`dspy.LM` speaks to providers through LiteLLM and needs a key. The Claude
Code CLI already holds a logged-in session, and `claude -p` answers a prompt
non-interactively. [Hmbown/dspy-local](https://github.com/Hmbown/dspy-local)
(MIT) bridges the two with a `BaseLM` subclass. This skill teaches the bridge
as a pattern: what to implement, what the CLI refuses, and how to budget.

## The bridge in five decisions

| Decision | Why |
|---|---|
| subclass `dspy.BaseLM`, override `forward` | DSPy's adapters, `Predict`, `ChainOfThought`, `Evaluate`, `GEPA` all speak `BaseLM`; nothing else changes |
| one process per request: `claude -p --output-format json --permission-mode plan --no-session-persistence [--model alias] [--system-prompt …]` | `plan` mode means text in, text out, no tools; JSON output carries `result`, `usage`, `is_error` |
| `cache=False`, always | the CLI has no deterministic sampling; a cached answer would be a lie about reproducibility |
| strip `temperature`, `max_tokens`, `rollout_id` in `__init__` and `copy()` | the CLI does not expose them; `BestOfN`, `Refine` and TetraFrame call `lm.copy(rollout_id=…, temperature=…)` and must not crash |
| `n > 1` raises | one completion per call; candidates come from `dspy.BestOfN`, which loops |

```python
class ClaudeLM(dspy.BaseLM):
    def __init__(self, model="claude/default", *, permission_mode="plan", timeout_seconds=120, cache=False, **kwargs):
        if cache:
            raise ValueError("ClaudeLM cannot cache")
        super().__init__(model=model, model_type="chat", temperature=None, max_tokens=None, cache=False, **kwargs)
        self.kwargs = {k: v for k, v in self.kwargs.items() if k not in STRIPPED_KWARGS}
        self.alias = model.split("/", 1)[1]

    def copy(self, **kwargs):
        for key in STRIPPED_KWARGS:
            kwargs.pop(key, None)
        return super().copy(**kwargs)

    def forward(self, prompt=None, messages=None, **kwargs):
        system, user = build_prompt(prompt, messages)
        proc = subprocess.run(build_command(self.alias, system, self.permission_mode), input=user,
                              capture_output=True, text=True, timeout=self.timeout_seconds)
        text, usage = parse_result(proc.stdout)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))], usage=usage, model=self.model)
```

Full teaching-sized implementation with a `--probe`:
[example_local_runtime.py](example_local_runtime.py). For production, vendor
the upstream file: it also isolates `HOME` (only credentials are copied in),
validates every kwarg with a named error, and records usage on DSPy's tracker.

## Backend selection

```python
def select_backend(env, which=shutil.which) -> str:
    chosen = env.get("DSPY_LOCAL_BACKEND", "auto")          # api | claude-cli | auto
    if chosen != "auto":
        return chosen
    if env.get("ANTHROPIC_API_KEY") or env.get("OPENAI_API_KEY"):
        return "api"
    return "claude-cli" if which("claude") else "api"       # no key, no CLI → fail loudly at first call
```

Role models differ per backend: `anthropic/claude-haiku-4-5` for the API,
`claude/haiku` for the CLI. Keep two sets of env overrides and warn when one
set is present while the other backend is active.

## What each program loses, and the budget

| Program | Without temperature / rollout_id | Sequential CLI cost |
|---|---|---|
| `dspy.Predict` / `ChainOfThought` | nothing | one call, ~5–10 s |
| `dspy.BestOfN`, `dspy.Refine` | candidates differ only by the instruction; still loops correctly | N calls |
| `dspy-tetraframe` corners | diversity rests on the four contract docstrings; read `branch_independence` strictly | ~8 calls + retries |
| `dspy.Evaluate` | use `num_threads=1`; no cache hits between runs | one call per example |
| `dspy.GEPA(auto="light")` | reflection runs at the CLI's default temperature | hundreds of calls; run in the background, set `max_metric_calls` |

## GEPA through the CLI

```python
dspy.configure(lm=ClaudeLM("claude/sonnet"), track_usage=True)
reflection_lm = ClaudeLM("claude/opus")
optimizer = dspy.GEPA(metric=my_metric, reflection_lm=reflection_lm, auto="light", max_metric_calls=300)
optimized = optimizer.compile(program, trainset=train, valset=val)
optimized.save("artifacts/program.json", save_program=False)     # artifacts are backend-independent
```

Saved artifacts are instructions and demos; they load under an API LM later
without change.

## Rules

1. `cache=False`, and never a persistent cache keyed on the prompt alone.
2. Strip, never error, on `rollout_id` and `temperature` in `copy()`.
3. Probe once per machine (`--probe`) before a long run; a stale login fails
   in the first call, not after an hour.
4. Budget in calls, not tokens; the CLI's latency is the constraint.
5. Keep the API path working; the CLI is a fallback, not a lock-in.

## Anti-patterns

- Enabling `cache=True` "because it is deterministic enough"; it is not.
- Parsing the CLI's plain-text output; use `--output-format json`.
- Running `dspy.Evaluate(num_threads=8)` against the CLI; the processes contend for the same session.
- Treating the CLI's permission mode as a tool sandbox for agents; use `plan` and keep the model text-only.

## Where to go next

- Optimisation on this runtime → `dspy-gepa-optimizer`
- Programs that call `lm.copy(rollout_id=…)` → `dspy-tetraframe`, `dspy-rlm-workflow`
- Full reference (CLI flags, result JSON, error handling, isolation) → [reference.md](reference.md)
- Runnable example → [example_local_runtime.py](example_local_runtime.py)
