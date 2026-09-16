# DSPy RLM Hooks — Reference

Source: `dspy-rlm-hooks` 0.1.14 (MIT), verified by importing the package and
reading `inspect.signature`, not from its README.

Install: `pip install dspy-rlm-hooks`. Requires Python >= 3.12, `dspy>=3.1.0`,
`pydantic>=2`. Extras: `predict-rlm`, `tracing` (`mlflow>=2.14`).

## Exports

`enable_rlm_hooks`, `enable_rlm_hooks_with_tracing`, `disable_rlm_hooks`,
`enable_rlm_speculation`, `disable_rlm_speculation`, the four hook protocols
and their four output models, `RLMHook`, `SpeculationConfig`,
`SpeculationPolicy`, `speculate`, `speculative`, `Spec`, `SpecTool`,
`Speculator`, `SpecSession`, `StreamTurn`.

`enable_predict_rlm_hooks` / `disable_predict_rlm_hooks` are **not** top-level:
import them from `dspy_rlm_hooks.core`.

## Hook output models

```python
PreIterationOutput(extra_vars: dict = {}, python_code: str = "",
                   persistent_python_code: str | None = None,
                   prompt_context: str = "")
PreExecutionOutput(code: str)
PostExecutionOutput(result: Any)
PostIterationOutput(history: Any, stop: bool = False)
```

| Field | Effect |
|---|---|
| `extra_vars` | merged into the REPL namespace for this iteration |
| `python_code` | executed this iteration only |
| `persistent_python_code` | replaces the persistent prelude. `""` clears it; `None` keeps the current one |
| `prompt_context` | appended to the variables blurb the model sees. Never executed |
| `stop` | ends iteration and force-extracts an answer via the max-iterations path |

## Attachment contract

`enable_rlm_hooks` validates the instance before patching and raises
`AttributeError` if it lacks `_execute_iteration`, `_aexecute_iteration`,
`_process_execution_result`, `generate_action`, `verbose`, or a
`max_iters` attribute (named `max_iterations` before DSPy 3.3.0). Patches are bound per instance with
`MethodType`; two RLM objects can carry different hooks.

Because the patch targets are private DSPy names, treat the pair
`(dspy, dspy-rlm-hooks)` as one version unit and pin both.

## Speculation

`SpeculationConfig` fields and defaults:

| Field | Default | Meaning |
|---|---|---|
| `enabled` | True | master switch |
| `max_inflight` | 8 | concurrent speculative dispatches |
| `max_dispatches_per_turn` | 2048 | per-turn dispatch ceiling |
| `speculate_llm_query` | True | speculate the built-in sub-LM query |
| `speculate_llm_query_batched` | True | and its batched form |
| `speculate_user_tools` | False | your tools are opt-in |
| `timeout_s` | 5.0 | per-speculation wait |
| `streaming` | True | segment while generating; False forces one-shot |
| `persistent_shadow` | True | reuse a warm worker across iterations |
| `latency_aware` | True | prioritize by latency hint |

`SpeculationPolicy` (dataclass): `speculatable=False`, `pure=False`,
`deterministic=False`, `latency_hint_ms=1000.0`, `gate` — a per-call predicate
over `(args, kwargs)` that can veto speculating a specific call.

```python
speculative(fn, *, name=None, deterministic=False, latency_hint_ms=1000.0)
speculate(tool, *, policy=None, **policy_kwargs)   # ValueError if speculatable without pure
```

The tool `name` must match its REPL registration name, or the claim hook never
fires and every speculation is evicted.

Subpackages: `streaming/` parses and plans but never executes (`StreamSegmenter`,
`safe_eval`, `plan_peeks`); `shadow/` runs predictions in a jailed subprocess
with a per-statement watchdog; `budget.py` bounds speculative concurrency only,
not RLM's real `max_llm_calls`; `guards.py` tags claim hooks so a hook cannot
wait on its own pending speculation, which would deadlock.

## Benchmark harness

```bash
python -m dspy_rlm_hooks.benchmark --variants default spec --repeats 5 \
    --tool-ms 60 --llm-ms 150 --json-out report.json
```

Programmatic: `run_ab(scenario=None, variants=None, repeats=3, ...)`,
`run_variant(scenario, variant="default", repeats=3, ...)`. Custom variant is
any `callable(rlm, tools)`, selected with `--variant-fn mymod:my_variant`.

`Scenario(name, iterations, tools, llm_latency_ms=150.0, max_llm_calls=50)` with
`ScriptedIteration(reasoning, code)` and `ToolPlan(name, latency_ms, result)`.
`default_scenario()` covers peeked constant arguments, a dependent-chain miss,
loop calls and a submit.

Read `equivalent_steps` and `mismatches` before any speedup number: the harness
only lets a comparison stand if the variants executed the same steps.

## PredictRLM

For `predict_rlm.PredictRLM` (optional extra), the same four-hook contract is
available through `enable_predict_rlm_hooks` / `disable_predict_rlm_hooks` in
`dspy_rlm_hooks.core`. It wraps `_execute_iteration`, `forward` and
`_process_execution_result`, and implements `stop=True` with an internal
exception caught by the wrapper so the caller still gets a `Prediction`.
`_is_predict_rlm(rlm)` returns False when the package is absent.

## Safety notes from SECURITY.md

- RLM executes LM-generated Python. Hooks observe and rewrite; they are not a sandbox.
- Any code holding the RLM instance can call `enable_rlm_hooks` and inject behaviour. Treat hook sources as trusted code.
- Pin DSPy: the package instruments internals that may change between releases.

## Status

Development Status 3 (Alpha). The speculation package's own docstrings still
reference internal build phases. Treat the hook API as the stable part and
speculation as the experimental part.
