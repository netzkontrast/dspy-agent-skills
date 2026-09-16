---
name: dspy-rlm-hooks
description: >-
  Instrument dspy.RLM with the dspy-rlm-hooks package — four lifecycle hooks
  (pre-iteration, pre-execution, post-execution, post-iteration) that inject
  variables, rewrite generated code, transform results and stop the loop early,
  plus speculative execution that pre-runs predicted tool calls in a sandboxed
  subprocess while the main context is still generating. Covers the exact hook
  signatures, the order-dependent composition with speculation, the purity rule
  speculated tools must satisfy, and the LM-free benchmark harness.
when_to_use: >-
  User says "RLM hooks", "instrument RLM", "speculative execution", "the RLM
  loop is slow", "inject state into RLM", "stop RLM early", "see what RLM is
  doing per iteration"; an RLM run needs observability, injected context, or
  overlapping tool latency with generation.
---

# DSPy RLM Hooks (dspy-rlm-hooks 0.1.14)

`dspy-rlm-hooks` (MIT) adds lifecycle instrumentation to `dspy.RLM`. This skill
covers only what it adds on top of plain RLM — `dspy-rlm-module` owns RLM
itself, and `dspy-rlm-workflow` owns the surrounding workflow.

Two independent features, usable separately:

| Feature | What it buys | Cost |
|---|---|---|
| **hooks** | see and shape every RLM iteration; inject variables, rewrite code, stop early | one call per stage |
| **speculation** | overlap tool and sub-LM latency with token generation | subprocesses, threads, a purity requirement |

It works by **instance-level monkeypatching of private DSPy internals**
(`_execute_iteration`, `_execute_code`, `generate_action`, …). That is the
central fact about this package: its own SECURITY.md says to pin a known-good
DSPy version, because a DSPy release is free to rename any of them. Validation
is upfront — `enable_rlm_hooks` raises `AttributeError` immediately if the
instance does not expose the expected internals, so breakage is loud.

## The four hooks

```python
import dspy
from dspy_rlm_hooks import enable_rlm_hooks, PreIterationOutput

rlm = dspy.RLM("question -> answer")

def inject(iteration, variables, history, input_args):
    return PreIterationOutput(extra_vars={"budget_left": 3},
                              prompt_context="Prefer the cached table.")

enable_rlm_hooks(rlm, pre_iteration_hook=inject)
result = rlm(question="What is 2 + 2?")
```

| Hook | Signature (positional) | Returns | Use it to |
|---|---|---|---|
| `pre_iteration_hook` | `(iteration, variables, history, input_args)` | `PreIterationOutput` | inject variables, prepend code, add prompt context |
| `pre_execution_hook` | `(iteration, code, variables, history, input_args)` | `PreExecutionOutput(code)` | rewrite or veto the generated code |
| `post_execution_hook` | `(iteration, code, result, variables, history, input_args)` | `PostExecutionOutput(result)` | redact, cap or transform the result |
| `post_iteration_hook` | `(iteration, pred, code, result, history)` | `PostIterationOutput(history, stop=False)` | log the turn, or `stop=True` to end early |

All are keyword-only on `enable_rlm_hooks`, all optional, and each stage takes
**one** callable. Calling `enable_rlm_hooks` again replaces the previous hooks;
it does not append. Hooks may be sync or async — a coroutine is awaited.

`PreIterationOutput` fields: `extra_vars` (dict merged into the namespace),
`python_code` (runs this iteration only), `persistent_python_code` (replaces
the persistent prelude; `""` clears it, `None` leaves it alone), and
`prompt_context` (appended to the variables blurb, **not executed**).

`stop=True` on `PostIterationOutput` halts the loop and force-extracts a final
answer through the same path RLM uses when it hits `max_iters`, so the caller
still receives a normal `Prediction`.

`disable_rlm_hooks(rlm)` reverts the instance. When MLflow is importable, the
top-level `enable_rlm_hooks` transparently upgrades to the tracing variant and
emits a span per hook; `enable_rlm_hooks_with_tracing` forces it.

## Speculative execution

While RLM is still streaming the `code` field, the package parses the emerging
statements, resolves constant call arguments, and pre-runs the predicted calls
in a **subprocess**. When real execution reaches that call, a claim hook takes
the finished result instead of calling again.

```python
from dspy_rlm_hooks import enable_rlm_speculation, speculative

enable_rlm_hooks(rlm, post_iteration_hook=log)      # hooks FIRST
enable_rlm_speculation(rlm, tools=[speculative(lookup_price, deterministic=True)])
```

**Order is load-bearing.** `enable_rlm_hooks` overwrites `_execute_code`, so
calling it *after* `enable_rlm_speculation` silently disables speculation. No
error, just no speedup.

**Speculated tools must be pure.** They run early, possibly more than once, and
possibly for a branch that never executes. `speculate()` refuses
`speculatable=True` without `pure=True`. User tools are off by default
(`speculate_user_tools=False`); the built-in `llm_query` and
`llm_query_batched` are on.

Key `enable_rlm_speculation` arguments: `tools`, `max_inflight=8`,
`max_dispatches_per_turn=2048`, `speculate_llm_query=True`,
`speculate_user_tools=False`, `timeout_s=5.0`, `streaming=True`,
`persistent_shadow=True`, `latency_aware=True`. Always call
`disable_rlm_speculation(rlm)` when done: it spawns subprocesses, threads and
an asyncio loop.

The subprocess is not a convenience. An in-process jail cannot stop the
`().__class__.__mro__[1].__subclasses__()` escape, so isolation is a real
process boundary with a per-statement watchdog.

## Measuring whether it helped, without an LM

```bash
python -m dspy_rlm_hooks.benchmark --variants default spec --repeats 5
```

Variants: `default` (plain RLM), `spec` (speculation on), `hooks` (no-op hooks,
which measures hook overhead alone). Scenarios are scripted and the sub-LM is
faked, so **no API key and no LM are needed** — dspy must be installed, but
generation is replaced wholesale.

The report gives wall time, generation and execution totals, tool critical path
versus serial time, and speculation counters (`speculated`, `claimed`,
`evicted`). It also asserts step equivalence across variants: a speedup only
counts when every variant ran the same steps with the same results. A high
`evicted` count relative to `claimed` means the predictions were wrong and you
paid for shadow work that was thrown away.

## Anti-patterns

- Enabling speculation before hooks — silently no-ops.
- Speculating an impure tool: one that writes, charges, or reads mutable state. It may run for a branch that never executes.
- Treating hooks as a sandbox. RLM still executes LM-generated Python; hooks observe and rewrite, they do not contain.
- Leaving a speculating RLM undisposed — subprocesses and threads outlive the call.
- Floating the DSPy version. This patches private internals; pin both packages together.
- Registering two hooks for one stage and expecting both to run.
- Claiming a speedup from wall time alone; read `equivalent_steps` and the evicted counter first.

## Where to go next

- RLM itself, its parameters and when to use it → `dspy-rlm-module`
- The verified multi-step workflow around it → `dspy-rlm-workflow`
- Callback-based observability for non-RLM programs → `dspy-production`
- Full reference (every field, the speculation internals, PredictRLM) → [reference.md](reference.md)
- Runnable, dry-run capable → [example_rlm_hooks.py](example_rlm_hooks.py)
