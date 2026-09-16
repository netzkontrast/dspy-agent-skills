---
name: dspy-production
description: >-
  Ship a compiled DSPy program and be able to see what it does — cache
  hardening with configure_cache(restrict_pickle=True), state-JSON versus
  cloudpickle save/load and which one is safe to accept from elsewhere, token
  and cost accounting through track_usage, async acall and aforward, streamify
  with StreamListener, dspy.Parallel for batch throughput, and the
  observability layer underneath: inspect_history, GLOBAL_HISTORY, BaseCallback
  hooks and MLflow autolog, with the sampling a high-volume service needs.
when_to_use: >-
  User says "deploy", "production", "ship it", "serve this", "streaming",
  "async", "how much does it cost", "it is slow", "why did it output that",
  "trace", "monitor", "load the saved program"; a compiled artifact must leave
  a notebook; a program behaves differently in production than in evaluation.
---

# DSPy Production (3.3.x)

Consolidated from `dspy-production-deployment` and `dspy-debugging-observability`
of `OmidZamani/dspy-skills` (MIT). They are one skill here because the
questions arrive together: everything below is what you need after the metric
is green and before anyone else depends on the program.

## Load the artifact, not a pickle you were handed

```python
compiled.save("./artifacts/program.json", save_program=False)    # state only, readable
loaded = MyProgram()                                             # you construct the class
loaded.load("./artifacts/program.json")
```

State-only JSON saves instructions and demos. You supply the class, so nothing
executes on load. This is the default for a reason.

```python
compiled.save("./artifacts/program/", save_program=True)          # whole program
loaded = dspy.load("./artifacts/program/")                        # cloudpickle
```

Whole-program save serializes the architecture too, through cloudpickle.
Loading one executes code from the artifact. Use it only for artifacts your own
pipeline produced, never for one that arrived over a wire or from a registry
you do not control. Keep the DSPy major version aligned across save and load.

## Cache hardening

DSPy caches in memory and on disk by default, and disk-cache deserialization
uses pickle unless you restrict it:

```python
dspy.configure_cache(restrict_pickle=True)                        # allowlist mode
dspy.configure_cache(restrict_pickle=True, safe_types=[MyResult]) # add your own types
dspy.configure_cache(enable_disk_cache=False)                     # ephemeral deployments
```

Full parameter list: `enable_disk_cache`, `enable_memory_cache`,
`disk_cache_dir`, `disk_size_limit_bytes`, `memory_max_entries`,
`restrict_pickle`, `safe_types`. Set `disk_cache_dir` to a project-local path
in CI so runs are reproducible and cleanable.

A shared disk cache across tenants is a data-leak path, not an optimization.
Separate the directory per tenant or disable the disk layer.

## Cost and usage

```python
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"), track_usage=True)
prediction = program(question="What is DSPy?")
print(prediction.get_lm_usage())
```

Cached calls report no new usage — which is why a benchmark run with a warm
cache looks free and tells you nothing about production cost. Measure cost with
the cache disabled at least once.

## Async and throughput

| Need | Call | Note |
|---|---|---|
| one request, non-blocking | `await program.acall(...)` | built-in modules support it |
| custom module | implement `aforward()` | mirror `forward()`; do not block inside it |
| wrap a sync program | `dspy.asyncify(program)` | a boundary adapter, not a speedup |
| many independent inputs | `dspy.Parallel(num_threads=8)` | batch throughput |

```python
results = dspy.Parallel(num_threads=8)([(program, {"question": q}) for q in questions])
```

Set `async_max_workers` on `dspy.configure` and `num_threads` to what the
provider's rate limit tolerates. Above that, the extra concurrency converts
directly into 429s and retries.

## Streaming

```python
stream = dspy.streamify(
    dspy.Predict("question -> answer"),
    stream_listeners=[dspy.streaming.StreamListener(signature_field_name="answer")],
)

async for chunk in stream(question="Explain DSPy briefly."):
    print(chunk, end="")
```

`StreamListener(signature_field_name, predict=None, predict_name=None,
allow_reuse=False)`. For a looped module such as `dspy.ReAct` that emits the
same field repeatedly, set `allow_reuse=True` or you get only the first
occurrence. A cache hit yields the final `Prediction` with no token chunks at
all — so test the streaming path with the cache off, or it will look broken in
production and fine in development.

## Seeing what happened

```python
dspy.inspect_history(n=3)                          # prints the last 3 LM calls

from dspy.clients.base_lm import GLOBAL_HISTORY    # same data, programmatic
last = GLOBAL_HISTORY[-1]
print(last["model"], last.get("usage", {}), last.get("cost"))
```

`inspect_history` is the first thing to reach for when output is wrong: it
shows the rendered prompt, so you see what the model actually received rather
than what you think the signature produced. Most "the model ignored my
instruction" bugs are visible in one line of it.

## Callbacks for a running service

```python
from dspy.utils.callback import BaseCallback

class UsageCallback(BaseCallback):
    def on_lm_start(self, call_id, instance, inputs):
        self.started[call_id] = time.perf_counter()

    def on_lm_end(self, call_id, outputs, exception=None):
        if exception:
            self.errors.append(repr(exception)); return
        self.latencies.append(time.perf_counter() - self.started.pop(call_id, 0))

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"), callbacks=[UsageCallback()])
```

Hooks come in start/end pairs: `on_lm_*`, `on_module_*`, `on_tool_*`,
`on_adapter_format_*`, `on_adapter_parse_*`, `on_evaluate_*`, `on_compile_*`,
and the `on_interpreter_*` family. Full list in [reference.md](reference.md).

Callbacks run **synchronously inside the call path**. Anything slow in
`on_lm_end` is added latency on every request. Buffer and flush elsewhere;
never log to a remote sink inline. At high volume, sample:

```python
if random.random() >= self.sample_rate:
    return
```

Redact before logging. Prompts and completions contain whatever the user sent,
and a trace store is rarely covered by the same retention policy as your
primary datastore.

## MLflow

```python
import mlflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("my-program")
mlflow.dspy.autolog(log_traces=True, log_traces_from_compile=True, log_evals=True)
```

Autolog instruments inference, compilation and evaluation without touching the
program. It adds per-call overhead, so enable the compile and eval traces in
development and consider inference-only, sampled, in production.

## Pre-ship checklist

1. Pin the DSPy version; a compiled artifact is not portable across majors.
2. State-only JSON unless a whole-program artifact is genuinely needed and trusted.
3. `restrict_pickle=True`, and a cache directory that is not shared across tenants.
4. Measure cost once with the cache disabled.
5. Load-test async, streaming and batch paths separately; they fail differently.
6. Callbacks sampled, buffered and redacting.
7. Keep the evaluation harness runnable against the deployed artifact, so a regression is measurable rather than anecdotal.

## Anti-patterns

- `dspy.load` on an artifact from an untrusted source; cloudpickle executes code.
- Benchmarking with a warm cache and reporting that as production cost.
- Slow work inside `on_lm_end`, turning observability into latency.
- Streaming tested only against a warm cache, where no chunks are emitted.
- Raising `num_threads` past the provider's rate limit and calling the resulting 429s a DSPy bug.
- Shipping a compiled artifact with no record of the DSPy version, metric and devset score that produced it.
- `track_usage=True` left on at high QPS when nothing reads the counters; it accumulates per prediction.

## Where to go next

- Choosing and running the optimizer that produced the artifact → `dspy-optimizer-selection`
- Metrics and the regression check to keep running after deploy → `dspy-evaluation-harness`
- Gating what the program emits before users see it → `dspy-autodialectics`
- Retrieval-side caching and index persistence → `dspy-retrieval`
- Full reference (every callback hook, configure_cache parameters, save formats) → [reference.md](reference.md)
- Runnable, dry-run capable → [example_production.py](example_production.py)
