# DSPy Production — Reference

Source: `dspy-production-deployment` and `dspy-debugging-observability` of
`OmidZamani/dspy-skills` (MIT), merged. Signatures below were read off the
installed wheel with `inspect.signature`.

## Verified signatures

```python
dspy.configure_cache(enable_disk_cache=True, enable_memory_cache=True,
                     disk_cache_dir="~/.dspy_cache", disk_size_limit_bytes=30_000_000_000,
                     memory_max_entries=1_000_000, restrict_pickle=False, safe_types=None)

dspy.Module.save(path, save_program=False, modules_to_serialize=None)
dspy.load(path)

dspy.streamify(program, status_message_provider=None, stream_listeners=None,
               include_final_prediction_in_output_stream=True,
               is_async_program=False, async_streaming=True)

dspy.streaming.StreamListener(signature_field_name, predict=None,
                              predict_name=None, allow_reuse=False)

dspy.asyncify(program)
dspy.inspect_history(n=1, file=None)

dspy.Parallel(num_threads=None, max_errors=None, access_examples=True,
              return_failed_examples=False, provide_traceback=None,
              disable_progress_bar=False)
```

## Save formats

| | `save_program=False` | `save_program=True` |
|---|---|---|
| What is written | instructions, demos, signature state | the whole program object |
| Format | JSON (also `.pkl` by extension) | cloudpickle directory |
| Loading | construct the class, then `.load(path)` | `dspy.load(path)` |
| Executes code on load | no | **yes** |
| Readable / diffable | yes | no |
| Survives a code refactor | yes, if the class still matches | often not |
| Accept from elsewhere | with review | never |

`modules_to_serialize` lets you name extra modules to include in a
whole-program save when they are not reachable from the program's own tree.

State-only is the default in every sense: safer, smaller, reviewable in a pull
request, and it survives the refactors that break a pickled object graph.

## Cache layers

| Layer | Default | Disable when |
|---|---|---|
| memory | on, 1,000,000 entries | a long-lived process must bound memory |
| disk | on, `~/.dspy_cache`, 30 GB | ephemeral containers, or per-tenant isolation is required |

`restrict_pickle=True` switches disk-cache deserialization to an allowlist.
Register application types explicitly:

```python
dspy.configure_cache(restrict_pickle=True, safe_types=[MyResult, Metadata])
```

Point `disk_cache_dir` at a project-local directory (`.cache/dspy`) in CI so
runs are reproducible and the cache can be cleared with the workspace.

Cache interactions worth knowing before they surprise you:

- A cached call returns **no** new usage, so `get_lm_usage()` under-reports on a warm cache.
- A cached call emits **no** stream chunks; `streamify` yields only the final `Prediction`.
- Optimizer runs warm the cache heavily. The second compile of the same program is not evidence that it got faster.

## Callback hooks

`dspy.utils.callback.BaseCallback` exposes start/end pairs. Override only the
ones you need:

| Family | Hooks |
|---|---|
| LM | `on_lm_start`, `on_lm_end` |
| Module | `on_module_start`, `on_module_end` |
| Tool | `on_tool_start`, `on_tool_end` |
| Adapter format | `on_adapter_format_start`, `on_adapter_format_end` |
| Adapter parse | `on_adapter_parse_start`, `on_adapter_parse_end` |
| Evaluate | `on_evaluate_start`, `on_evaluate_end` |
| Compile | `on_compile_start`, `on_compile_end` |
| Interpreter | `on_interpreter_startup_*`, `on_interpreter_execute_*`, `on_interpreter_tool_call_*`, `on_interpreter_shutdown_*` |

Signatures:

```python
def on_lm_start(self, call_id: str, instance: Any, inputs: dict[str, Any]) -> None: ...
def on_lm_end(self, call_id: str, outputs: dict[str, Any] | None,
              exception: Exception | None = None) -> None: ...
```

`call_id` correlates a start with its end — keep per-call state in a dict keyed
by it, and pop on end so a failed call cannot leak the entry.

Register globally with `dspy.configure(callbacks=[cb])`, or per module with
`dspy.Predict(sig, callbacks=[cb])`.

Constraints that matter in a service:

- Synchronous and inline: every millisecond is request latency.
- Exceptions in a callback surface in the call path; guard your own handlers.
- They do not see inside optimizer internals — use MLflow's compile traces for that.

## Execution history

```python
from dspy.clients.base_lm import GLOBAL_HISTORY
```

A process-global list of call records, each with the model, the rendered
prompt, the response, `usage` and `cost` when the provider reports them. It is
bounded, so treat it as a debugging window, not an audit log. Anything that has
to survive the process belongs in a callback that writes somewhere durable.

`dspy.inspect_history(n=3, file=...)` prints the same records in readable form;
pass `file` to capture it instead of printing to stdout.

## Async

- Built-in modules implement `acall()`.
- A custom module implements `aforward()` alongside `forward()`.
- `dspy.asyncify(program)` adapts a synchronous program at a boundary. It moves work to a thread; it does not make the program concurrent.
- `dspy.configure(async_max_workers=N)` bounds the pool.

Do not call blocking I/O inside `aforward()`. It stalls the event loop, and the
symptom is a throughput ceiling that looks like provider throttling.

## Batch throughput

```python
parallel = dspy.Parallel(num_threads=8, return_failed_examples=True)
results = parallel([(program, {"question": q}) for q in questions])
```

`max_errors` bounds failures before the run aborts; `return_failed_examples`
hands back which inputs failed instead of discarding them;
`provide_traceback` includes the traceback per failure. For evaluation runs,
prefer `dspy.Evaluate(num_threads=...)`, which does this and computes the
metric.

## MLflow autolog

```python
mlflow.dspy.autolog(
    log_traces=True,                # inference
    log_traces_from_compile=True,   # optimizer runs
    log_traces_from_eval=True,      # evaluation runs
    log_compiles=True,
    log_evals=True,
)
```

Per-call overhead is small but not zero. A reasonable split: everything on in
development and CI; inference traces only, sampled, in production.

## What to record with a deployed artifact

| Field | Why |
|---|---|
| DSPy version | artifacts are not portable across majors |
| optimizer, arguments, seed | the artifact cannot be regenerated without them |
| devset score and devset identity | the regression baseline |
| metric version | a changed metric invalidates the comparison |
| LM and provider, including model version | provider-side model updates move scores |
| save format | whether loading it executes code |

Without this, "the model got worse" is unanswerable.
