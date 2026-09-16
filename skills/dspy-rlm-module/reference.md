# `dspy.RLM` — Reference

Source: https://dspy.ai/api/modules/RLM/ (DSPy 3.3.x).

`dspy.RLM` is marked **Experimental** upstream: its docstring warns the class may change or be removed without notice. It already moved twice between 3.2.1 and 3.3.1 (see *Renamed in 3.3.0* below). Pin your DSPy version if you depend on it.

## Constructor

```python
dspy.RLM(
    signature: type[Signature] | str,
    max_iters: int = 20,
    max_llm_calls: int = 50,
    max_output_chars: int = 10_000,
    verbose: bool = False,
    tools: list[Callable] | None = None,
    sub_lm: dspy.LM | None = None,
    interpreter_factory: Callable[[], CodeInterpreter] = PythonInterpreter,
)

# forward() — the caller-owned interpreter is positional-only:
def forward(self, interpreter: CodeInterpreter | None = None, /, **input_args) -> Prediction
```

## Renamed in 3.3.0

| 3.2.x | 3.3.x |
|---|---|
| `max_iterations=20` | `max_iters=20` |
| `RLM(..., interpreter=instance)` | `RLM(..., interpreter_factory=callable)`; pass an instance to `rlm(instance, **inputs)` |

Both are hard `TypeError`s, not deprecation warnings. `ProgramOfThought` and
`CodeAct` took the same `interpreter` → `interpreter_factory` move in the same
release, so a codebase that passed a shared sandbox to any of the three needs the
same edit in every place.

## Parameter reference

| Param | Default | Purpose |
|---|---|---|
| `signature` | required | Standard DSPy signature (string or class) |
| `max_iters` | 20 | Max REPL steps before returning |
| `max_llm_calls` | 50 | Hard cap across the whole RLM invocation |
| `max_output_chars` | 10_000 | Truncates each REPL stdout blob before it reaches the LM |
| `verbose` | False | Stream thought/code/output to stdout |
| `tools` | None | Python callables exposed inside the sandbox |
| `sub_lm` | None → `dspy.settings.lm` | Model used for internal recursive calls |
| `interpreter_factory` | `PythonInterpreter` | Zero-arg callable returning a `CodeInterpreter`; called per invocation |

## Interpreter

Default is `dspy.PythonInterpreter` (also `dspy.primitives.PythonInterpreter`), a
Deno-hosted Pyodide WASM sandbox. Requires Deno installed (`brew install deno` /
https://deno.land). The sandbox has no network or filesystem access unless you
opt in via `enable_read_paths` / `enable_write_paths` / `enable_network_access`.

> These are **not** under `dspy.utils`. An earlier revision of this page said
> `dspy.utils.PythonInterpreter`; `dspy/utils/__init__.py` exports no interpreter
> in either 3.2.1 or 3.3.1, so that import has never resolved. `dspy.primitives.*`
> works on both; the bare `dspy.PythonInterpreter` alias is 3.3.x-only.

To use a custom runtime, subclass `dspy.primitives.CodeInterpreter` and implement
its four members — `start()`, `execute(code: str, variables: dict | None = None)`,
`shutdown()` and a `tools` mapping. RLM updates that mutable `tools` dict with
invocation-scoped tools before each execution.

Because `interpreter_factory` is a *factory*, each invocation gets a fresh sandbox
by default. To reuse one across calls, construct it yourself and pass it to the
call: `rlm(my_interpreter, context=..., query=...)`. One interpreter may be reused
sequentially by one RLM instance, but must not be shared by overlapping invocations.

## Tools

Tools are regular Python callables. DSPy introspects type hints and docstrings to expose them to the LLM. Tool dispatch is kwargs-only, so use explicit named parameters:

```python
def read_file(path: str) -> str:
    """Return the full text of a file."""
    return open(path).read()

def grep(pattern: str, text: str) -> list[str]:
    """Return lines matching the regex."""
    import re
    return [l for l in text.splitlines() if re.search(pattern, l)]

rlm = dspy.RLM("repo, q -> answer", tools=[read_file, grep])
```

## Return value

Calling `rlm(...)` returns a `dspy.Prediction` with the signature's output fields. With `track_usage=True` on the outer config, `.get_lm_usage()` aggregates tokens across every inner call.

## Common failures

| Symptom | Fix |
|---|---|
| `deno: command not found` | Install Deno. |
| `RLM hit max_iters` | Raise `max_iters`, or narrow the query. |
| `TypeError: unexpected keyword argument 'max_iterations'` | You are on 3.3.x; rename to `max_iters`. |
| `TypeError: unexpected keyword argument 'interpreter'` | You are on 3.3.x; pass `interpreter_factory=`, or hand the instance to the call. |
| `Sub-LM call count exceeded` | Raise `max_llm_calls`; check for infinite recursion in tools. |
| `Output truncated at 10000 chars` | Raise `max_output_chars`, or have the LM sample/aggregate. |
| `KeyError` in final `.answer` | The RLM gave up; print `verbose=True` trace to see why. |
