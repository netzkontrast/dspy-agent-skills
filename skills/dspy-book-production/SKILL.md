---
name: dspy-book-production
description: >-
  Serve and observe a compiled DSPy program, from chapter 10 of Context
  Engineering with DSPy — loading the program once at startup behind an async
  API, guardrails expressed as optimizable signature fields rather than
  hand-written checks, typed provider-failure fallback, exporting the rendered
  prompt for human review, and the MLflow tracing setup including its MCP
  server with the field-extraction rule that keeps traces from flooding a
  context window.
when_to_use: >-
  User says "serve this", "FastAPI", "deploy the compiled program", "MLflow",
  "trace", "guardrail", "injection", "fallback model", "export the prompt";
  a compiled artifact must run behind an interface and be observable.
---

# Serving and Tracing (book chapter 10)

From the O'Reilly companion repo (MIT). `dspy-production` owns cache hardening,
save formats, async, streaming and callbacks. This skill covers what chapter 10
adds on top: the serving shape, guardrails, and the tracing workflow.

## Load once, serve async

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    dspy.configure(lm=dspy.LM(MODEL), async_max_workers=4)
    app.state.extractor = dspy.asyncify(load_extractor())
    yield

@app.post("/extract")
async def extract(request: InvoiceRequest):
    result = await app.state.extractor(text=request.text)
    return result.toDict()
```

The program is built and loaded **once** in the lifespan, stashed on app state,
and wrapped with `asyncify`. Not per request.

`async_max_workers` is sized from the provider's rate limit, not the CPU count.
Start at 4 and raise under load test while watching for 429s.

The chapter's loader prints a warning when the compiled artifact is missing and
says explicitly to replace that with a hard startup failure in production. Do
that: a service silently serving an uncompiled program is worse than one that
refuses to boot.

## Guardrails as an optimizable field

The idea worth taking: instead of writing an injection check by hand, declare
it as an output field on the signature —

```python
is_invoice: bool = dspy.OutputField(desc="False if the text is not an invoice")
```

— and let the optimizer improve it along with everything else. The chapter
cites a DSPy guardrails result where attack success dropped from 75% to 5%.

Combine three layers: a Pydantic `field_validator` bounding the input, the
optimizable signature field, and a deterministic regex on the output, returning
422 on violation. Wrap the whole thing in `dspy.Refine(..., reward_fn=passes_guardrail,
threshold=1.0)` so a guardrail failure retries rather than fails.

## Fall back on typed errors only

```python
except (APIError, RateLimitError, ServiceUnavailableError, Timeout):
    with dspy.context(lm=fallback_lm):
        ...
```

Catch the provider's typed exceptions, never bare `Exception`. A bare catch
turns your own bugs into silent fallbacks to a weaker model.

## Exporting the prompt

`dspy.settings.lm.history[-1]["messages"]` is plain chat format, which is how
you hand an optimized prompt to a human or another system.

One trap: if a human will read it, configure a readable adapter **before**
optimizing. Swapping adapters at export time renders demos that were selected
under one format in a different one.

## MLflow

```python
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("DSPy")
mlflow.dspy.autolog(log_compiles=True, log_evals=True, log_traces_from_compile=True)
```

Use a SQLite backend store rather than the file store, which degrades past a
few hundred runs. A `compile()` produces a parent run plus one child run per
evaluation, each holding the candidate program state, its score and full
traces.

The genuinely new piece is the **MLflow MCP server**, which exposes traces to
an agent. Two tools matter: searching traces for triage, and logging feedback
or expectations on a trace — feedback being a judgment about what happened,
expectations being ground truth you can later curate into a trainset.

**Always pass an explicit field list when searching traces.** The default
returns the full span tree and will exhaust a context window after a few
results. This is the single most useful operational detail in the chapter.

Note that the chapter does **not** use the model registry. Versioning and
staging are left to you.

## Anti-patterns

- Loading or compiling the program inside the request handler.
- Sizing `async_max_workers` from CPU count instead of the rate limit.
- Booting with a missing compiled artifact and only printing a warning.
- `except Exception` around an LM call, which hides your own bugs as fallbacks.
- Searching traces without a field list, then wondering where the context went.
- Switching adapters between optimization and export.
- Hand-writing a guardrail that could be an optimizable output field.

## Where to go next

- Cache hardening, save formats, streaming, callbacks → `dspy-production`
- The compiled artifact being served → `dspy-optimizer-selection`
- Application architectures behind the endpoint → `dspy-book-use-cases`
- Full reference (lifespan code, MCP tools, Gradio pattern) → [reference.md](reference.md)
- Runnable, dry-run capable → [example_serving_checks.py](example_serving_checks.py)
