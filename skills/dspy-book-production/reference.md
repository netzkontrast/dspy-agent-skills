# Serving and Tracing — Reference

Source: chapter 10 of `context-engineering-dspy-book` (MIT). Pins
`dspy==3.3.0`, `mlflow==3.15.1`. Nothing here is outdated against 3.2/3.3.

## Serving

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    dspy.configure(lm=dspy.LM(MODEL), async_max_workers=4)
    app.state.extractor = dspy.asyncify(load_invoice_extractor())
    yield

app = FastAPI(title="Invoice Extraction API", lifespan=lifespan)

@app.post("/extract", response_model=InvoiceResponse)
async def extract_invoice(request: InvoiceRequest):
    try:
        result = await app.state.extractor(text=request.text)
        return result.toDict()
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc
```

`load_invoice_extractor` builds the module and calls `.load(path)` only when the
artifact exists, otherwise printing a warning — which the notebook itself says
to replace with a hard startup failure in production.

`async_max_workers` comes from the provider's rate limit. Start at 4, raise
under load test, watch 429s.

## Guardrails

Three layers:

1. Pydantic `field_validator` bounding the input (for example a 50,000-character cap).
2. An **optimizable** output field on the signature, so the optimizer improves the guard along with the task. The chapter cites a result where attack success fell from 75% to 5%.
3. A deterministic regex on the output, returning 422 on violation.

Wrap in `dspy.Refine(module=..., N=3, reward_fn=passes_guardrail, threshold=1.0)`
so a guardrail miss retries rather than fails outright.

## Fallback

```python
except (APIError, RateLimitError, ServiceUnavailableError, Timeout):
    with dspy.context(lm=fallback_lm):
        ...
```

Typed exceptions only. A bare `except Exception` converts your own bugs into a
silent downgrade to a weaker model.

## Prompt export

`dspy.settings.lm.history[-1]["messages"]` is plain chat format. If a human
will read the exported prompt, configure the readable adapter **before**
optimizing; swapping adapters at export renders demos selected under one format
in another.

## MLflow

```bash
mlflow server --backend-store-uri sqlite:///mydb.sqlite
```

SQLite rather than the file store, which degrades past a few hundred runs.

```python
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("DSPy")
mlflow.dspy.autolog(log_compiles=True, log_evals=True, log_traces_from_compile=True)
```

Logged: a span tree per call (module → adapter format → raw request and
response → adapter parse → output). A `compile()` creates a parent run plus one
child run per evaluation, each holding candidate program state, score and
traces.

Manual comparison pattern: loop models, open a run per model, log the score
plus token count, cost and latency percentiles as metrics.

The chapter does **not** use the model registry; versioning and staging are
left to you.

### The MCP server

`mlflow[mcp]` exposes traces to an agent — 26 tools, 11 of them trace tools.
The two that matter: searching traces for triage, and logging **feedback** (a
judgment about what happened) or an **expectation** (ground truth, curatable
into a later trainset).

Operational rule, and the most useful detail in the chapter: **always pass an
explicit field list when searching traces**, using dot paths, or the default
full span tree exhausts the context window after a few results. Scope the
available tools with the server's tool-filter environment variable.

## The UI pattern

The Gradio notebook builds a two-tab app: runtime signature construction with
`make_signature({name: (str, dspy.InputField(desc=...))}, instructions=...)`, a
module dropdown, CSV upload split 80/20 into examples, optimizer and metric
dropdowns, state-only save, then an inference tab that shows the prediction
**and** the rendered prompt.

Two facts worth carrying: `EvaluationResult.score` is on a 0–100 scale, not
0–1; and a teacher is made with `module.deepcopy()` followed by `set_lm`.
