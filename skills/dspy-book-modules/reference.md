# Customizing Programs — Reference

Source: chapter 7 of `context-engineering-dspy-book` (MIT). Repo pins `dspy==3.3.0`.

## Adapters

```python
dspy.configure(lm=lm, adapter=dspy.JSONAdapter())     # global and sticky
assert isinstance(dspy.settings.adapter, dspy.JSONAdapter)

main_lm = dspy.LM("<reasoning-model>", temperature=1.0, max_tokens=16000)
adapter = dspy.TwoStepAdapter(extraction_model=dspy.LM("<small-model>"))
dspy.configure(lm=main_lm, adapter=adapter)
```

| Adapter | Wire format | Notes |
|---|---|---|
| `ChatAdapter` | `[[ ## field_name ## ]]` markers | default |
| `JSONAdapter` | JSON schema from the Pydantic model | uses the API's structured-output mode where supported; returns a validated object |
| `XMLAdapter` | `<field>content</field>` | extends ChatAdapter minimally; easier to read |
| `TwoStepAdapter` | two calls | unconstrained generation, then a small model extracts fields |

`dspy.BAMLAdapter` ships in DSPy 3.x but is **not covered** by this chapter.

Custom adapter hooks:

```python
from dspy.adapters.chat_adapter import ChatAdapter, FieldInfoWithName
from dspy.adapters.utils import format_field_value, parse_value

class MarkdownAdapter(ChatAdapter):
    def format_field_with_value(self, fields_with_values: dict[FieldInfoWithName, Any]) -> str: ...
    def user_message_output_requirements(self, signature) -> str: ...
    def parse(self, signature, completion) -> dict[str, Any]: ...
```

Subclass `Adapter` directly for full control of the message structure.

## Multimodal

```python
dspy.Image(pil_image)                    # URL, path, PIL, bytes or data URI
dspy.Audio.from_array(arr, sampling_rate=16_000)
dspy.Predict("image: dspy.Image, category_options: list[str] -> category: str, confidence: float")
```

`Image.from_url()` / `Image.from_file()` are deprecated in favour of the
constructor. `Audio.from_file` / `from_url` / `from_array` are not.

Custom media type: subclass `dspy.adapters.types.base_type.Type` and implement
`format() -> list[dict] | str` returning LiteLLM content blocks. Two details
from the chapter's video example: some providers expect video through the
image-url block shape rather than a video-url type, and `format()` can read
`dspy.settings.lm.model` to reject models that cannot handle the medium.

Documents arrive through the third-party `attachments` package, which exposes a
`dspy.Type` subclass usable directly in a signature.

## Code execution

```python
dspy.ProgramOfThought("portfolio: str -> total_return: str")
dspy.CodeAct("question: str -> answer: str", tools=[get_population, get_temperature], max_iters=5)
dspy.RLM("document: str, question: str -> answer: str",
         max_iters=20, max_llm_calls=50, sub_lm=dspy.LM("<cheap-model>"))
```

Sandbox: Deno plus Pyodide. Install Deno from a terminal, not a notebook cell.
File system, environment variables and network are restricted by default.

`CodeAct` tools must be plain functions — they are injected into the
interpreter. `ReAct` accepts functions, callable objects or `dspy.Tool`.

RLM is marked experimental in DSPy and is the most likely of these to change.

## Parallel and voting

```python
result = cot(entity="Apple", config={"n": 5, "temperature": 1.0})
final = dspy.majority(result.completions, normalize=lambda s: s.strip().lower())

pairs = [(mod, dspy.Example(**kw).with_inputs(*kw)) for kw in inputs]
dspy.Parallel(num_threads=5, max_errors=2)(pairs)
module.batch(examples, num_threads=5)
```

`dspy.majority` operates on `.completions`; without `config={"n": ...}` there is
exactly one completion and voting is a no-op. `normalize` folds aliases before
counting.

Cost is unchanged by parallelism; only wall-clock time drops.

## Composition patterns

| Pattern | Shape |
|---|---|
| sequential | stages pass plain kwargs forward |
| conditional | a router predictor's output drives a Python `if` |
| wrapping | inner module inside `BestOfN(module=, N=, reward_fn=, threshold=)` |
| per-module LM | `self.extract.set_lm(dspy.LM(...))` |
| self-refinement | draft → critique (`-> feedback: str, score: float`) → revise, break on threshold |

`BestOfN` and `Refine` share the signature `(module=, N=, reward_fn=, threshold=)`.
The difference: `Refine` generates targeted feedback for the next retry after an
attempt misses the threshold. Note the calling convention —
`reward_fn(args_dict, pred)`, not the metric signature.

`MultiChainComparison(signature, M=3, temperature=1.0)` avoids writing a reward
function at all, comparing completions instead.

## Known gaps

The chapter has no stage-level error handling anywhere: no try/except, no
fallback stage, no validation between stages. `BestOfN`/`Refine` thresholds and
`Parallel(max_errors=)` are the only failure mechanisms present.

No module is flagged as rarely worth using; the tour is uniformly positive.

Model identifiers in these notebooks are forward-dated placeholders that
contradict the notebooks' own setup cells. Substitute real ones.
`ChainOfThought(rationale_field=...)` is the legacy spelling; verify against
your pinned version.
