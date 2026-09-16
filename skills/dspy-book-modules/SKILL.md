---
name: dspy-book-modules
description: >-
  Customize a DSPy program beyond Predict and ChainOfThought, from chapter 7 of
  Context Engineering with DSPy — adapters that change how a Signature becomes
  a prompt (chat, JSON, XML, two-step for reasoning models), multimodal inputs
  with dspy.Image and dspy.Audio, choosing between ProgramOfThought, CodeAct and
  RLM for code execution, dspy.Parallel with majority voting, and the
  composition patterns for sequential, branching and self-refining modules.
when_to_use: >-
  User says "adapter", "JSON output", "structured output is malformed",
  "image input", "audio", "multimodal", "ProgramOfThought vs CodeAct", "run
  these in parallel", "majority vote", "self-refine", "reasoning model breaks
  parsing"; a program needs more than one predictor or more than text.
---

# Customizing Programs (book chapter 7)

From the O'Reilly companion repo (MIT). Two of these topics — adapters and
multimodal — are covered nowhere else in this pack.

## Adapters: how a Signature becomes a prompt

An adapter translates your Signature into a prompt and parses the result back.
You usually never touch it. Selection is **explicit and global**, and it
persists until replaced.

```python
dspy.configure(lm=lm, adapter=dspy.JSONAdapter())
```

| Adapter | Use it when |
|---|---|
| `ChatAdapter` (default) | ordinary text fields |
| `JSONAdapter` | the output type is a Pydantic model and you want schema-validated structured output |
| `XMLAdapter` | you want a human-readable transcript; some models historically preferred XML |
| `TwoStepAdapter` | a reasoning model's long chain of thought breaks field parsing |

The stated "when the default is wrong" case is worth memorizing: **DSPy does
not select `JSONAdapter` automatically for a Pydantic output type.** You get
the default adapter parsing a model into text fields, and the failure looks
like a flaky model rather than a configuration mistake.

`TwoStepAdapter(extraction_model=...)` makes two calls: unconstrained
generation, then a small model extracts the fields. That is the fix for o3-class
models whose reasoning text swamps the field markers.

Debug any of it with `dspy.inspect_history(n=1)`, which shows the rendered
prompt. Custom adapters subclass `ChatAdapter` and override three hooks:
`format_field_with_value`, `user_message_output_requirements`, `parse`.

## Multimodal

```python
img = dspy.Image(pil_image)          # constructor accepts URL, path, PIL, bytes, data URI
classify = dspy.Predict("image: dspy.Image, category_options: list[str] -> category: str")

audio = dspy.Audio.from_array(audio_array, sampling_rate=16_000)
extract = dspy.ChainOfThought("recording: dspy.Audio, context: str -> action_items: list[str]")
```

`Image.from_url()` and `Image.from_file()` still work but are **deprecated** —
use the constructor. `Audio.from_file` / `from_url` / `from_array` are still the
sanctioned path, which is an inconsistency worth remembering.

Media types go in the signature string with the dotted name
(`image: dspy.Image`). There is no built-in video type; documents come through
the third-party `attachments` package.

Not every provider accepts audio. The chapter's own setup uses a text-and-image
model, which is insufficient for the audio example — check capability before
blaming the code.

## Code execution: three modules, one decision

| Module | What it does | Pick it when |
|---|---|---|
| `ProgramOfThought` | writes Python, runs it sandboxed, returns the result | execution beats reasoning: arithmetic, counting, sorting, data transforms |
| `CodeAct` | writes Python that calls **your tools** | the work needs your functions, composed in code |
| `RLM` | a sandboxed REPL with your context as variables, plus a cheap sub-model | the context is too long to reason over directly, roughly 50,000 tokens and up |

`CodeAct` has a narrower tool contract than ReAct: **tools must be plain Python
functions**, not callable objects, because they are injected into the
interpreter. ReAct accepts functions, callables or `dspy.Tool`.

All three run on Deno plus Pyodide, which must be installed in a terminal, not
a notebook cell. The sandbox restricts file system, environment and network by
default — which is why an RLM cannot read host files and you must pass contents
as input values.

## Parallel and voting

```python
result = cot(entity="Apple", config={"n": 5, "temperature": 1.0})
final = dspy.majority(result.completions, normalize=normalize_location)
```

`dspy.majority` is a **function over completions**, not a module. You must
generate the completions first with `config={"n": ...}`. A `normalize` callable
folds aliases together before counting, which is usually the difference between
voting working and voting splitting.

```python
pairs = [(translate, dspy.Example(text=t, target_language=lang).with_inputs("text", "target_language"))
         for lang in languages]
results = dspy.Parallel(num_threads=5, max_errors=2)(pairs)
```

`dspy.Parallel` takes `(module, Example)` pairs and the Example must already
have `.with_inputs()` applied. `module.batch(examples, num_threads=5)` is the
lighter form for one module over many inputs. Total cost is unchanged; only
wall-clock time drops.

## Composition patterns

Sequential, conditional branching on a router predictor's output, wrapping an
inner module in `BestOfN`, per-module LMs via `set_lm`, and a manual
draft/critique/revise loop that breaks on a score threshold — the chapter calls
that last one "the manual version of what `Refine` does automatically."

**The chapter contains no stage-level error handling at all**: no try/except,
no fallback stage, no validation gate between stages. The only failure
mechanisms anywhere are `BestOfN`/`Refine` thresholds and `Parallel(max_errors=)`.
Do not read these patterns as production-ready; add the gates yourself.

## Anti-patterns

- A Pydantic output type without configuring `JSONAdapter`, then debugging a "flaky model".
- `Image.from_file()` in new code; the constructor is the current path.
- Assuming an audio-capable signature works on a text-and-image model.
- `dspy.majority` without `config={"n": ...}` first — there are no completions to count.
- `dspy.Parallel` with raw dicts instead of `Example.with_inputs(...)`.
- A callable object as a `CodeAct` tool; it must be a plain function.
- Treating these composition patterns as robust; none of them handle a failing stage.

## Where to go next

- RLM in depth, and its workflow → `dspy-rlm-module`, `dspy-rlm-workflow`
- Agents and tools → `dspy-book-agents`
- Serving, caching and tracing → `dspy-production`
- Full reference (adapter hooks, media types, sandbox setup) → [reference.md](reference.md)
- Runnable, dry-run capable → [example_module_choice.py](example_module_choice.py)
