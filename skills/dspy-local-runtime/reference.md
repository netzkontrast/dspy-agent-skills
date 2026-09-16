# DSPy Local Runtime — Reference

Source: [Hmbown/dspy-local](https://github.com/Hmbown/dspy-local) (MIT), a
fork of DSPy that ships `dspy/clients/claude.py`. Only that file is the
pattern; the fork's DSPy version lags upstream, so vendor the one file into a
project on current DSPy and adapt two imports (`dspy.clients.base_lm.BaseLM`,
`dspy.dsp.utils.settings.settings`). The teaching-sized version in
[example_local_runtime.py](example_local_runtime.py) omits HOME isolation and
per-kwarg error messages.

## The CLI call

```
claude -p --output-format json --permission-mode plan --no-session-persistence [--model <alias>] [--system-prompt <text>]
```

| Flag | Why |
|---|---|
| `-p` | non-interactive: read the prompt from stdin, print, exit |
| `--output-format json` | one JSON object on stdout with `result`, `usage`, `is_error`, `session_id`, `total_cost_usd` |
| `--permission-mode plan` | the model may not run tools; text in, text out |
| `--no-session-persistence` | no session files written per call |
| `--model` | `haiku`, `sonnet`, `opus` or a full model id; omitted for `claude/default` |
| `--system-prompt` | DSPy's system message (adapter instructions and the signature docstring) |

Result JSON, the fields that matter:

```json
{"type": "result", "result": "…text…", "is_error": false,
 "usage": {"input_tokens": 12, "output_tokens": 140, "cache_read_input_tokens": 0},
 "session_id": "…", "total_cost_usd": 0.02}
```

`parse_result` maps `input_tokens` → `prompt_tokens`, `output_tokens` →
`completion_tokens`, and raises on `is_error`.

## Prompt building

DSPy calls `forward(prompt=…)` or `forward(messages=[…])`. `build_prompt`
joins all `system` messages into the system prompt and renders the rest as
user text (`role: content` for non-user turns), because the CLI takes exactly
one system prompt and one user text per process.

## Kwargs the CLI cannot honour

| kwarg | handling | consequence |
|---|---|---|
| `temperature`, `max_tokens`, `rollout_id` | stripped in `__init__` and `copy()` | `BestOfN` / `Refine` / TetraFrame corners get distinct instructions, not distinct sampling |
| `n`, `num_generations` > 1 | `ValueError` | one completion per call; loop with `dspy.BestOfN` |
| `cache=True` | `ValueError` | no persistent cache; `DSPY_CACHEDIR` is unused for this LM |
| `tools`, `tool_choice`, `response_format`, `logprobs` | `ValueError` in the upstream file | the CLI in `plan` mode has no tool or logprob surface |

## Backend selection

```
DSPY_LOCAL_BACKEND = api | claude-cli | auto (default)
auto: ANTHROPIC_API_KEY or OPENAI_API_KEY present → api; else `claude` on PATH → claude-cli; else api
```

Keep two model-id sets: API ids (`anthropic/claude-haiku-4-5`) and CLI
aliases (`claude/haiku`). Emit a warning when an override for the inactive
backend is set, so a user who exported `KP_LM_TASK`-style variables learns
why they are ignored.

## Isolation (upstream only)

The upstream `ClaudeLM` runs each process with `HOME` pointed at a temporary
directory into which only the CLI's credential files are copied, so a DSPy
run never reads or writes the user's real session state, and
`probe_claude_runtime()` reports `ready | missing_cli | missing_credentials`
before a long run. Reproduce this when vendoring; the example here uses the
caller's environment for simplicity.

## Budget

| Unit | Typical |
|---|---|
| one `Predict` call | 5–10 s, sequential |
| `Evaluate` on 20 examples | 2–4 min with `num_threads=1` |
| `GEPA(auto="light")`, 20 train / 10 val | a few hundred calls; run in the background with `max_metric_calls` |
| `BestOfN(N=3)` | 3 calls per prediction |

Cost is on the subscription; `total_cost_usd` in the JSON is informational.

## Failure modes

| Symptom | Cause | Fix |
|---|---|---|
| `claude exited 1: Not logged in` | stale CLI session | `claude login` on the machine; run `--probe` |
| `TimeoutExpired` | long system prompt plus slow model | raise `timeout_seconds`; prefer `claude/haiku` for workers |
| adapter parse error | the model answered outside DSPy's field markers | keep signatures typed with Pydantic outputs; DSPy's JSON adapter retries once |
| identical `BestOfN` candidates | no temperature | strengthen the instruction contrast between candidates or accept the first |
