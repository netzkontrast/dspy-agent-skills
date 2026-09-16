"""dspy-local-runtime — runnable smoke test.

Run DSPy programs through the local Claude Code CLI instead of an API key:
a minimal ``dspy.BaseLM`` subclass that turns every request into one
``claude -p --output-format json`` process, backend selection between an API
LM and the CLI, and the kwargs the CLI cannot honour (temperature, max_tokens,
rollout_id, n > 1, cache). Pattern from Hmbown/dspy-local (MIT); this file is
a teaching-sized re-implementation, not the vendored package (which also
isolates HOME and validates every kwarg). The dry run exercises prompt
building, result parsing, kwarg stripping and backend selection without
spawning anything; ``--probe`` makes one real call when ``claude`` is on PATH.

Usage:
    uv run python example_local_runtime.py --dry-run
    uv run python example_local_runtime.py --probe
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from types import SimpleNamespace
from typing import Any, Callable

import dspy

STRIPPED_KWARGS = ("temperature", "max_tokens", "rollout_id")
DEFAULT_TIMEOUT_SECONDS = 120
BACKENDS = ("api", "claude-cli", "auto")


# --- pure helpers (tested in the dry run) ------------------------------------------

def build_prompt(prompt: str | None = None, messages: list[dict[str, Any]] | None = None) -> tuple[str | None, str]:
    """DSPy hands either a prompt or chat messages; the CLI takes one system prompt and one user text."""
    if messages is None:
        return None, prompt or ""
    system = "\n\n".join(str(m["content"]) for m in messages if m.get("role") == "system") or None
    turns = [str(m["content"]) if m.get("role") == "user" else f"{m.get('role')}: {m['content']}"
             for m in messages if m.get("role") != "system"]
    return system, "\n\n".join(turns)


def build_command(alias: str, system: str | None, permission_mode: str = "plan") -> list[str]:
    cmd = ["claude", "-p", "--output-format", "json", "--permission-mode", permission_mode, "--no-session-persistence"]
    if alias != "default":
        cmd += ["--model", alias]
    if system:
        cmd += ["--system-prompt", system]
    return cmd


def parse_result(stdout: str) -> tuple[str, dict[str, int]]:
    payload = json.loads(stdout)
    if payload.get("is_error"):
        raise RuntimeError(f"claude returned an error: {payload.get('result')!r}")
    usage = payload.get("usage") or {}
    prompt_tokens = int(usage.get("input_tokens", 0))
    completion_tokens = int(usage.get("output_tokens", 0))
    return str(payload.get("result", "")), {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                                             "total_tokens": prompt_tokens + completion_tokens}


def select_backend(env: dict[str, str], which: Callable[[str], str | None] = shutil.which) -> str:
    """Explicit DSPY_LOCAL_BACKEND wins; otherwise the API when a key exists, else the CLI when installed."""
    chosen = env.get("DSPY_LOCAL_BACKEND", "auto")
    if chosen not in BACKENDS:
        raise ValueError(f"DSPY_LOCAL_BACKEND must be one of {BACKENDS}, got {chosen!r}")
    if chosen != "auto":
        return chosen
    if env.get("ANTHROPIC_API_KEY") or env.get("OPENAI_API_KEY"):
        return "api"
    return "claude-cli" if which("claude") else "api"


# --- the LM ----------------------------------------------------------------------

class ClaudeLM(dspy.BaseLM):
    """One ``claude -p`` process per request. Model strings are ``claude/<alias>`` (haiku, sonnet, opus, default)."""

    def __init__(self, model: str = "claude/default", *, permission_mode: str = "plan",
                 timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS, cache: bool = False, **kwargs: Any):
        if cache:
            raise ValueError("ClaudeLM cannot cache: the CLI has no deterministic sampling to cache against")
        if not model.startswith("claude/"):
            raise ValueError(f"model must be 'claude/<alias>', got {model!r}")
        super().__init__(model=model, model_type="chat", temperature=None, max_tokens=None, cache=False, **kwargs)
        self.kwargs = {k: v for k, v in self.kwargs.items() if k not in STRIPPED_KWARGS}
        self.alias = model.split("/", 1)[1]
        self.permission_mode = permission_mode
        self.timeout_seconds = timeout_seconds

    def copy(self, **kwargs: Any) -> "ClaudeLM":
        """`lm.copy(rollout_id=…, temperature=…)` is what BestOfN/Refine and TetraFrame call: strip silently."""
        for key in STRIPPED_KWARGS:
            kwargs.pop(key, None)
        new = super().copy(**kwargs)
        new.kwargs = {k: v for k, v in new.kwargs.items() if k not in STRIPPED_KWARGS}
        return new

    def forward(self, prompt: str | None = None, messages: list[dict[str, Any]] | None = None, **kwargs: Any):
        if kwargs.get("n", 1) not in (None, 1):
            raise ValueError("ClaudeLM returns one completion per call; use dspy.BestOfN for candidates")
        system, user = build_prompt(prompt, messages)
        proc = subprocess.run(build_command(self.alias, system, self.permission_mode), input=user,
                              capture_output=True, text=True, timeout=self.timeout_seconds, check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"claude exited {proc.returncode}: {proc.stderr.strip()[:300]}")
        text, usage = parse_result(proc.stdout)
        if dspy.settings.usage_tracker:
            dspy.settings.usage_tracker.add_usage(self.model, dict(usage))
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
                               usage=usage, model=self.model)


def build_lm(role_model: str, backend: str) -> dspy.BaseLM:
    """`role_model` is an API model id for the API backend and a `claude/<alias>` for the CLI."""
    return ClaudeLM(role_model) if backend == "claude-cli" else dspy.LM(role_model)


# --- smoke test ------------------------------------------------------------------

CANNED = json.dumps({"type": "result", "result": "Deutsch", "usage": {"input_tokens": 12, "output_tokens": 1}, "is_error": False})


def dry_run() -> None:
    system, user = build_prompt(messages=[{"role": "system", "content": "Antworte knapp."},
                                          {"role": "user", "content": "Welche Sprache ist das?"}])
    assert system == "Antworte knapp." and user == "Welche Sprache ist das?"
    assert build_command("haiku", system)[-4:] == ["--model", "haiku", "--system-prompt", "Antworte knapp."]
    assert "--model" not in build_command("default", None)
    text, usage = parse_result(CANNED)
    assert text == "Deutsch" and usage == {"prompt_tokens": 12, "completion_tokens": 1, "total_tokens": 13}
    lm = ClaudeLM("claude/haiku")
    assert "temperature" not in lm.kwargs and "max_tokens" not in lm.kwargs and lm.cache is False
    copied = lm.copy(rollout_id=3, temperature=0.9)
    assert isinstance(copied, ClaudeLM) and "rollout_id" not in copied.kwargs and "temperature" not in copied.kwargs
    for bad in (dict(cache=True), dict(model="anthropic/claude-haiku-4-5")):
        try:
            ClaudeLM(**{"model": "claude/haiku", **bad})
            raise AssertionError(f"expected rejection for {bad}")
        except ValueError:
            pass
    which_yes, which_no = (lambda _: "/usr/bin/claude"), (lambda _: None)
    assert select_backend({"ANTHROPIC_API_KEY": "sk"}, which_yes) == "api"
    assert select_backend({}, which_yes) == "claude-cli" and select_backend({}, which_no) == "api"
    assert select_backend({"DSPY_LOCAL_BACKEND": "claude-cli", "ANTHROPIC_API_KEY": "sk"}, which_no) == "claude-cli"
    assert isinstance(build_lm("claude/opus", "claude-cli"), ClaudeLM) and isinstance(build_lm("openai/gpt-4o", "api"), dspy.LM)
    print("OK: ClaudeLM builds prompts and commands, parses results, strips sampling kwargs, refuses cache; backend selection resolves")


def probe() -> int:
    if not shutil.which("claude"):
        print("SKIP: no `claude` binary on PATH")
        return 0

    class Language(dspy.Signature):
        """Name the language of the sentence in one word."""

        sentence: str = dspy.InputField()
        language: str = dspy.OutputField()

    dspy.configure(lm=ClaudeLM("claude/haiku"), track_usage=True)
    pred = dspy.Predict(Language)(sentence="Der Schleier hält bis Kapitel 13.")
    print(f"PROBE OK: language={pred.language!r} usage={dspy.settings.lm.history[-1].get('usage')}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="No subprocess, no LM call")
    ap.add_argument("--probe", action="store_true", help="One real call through the claude CLI")
    args = ap.parse_args()
    if args.probe:
        return probe()
    dry_run()
    if not args.dry_run:
        print(f"backend for this shell: {select_backend(dict(os.environ))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
