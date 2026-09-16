"""dspy-production — runnable smoke test.

Exercises the deployment surface this skill teaches: cache hardening, the two
save formats, a sampling usage callback built on BaseCallback, and the
streaming/async/batch entry points. The dry run asserts every signature and
runs the callback against synthetic LM events without an API key.

Usage:
    uv run python example_production.py --dry-run
    OPENAI_API_KEY=... uv run python example_production.py
"""

from __future__ import annotations

import argparse
import inspect
import os
import random
import time
from collections import deque

DEFAULT_SAMPLE_RATE = 0.1
FLUSH_EVERY = 100
MEMORY_MAX_ENTRIES = 1_000_000
DISK_SIZE_LIMIT_BYTES = 30_000_000_000


def harden_cache(cache_dir: str = ".cache/dspy", disk: bool = True) -> None:
    """Allowlist-mode deserialization and a project-local cache directory."""
    import dspy

    dspy.configure_cache(
        enable_disk_cache=disk,
        enable_memory_cache=True,
        disk_cache_dir=cache_dir,
        restrict_pickle=True,
    )


def build_usage_callback():
    """A production callback: sampled, buffered, non-blocking, redacting."""
    from dspy.utils.callback import BaseCallback

    class UsageCallback(BaseCallback):
        def __init__(self, sample_rate: float = DEFAULT_SAMPLE_RATE, rng=None) -> None:
            super().__init__()
            self.sample_rate = sample_rate
            self.rng = rng or random.Random(0)
            self.started: dict[str, float] = {}
            self.buffer: deque = deque(maxlen=FLUSH_EVERY)
            self.calls = self.errors = 0
            self.total_tokens = 0

        def on_lm_start(self, call_id, instance, inputs):
            # Sampling decided at start so start/end stay paired.
            if self.rng.random() < self.sample_rate:
                self.started[call_id] = time.perf_counter()

        def on_lm_end(self, call_id, outputs, exception=None):
            started = self.started.pop(call_id, None)   # pop: a failure cannot leak it
            if exception is not None:
                self.errors += 1
                return
            if started is None:
                return
            usage = (outputs or {}).get("usage", {}) if isinstance(outputs, dict) else {}
            tokens = usage.get("total_tokens", 0)
            self.calls += 1
            self.total_tokens += tokens
            # Buffer only; never write to a remote sink inside the call path.
            self.buffer.append({"latency_s": time.perf_counter() - started, "tokens": tokens})

        def metrics(self) -> dict[str, float]:
            latencies = [row["latency_s"] for row in self.buffer]
            return {
                "sampled_calls": self.calls,
                "errors": self.errors,
                "total_tokens": self.total_tokens,
                "avg_latency_s": sum(latencies) / len(latencies) if latencies else 0.0,
            }

    return UsageCallback


def build_stream(program):
    """Token streaming for one output field."""
    import dspy

    return dspy.streamify(
        program,
        stream_listeners=[dspy.streaming.StreamListener(signature_field_name="answer")],
    )


def run_batch(program, questions: list[str], num_threads: int = 8):
    """Batch throughput over independent inputs."""
    import dspy

    parallel = dspy.Parallel(num_threads=num_threads, return_failed_examples=True)
    return parallel([(program, {"question": q}) for q in questions])


def save_state_only(program, path: str) -> None:
    """The safe format: no code executes when this is loaded."""
    program.save(path, save_program=False)


def assert_api_surface() -> None:
    """Fail loudly if the production API this skill teaches has drifted."""
    import dspy
    from dspy.clients.base_lm import GLOBAL_HISTORY
    from dspy.utils.callback import BaseCallback

    cache = inspect.signature(dspy.configure_cache).parameters
    assert cache["restrict_pickle"].default is False, "restrict_pickle default changed"
    assert cache["memory_max_entries"].default == MEMORY_MAX_ENTRIES
    assert cache["disk_size_limit_bytes"].default == DISK_SIZE_LIMIT_BYTES
    assert {"enable_disk_cache", "enable_memory_cache", "disk_cache_dir", "safe_types"} <= set(cache)

    save = inspect.signature(dspy.Module.save).parameters
    assert save["save_program"].default is False, "save defaults to whole-program now"
    assert "modules_to_serialize" in save

    listener = inspect.signature(dspy.streaming.StreamListener.__init__).parameters
    assert listener["allow_reuse"].default is False, "allow_reuse default changed"
    assert "signature_field_name" in listener
    assert "stream_listeners" in inspect.signature(dspy.streamify).parameters
    assert inspect.signature(dspy.inspect_history).parameters["n"].default == 1

    for hook in ("on_lm_start", "on_lm_end", "on_module_start", "on_tool_end",
                 "on_evaluate_start", "on_compile_end", "on_adapter_parse_start"):
        assert hasattr(BaseCallback, hook), f"BaseCallback lost {hook}"
    assert isinstance(GLOBAL_HISTORY, list)
    for name in ("asyncify", "load", "Parallel"):
        assert hasattr(dspy, name), f"dspy.{name} is gone"


def dry_run() -> None:
    assert_api_surface()
    print("API surface OK: configure_cache, save, streamify, StreamListener, callbacks")

    harden_cache(cache_dir=".cache/dspy-dry-run", disk=False)
    print("cache hardened: restrict_pickle=True, disk cache off")

    # Drive the callback with synthetic events — no LM, no network.
    callback = build_usage_callback()(sample_rate=1.0)
    for i in range(5):
        callback.on_lm_start(f"call-{i}", None, {"question": "?"})
        callback.on_lm_end(f"call-{i}", {"usage": {"total_tokens": 100 + i}})
    callback.on_lm_start("boom", None, {})
    callback.on_lm_end("boom", None, exception=RuntimeError("provider 500"))

    metrics = callback.metrics()
    print(f"callback metrics: {metrics}")
    assert metrics["sampled_calls"] == 5 and metrics["errors"] == 1
    assert metrics["total_tokens"] == sum(range(100, 105))
    assert not callback.started, "start/end state leaked; pop on end"

    sampled = build_usage_callback()(sample_rate=0.0)
    sampled.on_lm_start("x", None, {})
    sampled.on_lm_end("x", {"usage": {"total_tokens": 999}})
    assert sampled.metrics()["sampled_calls"] == 0, "sample_rate=0 still recorded a call"
    print("sampling honored: 0.0 records nothing, 1.0 records everything")
    print("dry run OK")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Skip all LM calls")
    ap.add_argument("--model", default=os.environ.get("DSPY_MODEL", "openai/gpt-4o-mini"))
    args = ap.parse_args()
    if args.dry_run:
        dry_run()
        return
    import dspy

    harden_cache()
    monitor = build_usage_callback()(sample_rate=1.0)
    dspy.configure(lm=dspy.LM(args.model), track_usage=True, callbacks=[monitor])

    program = dspy.ChainOfThought("question -> answer")
    prediction = program(question="What is DSPy in one sentence?")
    print(f"answer: {prediction.answer}")
    print(f"usage: {prediction.get_lm_usage()}")
    print(f"callback: {monitor.metrics()}")

    dspy.inspect_history(n=1)
    save_state_only(program, "./artifacts/program.json")
    print("saved state-only artifact to ./artifacts/program.json")


if __name__ == "__main__":
    main()
