"""dspy-book-production — runnable smoke test.

The serving decisions that decide whether a deployed DSPy program is sound:
load-once versus per-request, worker sizing from rate limits, typed-only
fallback, and the trace-field rule that keeps observability from flooding a
context window. No LM, no server, no network.

Usage:
    uv run python example_serving_checks.py --dry-run
"""

from __future__ import annotations

import argparse

DEFAULT_ASYNC_WORKERS = 4
SECONDS_PER_MINUTE = 60
TRACE_FIELD_PATHS = ("info.trace_id", "info.state", "data.spans.*.name")

# Exceptions a fallback may catch. Anything else is your bug, not the provider's.
FALLBACK_SAFE = {"APIError", "RateLimitError", "ServiceUnavailableError", "Timeout"}


def program_load_is_sound(*, loaded_in_lifespan: bool, artifact_exists: bool,
                          fails_hard_when_missing: bool) -> tuple[bool, str]:
    if not loaded_in_lifespan:
        return False, "program built per request; load once at startup and stash on app state"
    if not artifact_exists and not fails_hard_when_missing:
        return False, ("compiled artifact missing and startup only warns; "
                       "a service silently serving an uncompiled program is worse than one that refuses to boot")
    return True, "loaded once at startup, missing artifact is fatal"


def suggested_workers(requests_per_minute_limit: int, mean_latency_s: float) -> int:
    """Size from the provider's rate limit, not the CPU count."""
    if requests_per_minute_limit <= 0 or mean_latency_s <= 0:
        raise ValueError("limit and latency must be positive")
    in_flight = (requests_per_minute_limit / SECONDS_PER_MINUTE) * mean_latency_s
    return max(1, min(int(in_flight), requests_per_minute_limit))


def fallback_is_safe(caught: set[str]) -> tuple[bool, str]:
    if "Exception" in caught or "BaseException" in caught:
        return False, "bare Exception turns your own bugs into a silent downgrade to a weaker model"
    unknown = caught - FALLBACK_SAFE
    if unknown:
        return False, f"{sorted(unknown)} are not provider failures; catch typed errors only"
    return True, "typed provider failures only"


def guardrail_layers(*, input_validator: bool, optimizable_field: bool,
                     output_regex: bool) -> tuple[int, list[str]]:
    """Three layers; the middle one is the idea worth taking."""
    missing = []
    if not input_validator:
        missing.append("input bounds (Pydantic field_validator)")
    if not optimizable_field:
        missing.append("an optimizable signature output field, so the optimizer improves the guard")
    if not output_regex:
        missing.append("a deterministic output check returning 422")
    return 3 - len(missing), missing


def trace_query_is_safe(extract_fields: tuple[str, ...] | None) -> tuple[bool, str]:
    """Without a field list, the default full span tree exhausts the context window."""
    if not extract_fields:
        return False, ("no extract_fields: the default returns the full span tree and "
                       "will exhaust the context window after a few results")
    return True, f"scoped to {len(extract_fields)} field path(s)"


def adapter_export_is_consistent(optimize_adapter: str, export_adapter: str) -> tuple[bool, str]:
    if optimize_adapter != export_adapter:
        return False, (f"optimized under {optimize_adapter} but exporting as {export_adapter}; "
                       f"demos selected under one format render in another")
    return True, f"same adapter ({optimize_adapter}) for optimization and export"


def dry_run() -> None:
    for label, kwargs in (
        ("per-request build", dict(loaded_in_lifespan=False, artifact_exists=True, fails_hard_when_missing=True)),
        ("warn on missing", dict(loaded_in_lifespan=True, artifact_exists=False, fails_hard_when_missing=False)),
        ("correct", dict(loaded_in_lifespan=True, artifact_exists=True, fails_hard_when_missing=True)),
    ):
        ok, why = program_load_is_sound(**kwargs)
        print(f"{label:18s} sound={str(ok):5s} | {why[:72]}")
    assert not program_load_is_sound(loaded_in_lifespan=False, artifact_exists=True,
                                     fails_hard_when_missing=True)[0]

    for rpm, latency in ((60, 2.0), (600, 2.0), (3000, 1.5)):
        print(f"rate limit {rpm:>5}/min, latency {latency}s -> ~{suggested_workers(rpm, latency)} workers "
              f"(default start: {DEFAULT_ASYNC_WORKERS})")
    assert suggested_workers(600, 2.0) > suggested_workers(60, 2.0)

    for caught in ({"RateLimitError", "Timeout"}, {"Exception"}, {"ValueError"}):
        ok, why = fallback_is_safe(caught)
        print(f"catching {sorted(caught)} -> {ok}: {why[:66]}")
    assert fallback_is_safe({"RateLimitError", "Timeout"})[0]
    assert not fallback_is_safe({"Exception"})[0]

    count, missing = guardrail_layers(input_validator=True, optimizable_field=False, output_regex=True)
    print(f"guardrail layers present: {count}/3; missing: {missing}")
    assert count == 2 and "optimizable" in missing[0]
    assert guardrail_layers(input_validator=True, optimizable_field=True, output_regex=True)[0] == 3

    ok, why = trace_query_is_safe(None)
    print(f"unscoped trace search safe: {ok} | {why[:70]}")
    assert not ok
    assert trace_query_is_safe(TRACE_FIELD_PATHS)[0]

    ok, why = adapter_export_is_consistent("ChatAdapter", "MarkdownAdapter")
    print(f"adapter switch at export ok: {ok} | {why[:70]}")
    assert not ok
    assert adapter_export_is_consistent("MarkdownAdapter", "MarkdownAdapter")[0]
    print("dry run OK")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Run without a server or LM")
    ap.parse_args()
    dry_run()


if __name__ == "__main__":
    main()
