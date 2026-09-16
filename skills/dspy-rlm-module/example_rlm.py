"""dspy.RLM — Recursive Language Model smoke test.

The RLM writes and runs Python in a sandboxed Pyodide/WASM REPL (spawned by
Deno) to explore inputs too large for a single prompt. This script only
constructs an RLM and exercises its signature offline; the live run requires
both Deno installed and a valid LM/API key.

Usage:
    uv run python example_rlm.py --dry-run
    OPENAI_API_KEY=... uv run python example_rlm.py
"""

from __future__ import annotations

import argparse
import os


def build_rlm(sub_model: str | None = None):
    import dspy

    # The signature declares a "context" input (treated as a Python variable
    # inside the REPL) and a "query" input; the RLM returns an "answer".
    sub_lm = dspy.LM(sub_model) if sub_model else None
    return dspy.RLM(
        "context, query -> answer",
        max_iters=10,
        max_llm_calls=20,
        max_output_chars=10_000,
        sub_lm=sub_lm,
        verbose=False,
    )


def assert_rlm_surface() -> None:
    """Pin the dspy.RLM constructor names this skill teaches.

    DSPy 3.3.0 renamed ``max_iterations`` to ``max_iters`` and replaced the
    ``interpreter`` constructor argument with an ``interpreter_factory``,
    moving a caller-owned ``interpreter`` to ``forward``. Asserting the names
    here means a future rename fails the smoke test rather than a user's run.
    """
    import inspect

    import dspy

    init = inspect.signature(dspy.RLM.__init__).parameters
    for name in ("signature", "max_iters", "max_llm_calls", "max_output_chars",
                 "verbose", "tools", "sub_lm", "interpreter_factory"):
        assert name in init, f"dspy.RLM.__init__ lost parameter {name!r}"
    assert "max_iterations" not in init, (
        "dspy.RLM re-introduced max_iterations; revisit this skill"
    )
    forward = inspect.signature(dspy.RLM.forward).parameters
    assert "interpreter" in forward, "dspy.RLM.forward lost its interpreter argument"


SAMPLE_CONTEXT = """\
Server log excerpt (abbreviated):
  [2026-04-18T11:04:02Z] WARN  RateLimit exceeded for user=42
  [2026-04-18T11:04:05Z] ERROR DatabaseConnection timeout
  [2026-04-18T11:04:18Z] WARN  RateLimit exceeded for user=7
  [2026-04-18T11:04:19Z] INFO  GC ran 400ms pause
  [2026-04-18T11:04:33Z] ERROR DatabaseConnection timeout
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Skip all LM calls")
    ap.add_argument("--model", default=os.getenv("DSPY_MODEL", "openai/gpt-4o"))
    ap.add_argument("--sub-model", default=os.getenv("DSPY_SUB_MODEL"))
    ap.add_argument(
        "--query",
        default="Count the unique error classes and how many times each appears.",
    )
    args = ap.parse_args()

    import dspy

    rlm = build_rlm(sub_model=args.sub_model)
    if args.dry_run:
        # Verify signature fields wire up correctly; no LM/Deno interaction.
        assert_rlm_surface()
        sig_in = sorted(rlm.signature.input_fields.keys())
        sig_out = sorted(rlm.signature.output_fields.keys())
        print("OK: dspy.RLM(signature='context, query -> answer') constructed.")
        print(f"    inputs:  {sig_in}")
        print(f"    outputs: {sig_out}")
        print(
            "    "
            f"max_iters={rlm.max_iters} "
            f"max_llm_calls={rlm.max_llm_calls} "
            f"max_output_chars={rlm.max_output_chars}"
        )
        return 0

    dspy.configure(lm=dspy.LM(args.model), track_usage=True)
    result = rlm(context=SAMPLE_CONTEXT, query=args.query)
    print("Answer:", result.answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
