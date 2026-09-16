"""dspy-book-datasets — runnable smoke test.

The dataset rules that decide whether an optimizer run means anything:
seeded reproducible splits, disjointness, difficulty stratification, and the
with_inputs contract. Also encodes the leakage checks the chapter omits.
No LM, no network, no dspy import required for the dry run.

Usage:
    uv run python example_dataset_builder.py --dry-run
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass, field

SEED = 2024
TRAIN_FRACTION, VAL_FRACTION = 0.5, 0.25
TIERS = ("clear", "boundary", "ambiguous")
LARGE_CORPUS_SIZES = {"train": 200, "validation": 100, "test": 500}


@dataclass
class Example:
    """Stand-in for dspy.Example: the inputs/gold split is the whole contract."""

    fields: dict
    input_keys: tuple[str, ...] = ()
    tier: str = "clear"
    notes: str = ""                       # the mechanism of failure -> optimizer feedback

    def with_inputs(self, *keys: str) -> "Example":
        missing = [k for k in keys if k not in self.fields]
        if missing:
            raise KeyError(f"with_inputs names fields that do not exist: {missing}")
        self.input_keys = keys
        return self

    @property
    def gold_keys(self) -> tuple[str, ...]:
        return tuple(k for k in self.fields if k not in self.input_keys)

    def uid(self) -> str:
        return str(sorted(self.fields.items()))


def convert_labelled_row(row: dict, label_map: dict[int, str]) -> Example:
    """Coerce types and map labels HERE, never inside the metric."""
    return Example({"text": row["text"], "sentiment": label_map[row["label"]]}).with_inputs("text")


def split(examples: list[Example], seed: int = SEED) -> tuple[list, list, list]:
    """Seeded and deterministic, or no two runs are comparable."""
    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    train_end = int(TRAIN_FRACTION * n)
    val_end = train_end + int(VAL_FRACTION * n)
    return shuffled[:train_end], shuffled[train_end:val_end], shuffled[val_end:]


def splits_are_disjoint(train: list[Example], val: list[Example],
                        test: list[Example]) -> tuple[bool, str]:
    """The check chapter 4 never makes and chapter 3 actually violates."""
    t, v, s = {e.uid() for e in train}, {e.uid() for e in val}, {e.uid() for e in test}
    for a, b, names in ((t, v, "train/val"), (t, s, "train/test"), (v, s, "val/test")):
        if a & b:
            return False, f"{names} overlap on {len(a & b)} example(s): leakage"
    return True, "train, validation and test are disjoint"


def tier_coverage(examples: list[Example]) -> dict[str, int]:
    counts = {tier: 0 for tier in TIERS}
    for example in examples:
        if example.tier not in counts:
            raise ValueError(f"unknown difficulty tier {example.tier!r}")
        counts[example.tier] += 1
    return counts


def every_tier_represented(examples: list[Example]) -> bool:
    """Stratify so each failure mode from error analysis appears."""
    return all(count > 0 for count in tier_coverage(examples).values())


def ambiguous_belong_in_a_log(examples: list[Example]) -> list[Example]:
    """Ambiguous cases mark an incomplete task definition, not a model error."""
    return [e for e in examples if e.tier == "ambiguous"]


def synthetic_batch_is_sound(requested: int, returned: int, unique: int) -> tuple[bool, str]:
    """n>1 is unsupported on some endpoints; zip() truncates silently."""
    if returned < requested:
        return False, (f"asked for {requested} completions, got {returned}; "
                       f"the endpoint ignored n and zip() would truncate")
    if unique < returned:
        return False, f"{returned - unique} duplicate generations; add a dedup pass"
    return True, f"{returned} distinct completions"


def dry_run() -> None:
    rows = [{"text": f"review {i}", "label": i % 3} for i in range(40)]
    label_map = {0: "negative", 1: "neutral", 2: "positive"}
    examples = [convert_labelled_row(r, label_map) for r in rows]
    for i, example in enumerate(examples):
        example.tier = TIERS[i % len(TIERS)]
    print(f"converted {len(examples)}; inputs={examples[0].input_keys} "
          f"gold={examples[0].gold_keys}")
    assert examples[0].input_keys == ("text",) and examples[0].gold_keys == ("sentiment",)

    try:
        Example({"text": "x"}).with_inputs("question")
    except KeyError:
        print("with_inputs rejects a field that does not exist")
    else:
        raise AssertionError("with_inputs must reject unknown fields")

    train, val, test = split(examples)
    print(f"split sizes: train={len(train)} val={len(val)} test={len(test)}")
    ok, why = splits_are_disjoint(train, val, test)
    print(f"disjoint: {ok} | {why}")
    assert ok

    again = split(examples)
    assert [e.uid() for e in again[0]] == [e.uid() for e in train], (
        "a seeded split must reproduce exactly"
    )
    different = split(examples, seed=SEED + 1)
    assert [e.uid() for e in different[0]] != [e.uid() for e in train], (
        "a different seed must give a different split"
    )
    print("seeded split reproduces; a new seed does not")

    leaky_val = train[:2] + val
    ok, why = splits_are_disjoint(train, leaky_val, test)
    print(f"leaky split detected: {not ok} | {why}")
    assert not ok, "overlap must be caught"

    print(f"tier coverage: {tier_coverage(examples)}")
    assert every_tier_represented(examples)
    print(f"ambiguous cases routed to a log, not to training: "
          f"{len(ambiguous_belong_in_a_log(examples))}")

    for requested, returned, unique in ((15, 15, 15), (15, 8, 8), (15, 15, 11)):
        ok, why = synthetic_batch_is_sound(requested, returned, unique)
        print(f"n={requested} returned={returned} unique={unique} -> {ok}: {why}")
    assert not synthetic_batch_is_sound(15, 8, 8)[0]
    assert not synthetic_batch_is_sound(15, 15, 11)[0]

    print(f"large-corpus sizes the chapter states: {LARGE_CORPUS_SIZES}")
    print("dry run OK")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Run without an LM")
    ap.parse_args()
    dry_run()


if __name__ == "__main__":
    main()
