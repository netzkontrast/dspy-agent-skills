"""AST-parse every example_*.py to catch syntax errors offline (no LM required)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
EXAMPLES = sorted(REPO.glob("skills/*/example_*.py"))
SCRIPTS = sorted(REPO.glob("scripts/*.py"))


def test_found_examples():
    assert EXAMPLES, "Expected at least one example_*.py under skills/"


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_example_parses(path: Path):
    src = path.read_text()
    try:
        ast.parse(src, filename=str(path))
    except SyntaxError as e:
        pytest.fail(f"{path}: {e}")


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: f"scripts/{p.name}")
def test_script_parses(path: Path):
    src = path.read_text()
    try:
        ast.parse(src, filename=str(path))
    except SyntaxError as e:
        pytest.fail(f"{path}: {e}")


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_example_has_dry_run(path: Path):
    src = path.read_text()
    assert "--dry-run" in src, (
        f"{path.name} should expose a --dry-run flag so it can be smoke-tested "
        f"without LM credentials."
    )
