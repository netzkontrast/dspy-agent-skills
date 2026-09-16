"""AST-parse every example_*.py to catch syntax errors offline (no LM required)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
EXAMPLES = sorted(REPO.glob("skills/*/example_*.py"))
SCRIPTS = sorted(REPO.glob("scripts/*.py"))
SCAFFOLDS = sorted(REPO.glob("scaffolding/*.py"))


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


@pytest.mark.parametrize("path", SCAFFOLDS, ids=lambda p: f"scaffolding/{p.name}")
def test_scaffold_parses(path: Path):
    """Scaffolds are inert but must stay importable-quality Python."""
    try:
        ast.parse(path.read_text(), filename=str(path))
    except SyntaxError as e:
        pytest.fail(f"{path}: {e}")


def test_scaffolds_are_documented():
    """Every scaffold must be referenced by a plan, or it is orphaned code."""
    docs = " ".join(p.read_text() for p in REPO.glob("docs/*.md"))
    for path in SCAFFOLDS:
        assert path.name in docs, (
            f"{path.name} is not referenced from docs/; a scaffold with no plan "
            f"is dead code."
        )
