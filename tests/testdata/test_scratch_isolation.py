"""The integration tests must not destroy the built dataset.

test_schema.py clones with drop_first=True. Pointed at bp_testdb, one `pytest
tests/testdata/` run silently drops the database holding the built dataset and
recreates it empty -- the build still reports success, and the loss is only
noticed later when a table reads zero. This guard keeps the targets separate.
"""
from __future__ import annotations

import pathlib
import re

from tests.testdata import SCRATCH_DB, SCRATCH_UICANVAS_DB

DATASET_DATABASES = ("bp_testdb", "uicanvas_test")

# Naming the dataset databases is fine where it is an assertion about defaults
# rather than a connection: those tests never open a connection to them.
ALLOWED = {
    "test_build.py",       # asserts parse_args defaults
    "test_guards.py",      # asserts the guard permits these names
    "test_schema.py",      # asserts SCHEMA_PAIRS maps live -> dataset databases
    "test_snapshot.py",    # asserts snapshot.PAIRS and filenames; opens nothing
}

_TESTS = pathlib.Path(__file__).parent


def test_scratch_databases_are_not_the_dataset_databases():
    assert SCRATCH_DB not in DATASET_DATABASES
    assert SCRATCH_UICANVAS_DB not in DATASET_DATABASES


def test_no_test_connects_to_a_dataset_database():
    offenders: list[str] = []
    pattern = re.compile(
        r"(connect|copy_reference|clone_schema|create_database)\s*\([^)]*"
        r"[\"'](bp_testdb|uicanvas_test)[\"']"
    )
    for path in sorted(_TESTS.glob("test_*.py")):
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            if pattern.search(line):
                offenders.append(f"{path.name}:{number}: {line.strip()}")
    assert not offenders, (
        "integration tests must target the scratch databases, not the built "
        "dataset:\n" + "\n".join(offenders)
    )


def test_dataset_database_names_appear_only_in_assertions():
    """A bare mention is allowed only in the files that assert on the names."""
    offenders: list[str] = []
    for path in sorted(_TESTS.glob("test_*.py")):
        if path.name in ALLOWED or path.name == pathlib.Path(__file__).name:
            continue
        text = path.read_text()
        for name in DATASET_DATABASES:
            if f'"{name}"' in text or f"'{name}'" in text:
                offenders.append(f"{path.name} mentions {name}")
    assert not offenders, "\n".join(offenders)
