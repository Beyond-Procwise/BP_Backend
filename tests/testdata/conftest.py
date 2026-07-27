"""Shared setup for the test-data integration tests.

The scratch databases are created empty, so anything that writes into `proc`
needs the structure cloned first. Doing it once per session keeps the ~15s
pg_dump off every individual test.
"""
from __future__ import annotations

import pytest

from tests.testdata import SCRATCH_DB


@pytest.fixture(scope="session")
def scratch_schema():
    """Clone bp_sqldb's structure into the scratch database, once per session."""
    from scripts.testdata.schema import clone_schema

    clone_schema("bp_sqldb", SCRATCH_DB, drop_first=True)
    return SCRATCH_DB
