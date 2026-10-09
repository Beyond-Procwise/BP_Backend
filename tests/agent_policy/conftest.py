"""Agent-policy test setup.

Conflict detection after a save scans EVERY non-retired policy in the database. bp_testdb is
shared and holds thousands of never-retired test drafts from earlier runs, many of which
contradict each other (same tools, different outcomes, different "Run test <hex>" sources), so
leaving the hook live in tests that save through the router or an extraction run raises
thousands of real cases there and makes the suite crawl. It is therefore a no-op here unless a
test is marked ``conflict_detection``; the conflict tests drive detection explicitly, narrowed
with ``among=``.
"""
import pytest

from services.agent_policy import conflict_cases


def pytest_configure(config):
    config.addinivalue_line("markers", "conflict_detection: keep conflict detection after a save live")


@pytest.fixture(autouse=True)
def _no_conflict_detection_after_save(request, monkeypatch):
    if request.node.get_closest_marker("conflict_detection") is None:
        monkeypatch.setattr(conflict_cases, "after_save", lambda conn, key: None)
