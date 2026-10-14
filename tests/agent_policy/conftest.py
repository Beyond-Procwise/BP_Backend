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
    config.addinivalue_line("markers", "people: keep remembering the caller of a write live (people.remember)")


@pytest.fixture(autouse=True)
def _no_conflict_detection_after_save(request, monkeypatch):
    if request.node.get_closest_marker("conflict_detection") is None:
        monkeypatch.setattr(conflict_cases, "after_save", lambda conn, key: None)


@pytest.fixture(autouse=True)
def _suggestion_notices_are_removed(monkeypatch):
    """The learning scan tells a policy's owner when it raises a card (stage 5b). On the live test
    database those notification rows are the test's own: each is found by its link (a suggestion id
    is never reused) and removed after the test. Nothing is read or written unless the scan told
    somebody."""
    import os

    from services.agent_policy import learning

    told = []
    real = learning._tell_owner

    def telling(cur, target, suggestion_id, *args, **kwargs):
        told.append(f"suggestion:{suggestion_id}")
        return real(cur, target, suggestion_id, *args, **kwargs)

    monkeypatch.setattr(learning, "_tell_owner", telling)
    yield
    if told and os.getenv("PROCWISE_TEST_LIVE_DB") == "1":
        from services.db import get_conn

        with get_conn() as conn:
            conn.cursor().execute("DELETE FROM proc.bp_policy_notification WHERE link = ANY(%s)", (told,))


@pytest.fixture(autouse=True)
def _nobody_is_remembered(request, monkeypatch):
    """Every agent-policy write through the router remembers the caller's email on a connection of
    its own (stage 5b, people.remember). The router tests drive hundreds of writes as made-up
    people, against scripted connections or the shared test database: remembering is a no-op here
    unless a test is marked ``people`` (the people tests drive it, and remove their own rows)."""
    if request.node.get_closest_marker("people") is None:
        from api.routers import agent_policies

        monkeypatch.setattr(agent_policies, "_remember", lambda principal: None)
