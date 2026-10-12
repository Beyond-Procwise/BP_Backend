"""The created-agent store against a real PostgreSQL.

Skipped unless PROCWISE_TEST_LIVE_DB is on: the statements here are written for
PostgreSQL (one statement changes the agent and records the version together),
and the in-memory stand-in the rest of the suite uses cannot run them.
"""
from __future__ import annotations

import os
import sys
import uuid

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

pytestmark = pytest.mark.skipif(
    os.environ.get("PROCWISE_TEST_LIVE_DB", "").lower() not in ("1", "true", "yes", "on"),
    reason="needs a real PostgreSQL (PROCWISE_TEST_LIVE_DB=1)",
)

from repositories import agent_definition_repo as repo
from services.db import get_conn


@pytest.fixture()
def slug():
    repo.ensure_schema()
    repo.invalidate()
    name = f"test-agent-{uuid.uuid4().hex[:10]}"
    yield name
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM proc.bp_agent_definition_version WHERE slug = %s", (name,))
        cur.execute("DELETE FROM proc.bp_agent_definition WHERE slug = %s", (name,))
        cur.close()
    repo.invalidate()


def _entry(slug, **extra):
    return {"slug": slug, "derived_from": "negotiation", "description": "v1", **extra}


def _live(slug):
    live, _ = repo.snapshot()
    return next((a for a in live if a["slug"] == slug), None)


def test_a_created_agent_is_stored_with_its_maker_and_a_first_version(slug):
    assert repo.create(slug, _entry(slug), derived_from="negotiation", by="sub-1") == 1

    assert _live(slug)["description"] == "v1"
    assert [(v["version"], v["change"], v["changed_by"]) for v in repo.history(slug)] == [
        (1, "created", "sub-1")
    ]


def test_a_live_slug_cannot_be_created_twice(slug):
    repo.create(slug, _entry(slug), derived_from="negotiation", by="sub-1")
    with pytest.raises(repo.AgentExists):
        repo.create(slug, _entry(slug, description="other"), derived_from="negotiation", by="sub-2")
    assert _live(slug)["description"] == "v1"
    assert len(repo.history(slug)) == 1            # the refused create recorded nothing


def test_an_update_keeps_what_the_agent_looked_like_before(slug):
    repo.create(slug, _entry(slug), derived_from="negotiation", by="sub-1")
    assert repo.update(slug, _entry(slug, description="v2"), by="sub-2") == 2

    assert _live(slug)["description"] == "v2"
    history = repo.history(slug)
    assert [(v["version"], v["change"], v["changed_by"], v["definition"]["description"])
            for v in history] == [(2, "updated", "sub-2", "v2"), (1, "created", "sub-1", "v1")]


def test_a_deleted_agent_leaves_the_catalogue_and_keeps_its_history(slug):
    repo.create(slug, _entry(slug), derived_from="negotiation", by="sub-1")
    repo.delete(slug, _entry(slug), derived_from="negotiation", by="sub-2")

    live, deleted = repo.snapshot()
    assert slug not in [a["slug"] for a in live]
    assert slug in deleted
    assert [v["change"] for v in repo.history(slug)] == ["deleted", "created"]


def test_a_deleted_agent_cannot_be_updated_or_deleted_again(slug):
    repo.create(slug, _entry(slug), derived_from="negotiation", by="sub-1")
    repo.delete(slug, _entry(slug), derived_from="negotiation", by="sub-1")
    with pytest.raises(repo.AgentNotFound):
        repo.update(slug, _entry(slug, description="v2"), by="sub-1")
    with pytest.raises(repo.AgentNotFound):
        repo.delete(slug, _entry(slug), derived_from="negotiation", by="sub-1")


def test_a_deleted_slug_can_be_created_again_and_its_history_carries_on(slug):
    repo.create(slug, _entry(slug), derived_from="negotiation", by="sub-1")
    repo.delete(slug, _entry(slug), derived_from="negotiation", by="sub-1")
    assert repo.create(slug, _entry(slug, description="again"), derived_from="negotiation",
                       by="sub-3") == 3

    assert _live(slug)["description"] == "again"
    assert [v["change"] for v in repo.history(slug)] == ["created", "deleted", "created"]


def test_an_agent_left_in_the_file_is_taken_in_once(slug):
    entry = _entry(slug, created_by="workspace")
    assert repo.import_from_file([entry]) == [slug]
    assert repo.import_from_file([{**entry, "description": "file changed"}]) == []

    assert _live(slug)["description"] == "v1"
    history = repo.history(slug)
    assert [(v["change"], v["changed_by"]) for v in history] == [("imported", None)]


def test_an_agent_deleted_here_is_not_taken_back_in_from_the_file(slug):
    repo.create(slug, _entry(slug), derived_from="negotiation", by="sub-1")
    repo.delete(slug, _entry(slug), derived_from="negotiation", by="sub-1")

    assert repo.import_from_file([_entry(slug)]) == []
    assert _live(slug) is None


def test_deleting_an_agent_that_only_the_file_had_hides_it(slug):
    repo.delete(slug, _entry(slug), derived_from="negotiation", by="sub-1")
    _, deleted = repo.snapshot()
    assert slug in deleted


def test_built_in_entries_are_never_taken_in(slug):
    assert repo.import_from_file([{"slug": slug, "description": "a built-in"}]) == []
    assert repo.history(slug) == []


def test_a_read_that_fails_serves_the_last_good_read(slug, monkeypatch):
    repo.create(slug, _entry(slug), derived_from="negotiation", by="sub-1")
    assert _live(slug) is not None                 # a good read is now held

    def broken():
        raise RuntimeError("database is down")

    monkeypatch.setattr(repo, "_read", broken)
    monkeypatch.setattr(repo, "CACHE_SECONDS", 0.0)   # force a fresh read
    assert _live(slug) is not None


def test_with_no_good_read_to_fall_back_on_the_failure_is_raised(slug, monkeypatch):
    repo.invalidate()

    def broken():
        raise RuntimeError("database is down")

    monkeypatch.setattr(repo, "_read", broken)
    with pytest.raises(RuntimeError):
        repo.snapshot()
