"""The run record against a real PostgreSQL. Skipped unless PROCWISE_TEST_LIVE_DB is on."""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

pytestmark = pytest.mark.skipif(
    os.environ.get("PROCWISE_TEST_LIVE_DB", "").lower() not in ("1", "true", "yes", "on"),
    reason="needs a real PostgreSQL (PROCWISE_TEST_LIVE_DB=1)",
)

from repositories import agent_run_repo as repo
from services.db import get_conn


def ToolCall(**kw):
    """A tool call as the loop records it. Not imported from the loop: that
    module reads governed limits as it loads, and this database has none."""
    from types import SimpleNamespace

    return SimpleNamespace(error=None, duration_ms=12, **kw)


@pytest.fixture()
def slug():
    repo.ensure_schema()
    name = f"test-agent-{repo.new_run_id()[:10]}"
    yield name
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM proc.bp_agent_run WHERE agent_slug = %s", (name,))
        cur.close()


def _record(slug, run_id, *, at, error=None):
    repo.record(
        run_id=run_id, agent_slug=slug, workflow_id="wf-9", user_id="u1",
        model="BeyondProcwise/AgentNick:unified", system_prompt="You are the overcharge hunter.",
        task="Deal D1", tools_offered=["get_corpus_facts", "get_policy"],
        calls=[ToolCall(name="get_corpus_facts", arguments={"query": "overbilling"}, ok=True,
                        result={"rows": [{"supplier": "Acme", "over": 812.5,
                                          "at": datetime(2026, 10, 1)}]})],
        answer="Acme billed 812.50 above its PO.", rounds=2, error=error,
        started_at=at, finished_at=at + timedelta(seconds=40),
    )


def test_a_recorded_run_reads_back_whole(slug):
    run_id = repo.new_run_id()
    _record(slug, run_id, at=datetime(2026, 10, 12, 9, 0))

    run = repo.get(run_id)
    assert run["agent_slug"] == slug
    assert run["model"] == "BeyondProcwise/AgentNick:unified"
    assert run["system_prompt"] == "You are the overcharge hunter."
    assert len(run["system_sha256"]) == 64
    assert run["tools_offered"] == ["get_corpus_facts", "get_policy"]
    (call,) = run["calls"]
    assert call["arguments"] == {"query": "overbilling"}
    assert "Acme" in call["result"] and call["result_truncated"] is False
    assert run["answer"] == "Acme billed 812.50 above its PO."


def test_an_agents_runs_are_listed_newest_first_without_their_contents(slug):
    first, second = repo.new_run_id(), repo.new_run_id()
    _record(slug, first, at=datetime(2026, 10, 12, 9, 0))
    _record(slug, second, at=datetime(2026, 10, 12, 10, 0), error="ollama unreachable")

    runs = repo.list_for_agent(slug)
    assert [r["run_id"] for r in runs] == [second, first]
    assert runs[0]["error"] == "ollama unreachable"
    assert "system_prompt" not in runs[0] and "calls" not in runs[0]


def test_a_run_that_was_never_recorded_is_not_found(slug):
    assert repo.get(repo.new_run_id()) is None
