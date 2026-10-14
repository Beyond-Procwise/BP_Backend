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


def _record(slug, run_id, *, at, error=None, user="u1", tokens=(2300, 180, 3)):
    repo.record(
        run_id=run_id, agent_slug=slug, workflow_id="wf-9", user_id=user,
        prompt_tokens=tokens[0], completion_tokens=tokens[1], model_calls=tokens[2],
        model="BeyondProcwise/AgentNick:unified", system_prompt="You are the overcharge hunter.",
        task="Deal D1", tools_offered=["get_corpus_facts", "get_policy"],
        calls=[ToolCall(name="get_corpus_facts", arguments={"query": "overbilling"}, ok=True,
                        result={"rows": [{"supplier": "Acme", "over": 812.5,
                                          "at": datetime(2026, 10, 1)}]})],
        answer="Acme billed 812.50 above its PO.", rounds=2, error=error,
        outputs={"supplier": {"value": "Acme", "status": "evidence_found",
                              "evidence": '"supplier": "Acme"'}},
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
    assert run["outputs"]["supplier"]["status"] == "evidence_found"


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


def test_usage_is_summed_by_agent_model_and_person_for_a_period(slug):
    _record(slug, repo.new_run_id(), at=datetime(2026, 9, 30, 23, 59))          # before
    _record(slug, repo.new_run_id(), at=datetime(2026, 10, 1, 0, 0))
    _record(slug, repo.new_run_id(), at=datetime(2026, 10, 20, 9, 0), tokens=(700, 20, 1))
    _record(slug, repo.new_run_id(), at=datetime(2026, 10, 20, 9, 0), user="u2",
            tokens=(50, 5, 1))
    _record(slug, repo.new_run_id(), at=datetime(2026, 11, 1, 0, 0))            # after

    mine = [r for r in repo.usage_summary(datetime(2026, 10, 1), datetime(2026, 11, 1))
            if r["agent_slug"] == slug]

    assert mine == [
        {"agent_slug": slug, "model": "BeyondProcwise/AgentNick:unified", "user_id": "u1",
         "runs": 2, "model_calls": 4, "prompt_tokens": 3000, "completion_tokens": 200},
        {"agent_slug": slug, "model": "BeyondProcwise/AgentNick:unified", "user_id": "u2",
         "runs": 1, "model_calls": 1, "prompt_tokens": 50, "completion_tokens": 5},
    ]


def test_a_recorded_run_carries_what_it_cost(slug):
    run_id = repo.new_run_id()
    _record(slug, run_id, at=datetime(2026, 10, 12, 9, 0))
    run = repo.get(run_id)
    assert (run["prompt_tokens"], run["completion_tokens"], run["model_calls"]) == (2300, 180, 3)
