"""POST /agents/reason stores its tool trace in proc.bp_agent_run (audit rec 10).

The route already returned the full trace to the caller; nothing kept it, so an answer given
through this route could not be checked afterwards: which tools ran, with what arguments,
what they returned, for whom. A record that cannot be written must not fail the run.
"""
import asyncio
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api.routers import agents as agents_router
from repositories import agent_run_repo
from services.tool_runtime import ToolCall, ToolRunResult


def _result():
    return ToolRunResult(
        answer="Acme billed 812.50 above its PO.", rounds=2, model="AgentNick:unified",
        system="You are AgentNick.", tools_offered=["get_corpus_facts", "get_policy"],
        calls=[ToolCall(name="get_corpus_facts", arguments={"query": "overbilling"}, ok=True,
                        result={"rows": [{"supplier": "Acme"}]}, error=None, duration_ms=12)])


class _Nick:
    def __init__(self, result=None, boom=None):
        self._result, self._boom, self.asked = result, boom, []

    def reason(self, task, **kw):
        self.asked.append((task, kw))
        if self._boom:
            raise self._boom
        return self._result


def _call(nick, task="Which suppliers over-billed?", user="u-42"):
    req = agents_router.ReasonRequest(task=task)
    return asyncio.run(agents_router.agent_nick_reason(
        req, agent_nick=nick, principal=SimpleNamespace(subject=user)))


@pytest.fixture
def recorded(monkeypatch):
    rows = []
    # About recording the trace, not the gate (tests/api/test_agents_reason_and_process_document_gate.py covers that).
    monkeypatch.setattr(agents_router, "gate", lambda *a, **k: None)
    monkeypatch.setattr(agent_run_repo, "record", lambda **kw: rows.append(kw))
    return rows


def test_the_run_is_recorded_with_who_asked_what_and_every_tool_call(recorded):
    body = _call(_Nick(_result()))
    assert len(recorded) == 1
    row = recorded[0]
    assert row["agent_slug"] == "agent_nick"
    assert row["user_id"] == "u-42"
    assert row["task"] == "Which suppliers over-billed?"
    assert row["answer"] == "Acme billed 812.50 above its PO."
    assert row["tools_offered"] == ["get_corpus_facts", "get_policy"]
    assert [c.name for c in row["calls"]] == ["get_corpus_facts"]
    assert row["system_prompt"] == "You are AgentNick."
    assert row["model"] == "AgentNick:unified"
    assert row["rounds"] == 2
    assert body["run_id"] == row["run_id"]
    assert body["run_recorded"] is True


def test_the_callers_response_is_otherwise_unchanged(recorded):
    body = _call(_Nick(_result()))
    assert body["answer"] == "Acme billed 812.50 above its PO."
    assert body["tools_used"] == ["get_corpus_facts"]
    assert body["trace"][0]["tool"] == "get_corpus_facts"


def test_a_record_that_cannot_be_written_does_not_fail_the_run(monkeypatch):
    def boom(**kw):
        raise RuntimeError("database is down")
    monkeypatch.setattr(agent_run_repo, "record", boom)
    monkeypatch.setattr(agents_router, "gate", lambda *a, **k: None)
    body = _call(_Nick(_result()))
    assert body["answer"] == "Acme billed 812.50 above its PO."
    assert body["run_recorded"] is False
    assert body["run_id"] is None


def test_a_failed_reasoning_call_is_still_a_500_and_records_nothing(recorded):
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as err:
        _call(_Nick(boom=RuntimeError("model unreachable")))
    assert err.value.status_code == 500
    assert recorded == []


def test_record_result_maps_the_run(monkeypatch):
    rows = []
    monkeypatch.setattr(agent_run_repo, "record", lambda **kw: rows.append(kw))
    from datetime import datetime
    run_id, ok = agent_run_repo.record_result(
        _result(), agent_slug="agent_nick", task="t", user_id="u", workflow_id="w",
        started_at=datetime(2026, 10, 20, 9, 0),
        usage={"prompt_tokens": 10, "completion_tokens": 5, "model_calls": 2})
    assert ok is True and rows[0]["run_id"] == run_id
    assert (rows[0]["prompt_tokens"], rows[0]["completion_tokens"], rows[0]["model_calls"]) \
        == (10, 5, 2)
    assert rows[0]["workflow_id"] == "w"
    assert rows[0]["finished_at"] >= rows[0]["started_at"]
