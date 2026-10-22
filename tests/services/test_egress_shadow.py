"""1.20 step (a): every outbound call says which AGENT made it, and the egress gate records the
decision it WOULD take (shadow) without taking it.

Behaviour is unchanged: every call that went out before still goes out. What is new is evidence.
Each event row now carries the agent, a shadow decision (allow / deny) and the reason, so a week of
shadow shows exactly what enforcing would block, caller by caller, before anything is enforced
(D-62: one week of shadow before enforcing; Ollama cloud is NOT allowed for company text).

Shadow rule set "default-v0": a LOCAL destination (loopback, or a name that resolves only to
private addresses) would be allowed; any other destination would be denied until an egress policy
rule allows that purpose to that host for that agent. No database, no network.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.services import egress, egress_log
from src.services.egress import Purpose


class _Resp:
    status_code = 200
    headers = {"Content-Type": "text/html"}
    text = "ok"


@pytest.fixture
def world(monkeypatch):
    """A fake transport and a captured audit: (calls made, events recorded)."""
    calls, events = [], []
    monkeypatch.setattr(egress.requests, "request", lambda m, u, **kw: calls.append((m, u)) or _Resp())
    def _resolve(host):
        if host[0].isdigit() or host.startswith("["):          # a literal address resolves to itself
            return [host.strip("[]")]
        return ["127.0.0.1"] if host in ("localhost", "model.internal") else ["93.184.216.34"]
    monkeypatch.setattr(egress, "_resolve", _resolve)
    monkeypatch.setattr(egress_log, "record", lambda **kw: events.append(kw))
    return calls, events


# ---- the agent travels with the call ------------------------------------------------------------

def test_the_calling_agent_is_recorded(world):
    calls, events = world
    assert egress.get("https://example.com/supplier", purpose=Purpose.SUPPLIER_RESEARCH,
                      agent="outside_research") is not None
    assert events[-1]["agent"] == "outside_research"


def test_a_call_that_names_no_agent_is_recorded_as_unknown(world):
    calls, events = world
    egress.get("https://example.com/", purpose=Purpose.FX_RATES)
    assert events[-1]["agent"] == "unknown"


def test_the_agent_flows_through_post_and_request_or_raise(world):
    calls, events = world
    egress.post("https://example.com/", purpose=Purpose.MODEL_INFERENCE, json={"q": 1}, agent="a1")
    egress.request_or_raise("GET", "https://example.com/", purpose=Purpose.FX_RATES, agent="a2")
    assert [e["agent"] for e in events[-2:]] == ["a1", "a2"]


# ---- shadow: decided, recorded, never applied ---------------------------------------------------

def test_an_external_destination_still_goes_out_but_the_shadow_says_deny(world):
    calls, events = world
    assert egress.get("https://api.ollama.com/v1/chat", purpose=Purpose.MODEL_INFERENCE,
                      agent="negotiation") is not None
    assert len(calls) == 1, "shadow must not block the call"
    e = events[-1]
    assert e["outcome"] == "http_200"
    assert e["shadow_decision"] == "deny"
    for word in ("model_inference", "api.ollama.com", "negotiation"):
        assert word in e["shadow_reason"], e["shadow_reason"]


def test_a_local_destination_is_allowed_in_shadow_too(world):
    calls, events = world
    egress.get("http://localhost:11434/api/tags", purpose=Purpose.MODEL_INFERENCE,
               require_global=False, agent="rag")
    assert events[-1]["shadow_decision"] == "allow"
    egress.get("http://model.internal:11434/api/tags", purpose=Purpose.MODEL_INFERENCE,
               require_global=False)
    assert events[-1]["shadow_decision"] == "allow", "a private name is local too"


def test_a_refused_destination_also_carries_agent_and_shadow(world):
    calls, events = world
    assert egress.get("http://169.254.169.254/latest/meta-data/", purpose=Purpose.SUPPLIER_RESEARCH,
                      agent="curious") is None
    e = events[-1]
    assert e["outcome"] == "refused" and e["agent"] == "curious"
    assert e["shadow_decision"] == "allow", "a link-local address is not an external destination"


def test_the_shadow_names_its_rule_set(world):
    calls, events = world
    egress.get("https://example.com/", purpose=Purpose.FX_RATES)
    assert events[-1]["policy_version"] == egress.SHADOW_RULES_VERSION == "shadow-default-v0"


def test_evaluate_never_refuses_in_shadow():
    d = egress._evaluate(Purpose.MODEL_INFERENCE, "api.ollama.com", agent="x")
    assert d.permitted is True and d.shadow == "deny"


# ---- the audit row ---------------------------------------------------------------------------------

def test_the_event_row_carries_agent_and_shadow_after_the_existing_columns(monkeypatch):
    written = {}

    class _Cur:
        def executemany(self, sql, rows):
            written["sql"], written["rows"] = sql, rows

    class _Conn:
        autocommit = True

        def cursor(self):
            return _Cur()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    import src.services.db as db
    monkeypatch.setattr(db, "get_conn", lambda: _Conn())
    monkeypatch.setattr(egress_log, "_take_drops", lambda: 0)
    egress_log.flush([{
        "occurred_at": None, "purpose": "fx_rates", "destination": "x", "method": "GET",
        "outcome": "http_200", "detail": None, "payload_sha256": "abc", "payload_bytes": 3,
        "agent": "fx", "shadow_decision": "deny", "shadow_reason": "why", "policy_version": "shadow-default-v0",
    }])
    row = written["rows"][0]
    assert row[9] == "shadow-default-v0" and row[10] == []
    assert row[11:] == ("fx", "deny", "why")
    assert "agent, shadow_decision, shadow_reason" in " ".join(written["sql"].split())


def test_a_row_recorded_without_the_new_fields_still_writes():
    """The dropped-events row and any older caller name none of the new fields."""
    row = egress_log._row({
        "occurred_at": None, "purpose": "audit", "destination": "-", "method": "-",
        "outcome": "events_dropped", "detail": "3 dropped", "payload_sha256": None, "payload_bytes": None,
    })
    assert row[9] is None and row[11:] == (None, None, None)


def test_record_passes_the_new_fields_into_the_queue(monkeypatch):
    seen = {}
    monkeypatch.setattr(egress_log._queue, "put_nowait", lambda ev: seen.update(ev))
    monkeypatch.setattr(egress_log, "_ensure_worker", lambda: None)
    egress_log.record(purpose="fx_rates", destination="x", method="GET", outcome="http_200",
                      agent="fx", shadow_decision="deny", shadow_reason="r", policy_version="v")
    assert (seen["agent"], seen["shadow_decision"], seen["shadow_reason"], seen["policy_version"]) == ("fx", "deny", "r", "v")


def test_the_schema_adds_the_columns_without_touching_existing_rows():
    ddl = egress_log.DDL
    for col in ("agent", "shadow_decision", "shadow_reason"):
        assert f"ADD COLUMN IF NOT EXISTS {col} " in ddl, col
    assert "DROP TABLE" not in ddl.upper() and "DROP COLUMN" not in ddl.upper()
    draft = Path(__file__).resolve().parents[2] / "deploy" / "sql" / "2026-10-22_egress_event_agent_shadow.sql"
    text = draft.read_text()
    for col in ("agent", "shadow_decision", "shadow_reason"):
        assert f"ADD COLUMN IF NOT EXISTS {col} " in text, col
