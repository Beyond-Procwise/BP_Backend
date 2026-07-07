"""Governance envelope: resolver + orchestrator hook (inject/exclude/fallback)."""
import pytest

from src.services.governance_tools.envelope import resolve_governance
from src.orchestration.orchestrator import Orchestrator


def _db_or_skip():
    from src.services.db import get_conn
    try:
        with get_conn() as c, c.cursor() as cur:
            cur.execute("select 1")
    except Exception:
        pytest.skip("no DB")


def _bare():
    return Orchestrator.__new__(Orchestrator)


class _Ctx:
    def __init__(self):
        self.input_data = {}


def test_resolve_governance_supplier_ranking():
    _db_or_skip()
    env = resolve_governance("supplier_ranking")
    assert env["agent"] == "supplier_ranking_agent"
    assert any(p.get("policy_type") == "supplier_ranking" for p in env["policies"])


def test_envelope_excludes_extraction():
    o = _bare(); ei = {}
    assert o._apply_governance_envelope("document_extraction", _Ctx(), ei) is None
    assert "governed" not in ei


def test_envelope_injects_for_agentic(monkeypatch):
    monkeypatch.setenv("WORKFLOW_GOVERNANCE_ENABLED", "1")
    import src.services.governance_tools.envelope as E
    import src.services.agent_actions as A
    monkeypatch.setattr(E, "resolve_governance",
                        lambda wf, agent=None: {"agent": "supplier_ranking_agent",
                                                "policies": [{"policy_type": "supplier_ranking"}], "prompts": []})
    monkeypatch.setattr(A, "record_action", lambda **k: None)  # no DB write
    o = _bare(); ei = {}; ctx = _Ctx()
    env = o._apply_governance_envelope("supplier_ranking", ctx, ei)
    assert env and ei["governed"]["agent"] == "supplier_ranking_agent"
    assert ctx.input_data["governed"] == ei["governed"]


def test_envelope_flag_off(monkeypatch):
    monkeypatch.setenv("WORKFLOW_GOVERNANCE_ENABLED", "0")
    o = _bare(); ei = {}
    assert o._apply_governance_envelope("supplier_ranking", _Ctx(), ei) is None
    assert "governed" not in ei
