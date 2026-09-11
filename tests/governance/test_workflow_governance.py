"""Governance envelope: resolver + orchestrator hook (inject/exempt/ungoverned).

The fail-closed behaviour these three used to describe changed on 2026-09-10 —
see test_workflow_governance_fails_closed.py for why. What is left here is the
resolver's happy path and the injection hook.
"""
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


class _Policies:
    """Minimal stand-in for PolicyEngine's exemption lookup."""

    def __init__(self, exempt=("document_extraction",)):
        self.exempt = list(exempt)

    def get_policy(self, slug):
        return {"details": {"rules": {"ungoverned_workflows": self.exempt}}}


def _bare(exempt=("document_extraction",)):
    orch = Orchestrator.__new__(Orchestrator)
    # The exemption list is policy now, so even a bare orchestrator needs
    # something to read it from -- a governance decision this class cannot make
    # on its own is exactly the point.
    orch.policy_engine = _Policies(exempt)
    return orch


class _Ctx:
    def __init__(self):
        self.input_data = {}


def test_resolve_governance_supplier_ranking():
    _db_or_skip()
    env = resolve_governance("supplier_ranking")
    assert env["agent"] == "supplier_ranking_agent"
    assert any(p.get("policy_type") == "supplier_ranking" for p in env["policies"])


def test_envelope_exempts_extraction():
    o = _bare(); ei = {}
    assert o._apply_governance_envelope("document_extraction", _Ctx(), ei) is None
    assert "governed" not in ei


def test_envelope_injects_for_agentic(monkeypatch):
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


def test_no_environment_variable_can_switch_governance_off(monkeypatch):
    """WORKFLOW_GOVERNANCE_ENABLED is gone, and must not come back.

    It defaulted to on, so it was doing nothing on any deployment that had not
    set it -- but a control with an off switch that leaves no trace is not a
    control, and this one turned the envelope off silently and completely. The
    exemption list in policy is the supported way to say "this workflow runs
    without an envelope", and it is versioned and attributable.
    """
    monkeypatch.setenv("WORKFLOW_GOVERNANCE_ENABLED", "0")
    import src.services.governance_tools.envelope as E
    import src.services.agent_actions as A
    monkeypatch.setattr(E, "resolve_governance",
                        lambda wf, agent=None: {"agent": "supplier_ranking_agent",
                                                "policies": [{"policy_type": "supplier_ranking"}], "prompts": []})
    monkeypatch.setattr(A, "record_action", lambda **k: None)

    o = _bare(); ei = {}
    env = o._apply_governance_envelope("supplier_ranking", _Ctx(), ei)

    assert env is not None, "an environment variable switched governance off"
    assert ei["governed"]["agent"] == "supplier_ranking_agent"


def test_the_flag_is_not_read_anywhere_in_the_orchestrator():
    """Named so a reintroduction shows up as a failing test, not a code review."""
    from pathlib import Path
    text = (Path(__file__).resolve().parents[2] / "src" / "orchestration" /
            "orchestrator.py").read_text()
    assert "WORKFLOW_GOVERNANCE_ENABLED" not in text, (
        "the governance off-switch is back in orchestrator.py"
    )
