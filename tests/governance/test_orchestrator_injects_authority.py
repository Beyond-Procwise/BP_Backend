"""The orchestrator must hand the agent its limit, not merely resolve one.

Only 1 of 14 agents reads the existing `governed` envelope, because that envelope is
additive and nothing obliges anyone to read it. `authority` is different: the email
path reads it and refuses to send without it, so the injection has to be real.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

from orchestration.orchestrator import Orchestrator

EMAIL_WORKFLOWS = ["supplier_interaction", "negotiation"]


def _orchestrator():
    # Construct without __init__: the real one builds AgentNick, a DB pool and every
    # agent. This test is about one method's behaviour, not wiring.
    orch = Orchestrator.__new__(Orchestrator)
    return orch


def test_authority_injected_for_email_workflows():
    orch = _orchestrator()
    resolved = {"email_drafting_agent": {"agent": "email_drafting_agent", "governed": True}}
    orch._resolve_authority_for = lambda agents: resolved  # type: ignore[attr-defined]

    class Ctx:
        input_data: dict = {}

    for workflow in EMAIL_WORKFLOWS:
        ctx, enriched = Ctx(), {}
        ctx.input_data = enriched
        out = orch._apply_authority(workflow, ctx, enriched)
        assert out == resolved
        assert enriched["authority"]["email_drafting_agent"]["governed"] is True
        assert ctx.input_data["authority"] is enriched["authority"]


def test_no_authority_for_extraction():
    orch = _orchestrator()
    orch._resolve_authority_for = lambda agents: {"x": {}}  # type: ignore[attr-defined]

    class Ctx:
        input_data: dict = {}

    ctx, enriched = Ctx(), {}
    ctx.input_data = enriched
    # document_extraction must stay deterministic and unaffected, exactly as the
    # existing governance envelope excludes it.
    assert orch._apply_authority("document_extraction", ctx, enriched) is None
    assert "authority" not in enriched


def test_resolution_failure_injects_ungoverned_not_nothing():
    orch = _orchestrator()

    def boom(agents):
        raise RuntimeError("resolver down")

    orch._resolve_authority_for = boom  # type: ignore[attr-defined]

    class Ctx:
        input_data: dict = {}

    ctx, enriched = Ctx(), {}
    ctx.input_data = enriched
    orch._apply_authority("negotiation", ctx, enriched)
    # Absent authority and ungoverned authority must be indistinguishable downstream,
    # and both must mean escalate. Injecting nothing would let a reader that forgets
    # to check treat it as "no restriction".
    block = enriched["authority"]["email_drafting_agent"]
    assert block["governed"] is False
