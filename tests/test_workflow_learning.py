"""Phase 4 — learning loop: workflow outcomes feed the shared pattern store.

On success the orchestrator reinforces the patterns that informed the run;
concern signals are recorded as low-confidence patterns. Best-effort: a failing
pattern store never breaks workflow completion.
"""
import os
import sys
from types import SimpleNamespace

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from orchestration.orchestrator import Orchestrator
from orchestration.workflow_context import SignalType
from agents.base_agent import AgentContext


class _RecordingPatternService:
    def __init__(self):
        self.reinforced = []
        self.recorded = []

    def get_patterns(self, category=None, min_confidence=0.0):
        return []

    def reinforce_pattern(self, pattern_type, pattern_text, delta=0.05):
        self.reinforced.append((pattern_type, pattern_text))
        return True

    def record_pattern(self, pattern_type, pattern_text, category="", confidence=0.5):
        self.recorded.append((pattern_type, pattern_text, category))
        return {"pattern_type": pattern_type}


def _orch(ps):
    nick = SimpleNamespace(
        settings=SimpleNamespace(script_user="tester", max_workers=1, parallel_processing=False),
        agents={},
        policy_engine=SimpleNamespace(),
        query_engine=SimpleNamespace(),
        routing_engine=SimpleNamespace(routing_model=None),
        pattern_service=ps,
    )
    return Orchestrator(nick)


def _seed_ctx(orch, wid="wf-learn"):
    ctx = AgentContext(workflow_id=wid, agent_id="root", user_id="t",
                       input_data={"workflow": "demo", "category": "Raw Materials"})
    wf = orch._get_or_create_wf_context(ctx)
    return ctx, wf


def test_success_reinforces_brief_patterns():
    ps = _RecordingPatternService()
    orch = _orch(ps)
    ctx, wf = _seed_ctx(orch)
    wf.update_shared("patterns", [
        {"pattern_type": "billing", "pattern_text": "EU suppliers bill in EUR"},
        {"pattern_type": "lead_time", "pattern_text": "steel lead time ~6 weeks"},
    ])

    orch._learn_from_workflow("wf-learn", "demo", ctx, {"ok": True}, success=True)

    assert ("billing", "EU suppliers bill in EUR") in ps.reinforced
    assert ("lead_time", "steel lead time ~6 weeks") in ps.reinforced


def test_concern_signals_recorded_as_patterns():
    ps = _RecordingPatternService()
    orch = _orch(ps)
    ctx, wf = _seed_ctx(orch, "wf-c")
    wf.emit_signal(agent="X", signal_type=SignalType.CONFIDENCE_LOW,
                   message="supplier match weak")

    orch._learn_from_workflow("wf-c", "demo", ctx, {"ok": True}, success=True)

    assert any(r[0] == "workflow_concern" and "supplier match weak" in r[1]
               and r[2] == "Raw Materials" for r in ps.recorded)


def test_failure_does_not_reinforce_or_raise():
    ps = _RecordingPatternService()
    orch = _orch(ps)
    ctx, wf = _seed_ctx(orch, "wf-f")
    wf.update_shared("patterns", [{"pattern_type": "x", "pattern_text": "y"}])

    orch._learn_from_workflow("wf-f", "demo", ctx, {"error": "boom"}, success=False)

    assert ps.reinforced == []   # no reinforcement on failure


def test_learning_is_best_effort_when_store_raises():
    class Boom:
        def reinforce_pattern(self, *a, **k):
            raise RuntimeError("down")

        def record_pattern(self, *a, **k):
            raise RuntimeError("down")

    orch = _orch(Boom())
    ctx, wf = _seed_ctx(orch, "wf-b")
    wf.update_shared("patterns", [{"pattern_type": "x", "pattern_text": "y"}])
    wf.emit_signal(agent="X", signal_type=SignalType.NEEDS_ATTENTION, message="m")

    # must not raise
    orch._learn_from_workflow("wf-b", "demo", ctx, {"ok": True}, success=True)
