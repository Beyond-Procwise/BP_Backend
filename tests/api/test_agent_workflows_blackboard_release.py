"""A canvas run's shared blackboard is dropped when the run ends.

The engine's agent wiring creates one WorkflowContext per run id on the
orchestrator. The orchestrator's own execute_workflow releases it in a
``finally``; the canvas path called the engine directly and never did, so
every canvas run's blackboard (brief, prior results) stayed resident until
the FIFO cap pushed it out.
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import api.routers.agent_workflows as awf


class _Engine:
    def __init__(self, fail=False):
        self.fail = fail

    def execute(self, graph, **kw):
        if self.fail:
            raise RuntimeError("engine blew up")
        kw["resume_state"].status = "completed"


def _state():
    return SimpleNamespace(errors=[], status="running", shared_data={})


def _run(monkeypatch, engine):
    released, finished = [], []
    monkeypatch.setattr(awf.reqrepo, "finish_run", lambda rid, s: finished.append((rid, s)))
    awf._run_to_completion(engine, object(), _state(), "run-1", "u1",
                           release=released.append)
    return released, finished


def test_a_finished_canvas_run_releases_its_blackboard(monkeypatch):
    released, finished = _run(monkeypatch, _Engine())
    assert released == ["run-1"]
    assert finished == [("run-1", "completed")]


def test_a_crashed_canvas_run_releases_its_blackboard_too(monkeypatch):
    released, finished = _run(monkeypatch, _Engine(fail=True))
    assert released == ["run-1"]
    assert finished == [("run-1", "failed")]
