"""The egress audit names who made the call (1.20k).

In the first hour of the shadow week 110 of 112 live audit rows read agent "unknown": the scheduler's
jobs and every agent run that did not go through the tool loop. Now a scheduled job runs as
``job:<name>`` and an agent's ``execute()`` runs as the agent's own slug, which wins inside a job.
One module: ``services.egress`` and ``src.services.egress`` must be the same object, or a caller that
imports the bare name gets a second copy with its own caller context, queue and policy cache.
"""
from __future__ import annotations

import os
import sys
from datetime import timedelta
from types import SimpleNamespace

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.services import egress
import services.backend_scheduler as backend_scheduler   # imported as the scheduler suite imports it
from agents import base_agent


# ---------------------------------------------------------------- scheduler jobs

def _job(name, runner):
    return backend_scheduler._ScheduledJob(name, runner, timedelta(hours=1))


def test_a_scheduled_job_runs_as_its_name(monkeypatch):
    monkeypatch.setattr(backend_scheduler, "configure_gpu", lambda: None)
    seen = []
    host = SimpleNamespace(_deregister_job=lambda name: None)
    backend_scheduler.BackendScheduler._execute_job(host, _job("price_outliers", lambda: seen.append(egress.current_agent())))
    assert seen == ["job:price_outliers"]
    assert egress.current_agent() is None   # the name does not outlive the job


def test_a_failing_job_still_clears_its_name(monkeypatch):
    monkeypatch.setattr(backend_scheduler, "configure_gpu", lambda: None)

    def _boom():
        raise RuntimeError("job failed")
    host = SimpleNamespace(_deregister_job=lambda name: None)
    backend_scheduler.BackendScheduler._execute_job(host, _job("nightly", _boom))   # logged, not raised
    assert egress.current_agent() is None


# ---------------------------------------------------------------- agent runs

class _Routing:
    def log_process(self, **_kw):
        return None

    def validate_workflow_id(self, *_a, **_k):
        return True


class _Nick:
    def __init__(self):
        self.settings = SimpleNamespace(script_user="tester", extraction_model="x", ollama_quantized_model=None)
        self.prompt_engine = SimpleNamespace()
        self.learning_repository = None
        self.process_routing_service = _Routing()

    def ollama_options(self):
        return {}


class RecorderAgent(base_agent.BaseAgent):
    def __init__(self, nick, seen):
        super().__init__(nick)
        self.seen = seen

    def run(self, context):
        self.seen.append(egress.current_agent())
        return base_agent.AgentOutput(status=base_agent.AgentStatus.SUCCESS, data={})


def _context():
    return base_agent.AgentContext(workflow_id="wf", agent_id="recorder", user_id=None, input_data={})


def test_an_agent_run_is_audited_under_its_own_slug():
    seen = []
    agent = RecorderAgent(_Nick(), seen)
    out = agent.execute(_context())
    assert out.status == base_agent.AgentStatus.SUCCESS
    slug = agent._governance_slug()
    assert slug and seen == [slug]
    assert egress.current_agent() is None


def test_the_agent_name_wins_inside_a_job():
    seen = []
    agent = RecorderAgent(_Nick(), seen)
    with egress.calling("job:nightly"):
        agent.execute(_context())
        assert egress.current_agent() == "job:nightly"   # restored once the agent is done
    assert seen == [agent._governance_slug()]


# ---------------------------------------------------------------- one egress module

def test_egress_is_one_module_however_it_is_imported():
    """src/services/__init__.py answers the bare name with the one canonical module (1.20k); a second copy
    would carry its own caller context, so a name set by the scheduler would be invisible to the model client."""
    import importlib
    assert importlib.import_module("services.egress") is egress
    assert importlib.import_module("services.egress_log") is importlib.import_module("src.services.egress_log")
