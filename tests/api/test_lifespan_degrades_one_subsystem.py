"""One subsystem failing at startup must cost that subsystem, not the process.

`lifespan` wrapped ~250 lines of initialisation in a single `except Exception`
that nulls `agent_nick`, the orchestrator, the agent registry and eleven other
pieces of app state. Thirteen steps inside it carry their own guard; about
eleven did not, so any one of those took the whole system down to a degraded
boot -- serving requests from a hollow process, announced by one CRITICAL line.

That is not hypothetical. It fired on this host on every boot: a cross-encoder
OOM inside `RAGPipeline(agent_nick)` (the GPU is 85% held by Ollama's runner)
wiped the state and, because everything after the failure point is skipped,
silently took `SessionNotifyListener`, the extraction-hint cache and the formula
audit sink with it.

The rule these tests pin:

  * an OPTIONAL subsystem fails  -> it alone is None, the core survives, and
    nothing is logged as CRITICAL;
  * a CORE subsystem fails       -> the fatal path still fires, exactly as
    before. Degrading everything is right when there is nothing left to serve
    with; it was only ever wrong as a response to a reranker.

The routers already agree with this: `documents.get_rag_pipeline` and
`system.get_rag_pipeline` both raise a clean 503 when `state.rag_pipeline` is
absent, so a missing RAG pipeline was always meant to be survivable.
"""
from __future__ import annotations

import asyncio
import logging
import types

import pytest

from unittest import mock

import api.main as main_module
from src.services.formulas import get_audit_sink, set_audit_sink


class FakeNick:
    def __init__(self, *a, **k):
        self.agents = None
        self.auto_registry = None


class FakeAgentRegistry:
    def __init__(self, *a, **k):
        pass

    def add_aliases(self, *a, **k):
        pass


class FakeAutoRegistry:
    agent_ids: list = []

    @classmethod
    def from_json(cls, *a, **k):
        return cls()

    def set_agent_nick(self, *a, **k):
        pass

    def get_contract(self, agent_id):
        return types.SimpleNamespace(class_path=None)


def _boom(message):
    def _raise(*a, **k):
        raise RuntimeError(message)

    return _raise


@pytest.fixture
def restore_audit_sink():
    """lifespan installs the durable sink; do not leak it into other tests."""
    previous = get_audit_sink()
    yield
    set_audit_sink(previous)


@pytest.fixture
def run_lifespan(restore_audit_sink):
    """Drive the real `lifespan` with the heavy constructors faked.

    Everything patched here is either a hard dependency on hardware we do not
    want in a unit test (the GPU, Ollama) or a network/database client. The
    control flow under test -- which failures are contained and which are not --
    is the real thing.
    """

    def _run(**overrides):
        return asyncio.run(_run_async(**overrides))

    async def _run_async(**overrides):
        app = types.SimpleNamespace(state=types.SimpleNamespace(), routes=[])
        patches = {
            "AgentNick": FakeNick,
            "AgentRegistry": FakeAgentRegistry,
            "Orchestrator": lambda *a, **k: types.SimpleNamespace(
                backend_scheduler=None
            ),
            "ModelTrainingEndpoint": lambda *a, **k: object(),
            "RAGPipeline": lambda *a, **k: object(),
        }
        patches.update(overrides)

        with mock.patch("agents.auto_registry.AutoRegistry", FakeAutoRegistry):
            with mock.patch.multiple(main_module, **patches):
                async with main_module.lifespan(app):
                    # Snapshot inside the context: the shutdown half of lifespan
                    # nulls this state on the way out.
                    return types.SimpleNamespace(**vars(app.state))

    return _run


class TestAnOptionalSubsystemFailing:
    """RAGPipeline is the one that actually fired."""

    def test_the_core_survives(self, run_lifespan):
        state = run_lifespan(RAGPipeline=_boom("cross encoder OOM"))
        assert state.agent_nick is not None, (
            "a reranker that would not fit on the GPU nulled agent_nick"
        )

    def test_the_orchestrator_survives(self, run_lifespan):
        state = run_lifespan(RAGPipeline=_boom("cross encoder OOM"))
        assert state.orchestrator is not None

    def test_only_the_failed_subsystem_is_missing(self, run_lifespan):
        state = run_lifespan(RAGPipeline=_boom("cross encoder OOM"))
        assert state.rag_pipeline is None, "the routers 503 on this, by design"

    def test_it_is_not_announced_as_a_fatal(self, run_lifespan, caplog):
        with caplog.at_level(logging.INFO, logger="api.main"):
            run_lifespan(RAGPipeline=_boom("cross encoder OOM"))

        assert not [r for r in caplog.records if r.levelno >= logging.CRITICAL], (
            "an optional subsystem failing is not a system failure"
        )

    def test_a_second_optional_failure_is_also_contained(self, run_lifespan):
        """ModelTrainingEndpoint: Orchestrator's own signature defaults it to None."""
        state = run_lifespan(ModelTrainingEndpoint=_boom("no training host"))
        assert state.agent_nick is not None
        assert state.model_training_endpoint is None


class TestACoreSubsystemFailing:
    """The safety valve must stay. This is the over-fixing guard."""

    def test_a_failed_agent_nick_is_still_fatal(self, run_lifespan):
        state = run_lifespan(AgentNick=_boom("no database"))
        assert state.agent_nick is None
        assert state.orchestrator is None

    def test_a_failed_orchestrator_is_still_fatal(self, run_lifespan):
        state = run_lifespan(Orchestrator=_boom("cannot build the graph"))
        assert state.agent_nick is None

    def test_it_is_still_announced_as_a_fatal(self, run_lifespan, caplog):
        with caplog.at_level(logging.INFO, logger="api.main"):
            run_lifespan(AgentNick=_boom("no database"))

        assert [r for r in caplog.records if r.levelno >= logging.CRITICAL], (
            "losing the core must stay loud"
        )
