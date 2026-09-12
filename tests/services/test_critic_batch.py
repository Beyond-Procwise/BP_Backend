"""A critic batch must never make Ollama reload the shared model.

AgentNick() preloads with num_gpu=999 while the Modelfile pins 25, so each
preload forces a failed full-GPU load and a reload -- the stall that timed out
two live runs on 2026-09-11. The batch skips it.
"""
from types import SimpleNamespace

import pytest

from src.services import ollama_client
from src.services.opportunity_critic.batch import (
    skip_embedding_model, skip_model_preload,
)


def test_the_batch_loads_no_embedding_model(monkeypatch):
    # With the GPU hidden the model lands in RAM: 2.6 GiB on a host that had
    # 361 MiB free, which is what got the last live run killed.
    from agents import base_agent
    loaded = []
    monkeypatch.setattr(base_agent, "SentenceTransformer",
                        lambda *a, **k: loaded.append(a) or object())

    skip_embedding_model()

    absent = base_agent.SentenceTransformer("BAAI/bge-large-en-v1.5", device="cpu")
    assert loaded == [], "the batch still loaded a sentence-transformer"
    with pytest.raises(RuntimeError, match="without an embedding model"):
        absent.encode(["anything"])


def test_agentnicks_preload_never_reaches_ollama_in_a_batch(monkeypatch):
    calls = []
    # Stands in for the real preload (an HTTP call to Ollama). monkeypatch
    # restores the genuine function afterwards, so nothing leaks to other tests.
    monkeypatch.setattr(ollama_client, "preload_model",
                        lambda *a, **k: calls.append((a, k)) or True)

    skip_model_preload()

    from agents.base_agent import AgentNick
    AgentNick._preload_ollama_model(SimpleNamespace())
    assert calls == [], "AgentNick's preload still reached Ollama"
