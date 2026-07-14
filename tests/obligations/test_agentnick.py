"""The bridge must never turn a dead model into a green empty result."""
import pytest
from pydantic import BaseModel

from src.services.obligations.agentnick import AgentNickChat, AgentNickEmbeddings


class Thing(BaseModel):
    name: str


def test_structured_output_returns_validated_schema(monkeypatch):
    captured = {}

    def fake_generate(prompt, **kw):
        captured.update(kw)
        return '{"name": "Contractor"}'

    monkeypatch.setattr("src.services.obligations.agentnick.ollama_generate", fake_generate)

    out = AgentNickChat().with_structured_output(Thing).invoke("extract")

    assert isinstance(out, Thing) and out.name == "Contractor"
    # The schema must reach Ollama as a grammar constraint, not a function-calling tool.
    assert captured["format"] == Thing.model_json_schema()


def test_hybrid_reasoner_is_told_not_to_think(monkeypatch):
    """Without think=False, AgentNick:unified returns an empty `response`."""
    captured = {}

    def fake_generate(prompt, **kw):
        captured.update(kw)
        return '{"name": "x"}'

    monkeypatch.setattr("src.services.obligations.agentnick.ollama_generate", fake_generate)
    AgentNickChat().with_structured_output(Thing).invoke("extract")

    assert captured["think"] is False


def test_dead_model_raises_rather_than_returning_empty(monkeypatch):
    """ollama_generate returns None on failure. Swallowing that would report a contract
    as having zero obligations when in fact it was never read."""
    monkeypatch.setattr(
        "src.services.obligations.agentnick.ollama_generate", lambda prompt, **kw: None
    )

    with pytest.raises(RuntimeError, match="no response"):
        AgentNickChat().invoke("anything")


def test_embeddings_reject_a_missing_model():
    """A stub embedder silently collapses dedup to zero results — refuse to construct one."""
    with pytest.raises(ValueError):
        AgentNickEmbeddings(None)
