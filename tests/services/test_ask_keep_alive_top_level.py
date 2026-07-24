"""``keep_alive`` has to travel at the top level of the request.

``ollama_options()`` returns ``{"keep_alive": "10m", ...}`` and the ask path
hands that dict straight to ollama.chat as ``options``. Ollama reads
``keep_alive`` as a top-level request field; inside ``options`` it is ignored,
so the 10 minutes never applied and the model expired on the service default
(OLLAMA_KEEP_ALIVE=5m). Verified against the live daemon — the resident TTL only
moved when the field was passed top-level:

    keep_alive inside options  ->  TTL unchanged
    keep_alive top-level       ->  TTL = 44 minutes from now

The cost of getting it wrong is a ~6.5s reload of a 20GB model on the first ask
after any five-minute lull, which on a 3s answer is most of the wait.
"""

import os
import sys
from types import SimpleNamespace
from typing import Any, Dict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from services.model_selector import RAGPipeline


def _capture_chat_kwargs(monkeypatch) -> Dict[str, Any]:
    seen: Dict[str, Any] = {}

    def _chat(**kwargs):
        seen.update(kwargs)
        return {"message": {"content": '{"answer": "ok", "follow_ups": []}'}}

    monkeypatch.setattr("services.model_selector.ollama.chat", _chat)
    return seen


def _pipeline(monkeypatch, options: Dict[str, Any]) -> RAGPipeline:
    pipeline = RAGPipeline.__new__(RAGPipeline)
    monkeypatch.setattr(pipeline, "_ask_persona", lambda: "persona", raising=False)
    monkeypatch.setattr(
        pipeline, "agent_nick", SimpleNamespace(ollama_options=lambda: options), raising=False
    )
    return pipeline


def test_keep_alive_is_sent_top_level_not_buried_in_options(monkeypatch):
    seen = _capture_chat_kwargs(monkeypatch)
    pipeline = _pipeline(monkeypatch, {"keep_alive": "10m", "num_gpu": 999})

    pipeline._generate_response("prompt", "model")

    assert seen.get("keep_alive") == "10m"
    assert "keep_alive" not in seen.get("options", {})


def test_the_generation_options_still_reach_the_model(monkeypatch):
    """Lifting keep_alive out must not strip the rest of the dict."""

    seen = _capture_chat_kwargs(monkeypatch)
    pipeline = _pipeline(monkeypatch, {"keep_alive": "10m", "num_gpu": 999})

    pipeline._generate_response("prompt", "model")

    assert seen["options"]["num_gpu"] == 999


def test_absent_keep_alive_is_not_invented(monkeypatch):
    """A CPU host returns no keep_alive; don't fabricate one."""

    seen = _capture_chat_kwargs(monkeypatch)
    pipeline = _pipeline(monkeypatch, {})

    pipeline._generate_response("prompt", "model")

    assert "keep_alive" not in seen
