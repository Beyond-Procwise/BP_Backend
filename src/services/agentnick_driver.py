"""The one wire to AgentNick.

AgentNick is a processor that sits alongside the orchestrator, not inside it. The orchestrator hands
it a prompt and gets back a proposal: some text, and any tool calls it would like made. This module
is that boundary and nothing else. It has no database, no policy, no tools and no memory; it cannot
execute what a proposal asks for. A proposal is never verified here, and ``Proposal.verified`` cannot
be set: only the orchestrator decides whether to act on one.

Every request to the model that the tool loop makes goes through here, so there is one place to see
what is asked, one place to count tokens, and one place that changes if the way the model is reached
changes. Other callers still reach the model directly; ``tests/test_agentnick_driver_boundary.py``
lists them and only lets that list shrink.
"""
from __future__ import annotations

import contextlib
import contextvars
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.services import egress

OLLAMA_CHAT = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/") + "/api/chat"
# AgentNick is the only base model. Never repoint this at another family.
DEFAULT_MODEL = os.getenv("AGENTNICK_MODEL", "BeyondProcwise/AgentNick:unified")
DEFAULT_TIMEOUT_S = int(os.getenv("TOOL_RUNTIME_TIMEOUT_S", "180"))


@dataclass(frozen=True)
class Proposal:
    """What the model suggested. It is a suggestion: ``verified`` is always False here."""

    content: str
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    model: str = ""
    message: Dict[str, Any] = field(default_factory=dict)
    verified: bool = False
    # The egress guard refused the call, so nothing was sent. Different from a model that answered nothing.
    refused: bool = False


_COUNTS: contextvars.ContextVar[tuple] = contextvars.ContextVar("token_counts", default=())


@contextlib.contextmanager
def count_tokens():
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "model_calls": 0}
    token = _COUNTS.set(_COUNTS.get() + (usage,))
    try:
        yield usage
    finally:
        _COUNTS.reset(token)


def note_usage(body: Any) -> None:
    """Add one model reply's token counts to every count that is open."""
    counts = _COUNTS.get()
    if not counts:
        return
    body = body if isinstance(body, dict) else {}
    prompt = int(body.get("prompt_eval_count") or 0)
    completion = int(body.get("eval_count") or 0)
    for usage in counts:
        usage["prompt_tokens"] += prompt
        usage["completion_tokens"] += completion
        usage["model_calls"] += 1


def _sampling() -> Dict[str, Any]:
    """The sampling options for a turn. One place, so a change to what the loaded model needs lands once."""
    return {"temperature": 0, "num_predict": 2048}


def _send(payload: Dict[str, Any], timeout: int, *, stream: bool = False) -> Any:
    kwargs: Dict[str, Any] = dict(
        purpose=egress.Purpose.MODEL_INFERENCE,
        json=payload,
        timeout=timeout,
        require_global=False,   # the model daemon is on localhost by design
        raise_transport_errors=True,
    )
    if stream:
        kwargs["stream"] = True
    return egress.post(OLLAMA_CHAT, **kwargs)


def chat(messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None, *,
         model: Optional[str] = None, timeout: Optional[int] = None) -> Proposal:
    """One turn, answered whole. A call egress refused comes back as an empty proposal."""
    name = model or DEFAULT_MODEL
    payload: Dict[str, Any] = {
        "model": name,
        "messages": messages,
        "stream": False,
        # Reasoning-tuned models return an EMPTY `response`/content unless think is
        # off. Leaving this on silently yields blank answers.
        "think": False,
        "keep_alive": -1,
        "options": _sampling(),
    }
    if tools:
        payload["tools"] = tools
    response = _send(payload, timeout or DEFAULT_TIMEOUT_S)
    if response is None:
        return Proposal(content="", model=name, refused=True)
    response.raise_for_status()
    body = response.json()
    note_usage(body)
    message = body.get("message", {}) or {}
    return Proposal(content=message.get("content") or "", tool_calls=list(message.get("tool_calls") or []),
                    model=name, message=message)


def chat_stream(messages: List[Dict[str, Any]], tools: List[Dict[str, Any]], *,
                model: Optional[str] = None, timeout: Optional[int] = None) -> Proposal:
    """One turn read as a stream and returned whole. Raises if egress refuses or the call fails."""
    name = model or DEFAULT_MODEL
    payload: Dict[str, Any] = {
        "model": name,
        "messages": messages,
        "stream": True,
        "think": False,
        "keep_alive": -1,
        "options": _sampling(),
        "tools": tools,
    }
    response = _send(payload, timeout or DEFAULT_TIMEOUT_S, stream=True)
    if response is None:
        raise RuntimeError("egress refused the model call")
    response.raise_for_status()
    parts: List[str] = []
    calls: List[Dict[str, Any]] = []
    for line in response.iter_lines():
        if not line:
            continue
        try:
            chunk = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        message = chunk.get("message") or {}
        if message.get("tool_calls"):
            calls.extend(message["tool_calls"])
        fragment = message.get("content")
        if fragment:
            parts.append(fragment)
    return Proposal(content="".join(parts), tool_calls=calls, model=name)


def generate(prompt: str, **kwargs: Any) -> Optional[str]:
    """One completion through the model's other route (``/api/generate``), with the client's queue, retries
    and foreground accounting. Every option goes to the client unchanged and its text comes back unchanged:
    this adds nothing of its own, so a caller moved onto it behaves as it did. The text is a proposal.

    The client is read at call time, so a caller or a test that replaces ``ollama_client.ollama_generate``
    is still honoured.
    """
    from src.services import ollama_client

    return ollama_client.ollama_generate(prompt, **kwargs)


def structured(system: str, user: str, schema: Dict[str, Any], *, model: Optional[str] = None,
               timeout_s: Optional[int] = None) -> Dict[str, Any]:
    """One turn, no tools, answered as a JSON object in ``schema``.

    The schema is passed as the model server's ``format``, so the answer is constrained to it as it is
    generated rather than parsed out of prose afterwards. Raises when the model cannot be reached or
    answers nothing.
    """
    from src.services import ollama_client

    def _once() -> Any:
        payload: Dict[str, Any] = {
            "model": model or DEFAULT_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "think": False,
            "keep_alive": -1,
            "format": schema,
            # Every load-affecting option, as every other caller sends them. The
            # model server keys a loaded model on these: a request that named
            # none would ask for a runner of its own and evict the one everyone
            # shares. Read afresh on each send, so a refusal just noted is honoured.
            "options": {**ollama_client.load_options(), **_sampling()},
        }
        return payload, _send(payload, timeout_s or DEFAULT_TIMEOUT_S)

    payload, response = _once()
    asked = payload["options"].get("num_gpu")
    if (response is not None and asked is not None
            and getattr(response, "status_code", 200) >= 500
            and ollama_client.is_layout_rejection(getattr(response, "text", ""))):
        # The card will not take the whole model just now. Say so once, for
        # every caller, and go again on the server's own split.
        ollama_client.note_layout_rejection(asked)
        payload, response = _once()
    if response is None:
        raise RuntimeError("the model could not be reached")
    response.raise_for_status()
    body = response.json()
    note_usage(body)
    content = (body.get("message") or {}).get("content") or ""
    if not content.strip():
        raise RuntimeError("the model answered nothing")
    parsed = json.loads(content)
    if not isinstance(parsed, dict):
        raise RuntimeError("the model did not answer with an object")
    return parsed
