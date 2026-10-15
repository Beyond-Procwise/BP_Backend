"""The ask endpoint carries who should answer, and says who did.

The endpoint is called directly with a stub pipeline: this is about what the
route passes on and passes back, not about routing itself (that is
tests/services/test_ask_routes_to_agents.py).
"""

import asyncio
import json
import os
import sys
import types

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.api.routers import workflows  # noqa: E402
from src.api.routers.workflows import AskRequest, ask_question, ask_question_stream  # noqa: E402

ANSWERED_BY = {"agent": "negotiation", "label": "Negotiation agent", "source": "screen", "note": None}


class _Pipeline:
    def __init__(self):
        self.seen = {}

    def answer_question(self, **kwargs):
        self.seen = kwargs
        cb = kwargs.get("on_event")
        if cb:
            cb("stage", {"stage": "routed", "answered_by": ANSWERED_BY})
        return {"answer": "x", "follow_ups": [], "retrieved_documents": [],
                "next_steps": [], "answered_by": ANSWERED_BY}


class _Request:
    headers: dict = {}


def test_agent_and_screen_reach_the_pipeline():
    p = _Pipeline()
    asyncio.run(ask_question(AskRequest(query="q", user_id="u", agent="negotiation",
                                        screen="negotiations"), _Request(), p, principal=object()))
    assert p.seen["agent"] == "negotiation" and p.seen["screen"] == "negotiations"


def test_absent_fields_reach_the_pipeline_as_none():
    p = _Pipeline()
    asyncio.run(ask_question(AskRequest(query="q", user_id="u"), _Request(), p, principal=object()))
    assert p.seen["agent"] is None and p.seen["screen"] is None


def test_answered_by_comes_back():
    result = asyncio.run(ask_question(AskRequest(query="q", user_id="u"), _Request(),
                                      _Pipeline(), principal=object()))
    assert result["answered_by"] == ANSWERED_BY


def _stream_frames(req, pipeline):
    async def run():
        resp = await ask_question_stream(req, _Request(), pipeline, principal=object())
        out = []
        async for chunk in resp.body_iterator:
            out.append(chunk if isinstance(chunk, str) else chunk.decode())
        return "".join(out)

    text = asyncio.run(run())
    return [json.loads(line[6:]) for line in text.split("\n") if line.startswith("data: ")]


def test_the_stream_passes_agent_and_screen():
    p = _Pipeline()
    _stream_frames(AskRequest(query="q", user_id="u", agent="general", screen="quotes"), p)
    assert p.seen["agent"] == "general" and p.seen["screen"] == "quotes"


def test_the_stream_says_who_answers_before_the_answer_and_in_done():
    frames = _stream_frames(AskRequest(query="q", user_id="u", screen="negotiations"), _Pipeline())
    routed = next(f for f in frames if f.get("stage") == "routed")
    done = next(f for f in frames if f["type"] == "done")
    assert routed["answered_by"] == ANSWERED_BY
    assert done["answered_by"] == ANSWERED_BY
    assert frames.index(routed) < frames.index(done)


def test_the_askable_agents_are_listed_with_names():
    nick = types.SimpleNamespace(agents={"negotiation": object(), "email_dispatch": object()})
    result = asyncio.run(workflows.list_ask_agents(agent_nick=nick, principal=object()))
    assert result == {"agents": [{"slug": "negotiation", "label": "Negotiation agent"}]}


def test_the_list_offers_a_draft_only_to_its_creator():
    draft = types.SimpleNamespace(draft_owner="nick")
    nick = types.SimpleNamespace(agents={"negotiation": object(), "my-draft": draft})
    mine = asyncio.run(workflows.list_ask_agents(agent_nick=nick, principal=types.SimpleNamespace(subject="nick")))
    theirs = asyncio.run(workflows.list_ask_agents(agent_nick=nick, principal=types.SimpleNamespace(subject="x")))
    assert [a["slug"] for a in mine["agents"]] == ["my-draft", "negotiation"]
    assert [a["slug"] for a in theirs["agents"]] == ["negotiation"]
