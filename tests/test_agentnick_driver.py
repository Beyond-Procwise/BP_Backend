"""AgentNick as a device: one narrow wire, stateless, returning proposals that are never verified.

The driver sends exactly what the tool loop used to send. These tests capture the payload at the
network boundary (egress.post) so a refactor cannot quietly change what the model is asked.
"""
import json
import os
import re

import pytest

from src.services import agentnick_driver as driver
from src.services import egress, tool_runtime

MSGS = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "hi"}]
TOOLS = [{"type": "function", "function": {"name": "t", "description": "d", "parameters": {"type": "object"}}}]


class _Reply:
    status_code = 200
    text = ""

    def __init__(self, message=None, lines=None):
        self._message = message if message is not None else {"role": "assistant", "content": "hello"}
        self._lines = lines or []

    def raise_for_status(self):
        pass

    def json(self):
        return {"message": self._message, "prompt_eval_count": 7, "eval_count": 3}

    def iter_lines(self):
        return iter(self._lines)


@pytest.fixture
def sent(monkeypatch):
    calls = []

    def fake_post(url, **kw):
        calls.append({"url": url, **kw})
        return calls[-1].get("reply") or _Reply()

    monkeypatch.setattr(egress, "post", fake_post)
    return calls


def test_chat_sends_the_same_payload_the_tool_loop_always_did(sent):
    p = driver.chat(MSGS, TOOLS, model="m", timeout=9)
    call = sent[0]
    assert call["url"] == driver.OLLAMA_CHAT and call["timeout"] == 9 and call["require_global"] is False
    assert call["json"] == {"model": "m", "messages": MSGS, "stream": False, "think": False, "keep_alive": -1,
                            "options": {"temperature": 0, "num_predict": 2048}, "tools": TOOLS}
    assert p.content == "hello" and p.tool_calls == [] and p.model == "m"


def test_chat_without_tools_sends_no_tools_key(sent):
    driver.chat(MSGS, None, model="m", timeout=9)
    assert "tools" not in sent[0]["json"]


def test_tool_runtime_chat_still_returns_the_raw_message_dict(sent):
    msg = tool_runtime._chat(MSGS, TOOLS, "m", 9)
    assert msg == {"role": "assistant", "content": "hello"}


def test_a_refused_call_gives_an_empty_message_exactly_as_before(monkeypatch):
    monkeypatch.setattr(egress, "post", lambda url, **kw: None)
    assert tool_runtime._chat(MSGS, TOOLS, "m", 9) == {}


def test_what_comes_back_is_a_proposal_never_verified(sent):
    p = driver.chat(MSGS, TOOLS, model="m", timeout=9)
    assert p.verified is False
    with pytest.raises(Exception):
        p.verified = True  # frozen: nothing downstream may stamp it


def test_tool_calls_are_returned_as_the_model_wrote_them(monkeypatch):
    call = {"function": {"name": "get_policy", "arguments": {"query": "x"}}}
    reply = _Reply({"role": "assistant", "content": "", "tool_calls": [call]})
    monkeypatch.setattr(egress, "post", lambda url, **kw: reply)
    assert driver.chat(MSGS, TOOLS, model="m", timeout=9).tool_calls == [call]


def test_stream_accumulates_content_and_tool_calls_and_asks_for_a_stream(monkeypatch):
    seen = {}
    lines = [json.dumps({"message": {"content": "he"}}).encode(), b"",
             json.dumps({"message": {"content": "llo", "tool_calls": [{"function": {"name": "t"}}]}}).encode(),
             b"not json"]

    def fake_post(url, **kw):
        seen.update(kw)
        return _Reply(lines=lines)

    monkeypatch.setattr(egress, "post", fake_post)
    p = driver.chat_stream(MSGS, TOOLS, model="m", timeout=5)
    assert seen["stream"] is True and seen["json"]["stream"] is True and seen["json"]["tools"] == TOOLS
    assert p.content == "hello" and p.tool_calls == [{"function": {"name": "t"}}]


def test_stream_says_so_when_egress_refuses(monkeypatch):
    monkeypatch.setattr(egress, "post", lambda url, **kw: None)
    with pytest.raises(RuntimeError, match="egress refused the model call"):
        driver.chat_stream(MSGS, TOOLS, model="m", timeout=5)


def test_structured_sends_the_schema_as_format_and_returns_the_object(monkeypatch):
    captured = {}

    def fake_post(url, **kw):
        captured.update(kw)
        return _Reply({"role": "assistant", "content": '{"a": 1}'})

    monkeypatch.setattr(egress, "post", fake_post)
    out = driver.structured("SYS", "USER", {"type": "object"}, model="m", timeout_s=4)
    assert out == {"a": 1} and captured["json"]["format"] == {"type": "object"}
    assert captured["json"]["messages"] == [{"role": "system", "content": "SYS"}, {"role": "user", "content": "USER"}]


@pytest.mark.parametrize("content,match", [("", "answered nothing"), ("[1]", "did not answer with an object")])
def test_structured_refuses_an_empty_or_non_object_answer(monkeypatch, content, match):
    monkeypatch.setattr(egress, "post", lambda url, **kw: _Reply({"role": "assistant", "content": content}))
    with pytest.raises(RuntimeError, match=match):
        driver.structured("S", "U", {"type": "object"})


def test_tokens_are_counted_through_the_driver(sent):
    with driver.count_tokens() as usage:
        driver.chat(MSGS, TOOLS, model="m", timeout=9)
    assert usage == {"prompt_tokens": 7, "completion_tokens": 3, "model_calls": 1}


def test_tool_runtime_keeps_the_names_other_modules_import():
    assert tool_runtime.count_tokens is driver.count_tokens
    assert tool_runtime.note_usage is driver.note_usage
    assert tool_runtime._OLLAMA_CHAT == driver.OLLAMA_CHAT
    assert tool_runtime._DEFAULT_MODEL == driver.DEFAULT_MODEL
    assert tool_runtime._DEFAULT_TIMEOUT_S == driver.DEFAULT_TIMEOUT_S


def test_the_driver_holds_no_data_policy_or_tool_machinery():
    src = open(driver.__file__, encoding="utf-8").read()
    for forbidden in ("get_conn", "psycopg", "agent_policy", "governance", "orchestration", "run_calculation",
                      "Tool(", "handler"):
        assert not re.search(rf"\b{re.escape(forbidden)}", src), forbidden


# ---- generate: the second route to the model ----------------------------------------------------------

def test_generate_hands_the_prompt_and_every_option_to_the_client_and_returns_its_text(monkeypatch):
    from src.services import ollama_client
    seen = {}
    monkeypatch.setattr(ollama_client, "ollama_generate",
                        lambda prompt, **kw: (seen.update(prompt=prompt, **kw), "the text")[1])
    out = driver.generate("write it", model="m", temperature=0.3, think=False, format={"type": "object"},
                          background=True, use_load_options=False)
    assert out == "the text"
    assert seen == {"prompt": "write it", "model": "m", "temperature": 0.3, "think": False,
                    "format": {"type": "object"}, "background": True, "use_load_options": False}


def test_generate_returns_none_when_the_client_does(monkeypatch):
    from src.services import ollama_client
    monkeypatch.setattr(ollama_client, "ollama_generate", lambda prompt, **kw: None)
    assert driver.generate("x") is None


def test_generate_reads_the_client_at_call_time_so_a_patch_made_later_is_honoured(monkeypatch):
    from src.services import ollama_client
    first = driver.generate.__globals__["__name__"]
    monkeypatch.setattr(ollama_client, "ollama_generate", lambda prompt, **kw: "patched")
    assert driver.generate("x") == "patched" and first == driver.__name__


def test_a_patch_made_through_the_other_import_name_reaches_generate_too(monkeypatch):
    import services.ollama_client as other
    monkeypatch.setattr(other, "ollama_generate", lambda prompt, **kw: "via services")
    assert driver.generate("x") == "via services"


def test_generate_adds_no_behaviour_of_its_own(monkeypatch):
    from src.services import ollama_client
    boom = RuntimeError("model down")

    def raising(prompt, **kw):
        raise boom

    monkeypatch.setattr(ollama_client, "ollama_generate", raising)
    with pytest.raises(RuntimeError) as e:
        driver.generate("x")
    assert e.value is boom


def test_a_refused_call_is_marked_refused_and_an_empty_answer_is_not(monkeypatch):
    monkeypatch.setattr(egress, "post", lambda url, **kw: None)
    refused = driver.chat(MSGS, None, model="m", timeout=1)
    assert refused.refused is True and refused.content == ""
    monkeypatch.setattr(egress, "post", lambda url, **kw: _Reply({"role": "assistant", "content": ""}))
    empty = driver.chat(MSGS, None, model="m", timeout=1)
    assert empty.refused is False and empty.content == ""


def test_a_normal_answer_is_not_marked_refused(sent):
    assert driver.chat(MSGS, TOOLS, model="m", timeout=9).refused is False
