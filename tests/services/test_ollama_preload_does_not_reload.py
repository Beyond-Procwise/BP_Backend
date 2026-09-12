"""A preload must not evict the model that is already loaded.

Every path to Ollama pins num_gpu to ALL_GPU_LAYERS (999) because the pin is
worth 189 tok/s against 19.7 — see test_ollama_gpu_fallback. AgentNick's own
Modelfile pins num_gpu 25, because the whole 19.1 GB model only fits on an empty
card. Ollama keys a loaded runner on its load-affecting options, so a request
naming a DIFFERENT num_gpu does not reuse the resident runner: it loads another
one, and this model takes ~140 s to load.

Every process that constructs AgentNick preloads at startup, and the preload
asked for 999. On 2026-09-11 the live service, a test suite and another
session's runner did that against a card that could not take 999, and Ollama
logged 22 loads at requested=999 against 27 at requested=25, alternating: load
999, fail, fall back, load 25, back-off expires, load 999 again. While that ran,
the model was unavailable and every AI call in the product timed out.

Two rules here:

  * If the model is ALREADY loaded, the preload does nothing at all. Its whole
    purpose is to pay the cold load at startup instead of in a reader's first
    question; there is no cold load left to pay, and asking for a different
    layout would throw away a resident runner.
  * A load that FAILS takes the pin off, whatever the failure says. Only the
    server's "memory layout cannot be allocated" did that before, so a load
    that was killed or timed out left every later caller still demanding 999.
"""

from __future__ import annotations

import pytest

from src.services import egress, ollama_client as oc

MODEL = "BeyondProcwise/AgentNick:unified"


class _Resp:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise egress.HTTPError(f"{self.status_code} Server Error")


@pytest.fixture(autouse=True)
def _clean():
    oc.clear_layout_rejection()
    yield
    oc.clear_layout_rejection()


def _posts(seen, response=None):
    def _post(url, json=None, timeout=None, **kw):
        seen.append(json)
        return response or _Resp()
    return _post


# ---------------------------------------------------------------------------
# what is already loaded
# ---------------------------------------------------------------------------
def test_the_loaded_models_are_read_from_the_server(monkeypatch):
    monkeypatch.setattr(oc.egress, "get", lambda url, **kw: _Resp(
        payload={"models": [{"name": MODEL, "size_vram": 10650803712}]}))

    assert oc.loaded_models() == [MODEL]


@pytest.mark.parametrize("outcome", ["error", "bad-status", "junk"], ids=str)
def test_an_unreadable_server_lists_nothing_rather_than_raising(monkeypatch, outcome):
    """Nothing loaded is the safe answer: the preload then goes ahead, which is
    what it did before this existed."""
    def _get(url, **kw):
        if outcome == "error":
            raise egress.RequestException("connection refused")
        if outcome == "bad-status":
            return _Resp(status_code=500, text="nope")
        return _Resp(payload={"unexpected": True})

    monkeypatch.setattr(oc.egress, "get", _get)

    assert oc.loaded_models() == []


# ---------------------------------------------------------------------------
# the preload
# ---------------------------------------------------------------------------
def test_a_resident_model_is_not_preloaded_again(monkeypatch):
    """The 140-second load this would start is the outage it used to cause."""
    seen = []
    monkeypatch.setattr(oc, "loaded_models", lambda: [MODEL])
    monkeypatch.setattr(oc.egress, "post", _posts(seen))

    assert oc.preload_model(MODEL) is True
    assert seen == [], f"the preload asked the server to load a resident model: {seen}"


def test_a_model_that_is_not_loaded_is_preloaded(monkeypatch):
    seen = []
    monkeypatch.setattr(oc, "loaded_models", lambda: ["some-other-model"])
    monkeypatch.setattr(oc.egress, "post", _posts(seen))

    assert oc.preload_model(MODEL) is True
    assert len(seen) == 1
    assert seen[0]["options"]["num_gpu"] == oc.ALL_GPU_LAYERS


# ---------------------------------------------------------------------------
# a failed load takes the pin off
# ---------------------------------------------------------------------------
def test_a_failed_preload_takes_the_pin_off(monkeypatch):
    """It was killed, or it timed out, or the runner never came up. Whatever the
    words, the card would not take this layout, and every later caller asking
    for it again is how the reload loop kept going."""
    monkeypatch.setattr(oc, "loaded_models", lambda: [])
    monkeypatch.setattr(oc.egress, "post",
                        lambda *a, **k: (_ for _ in ()).throw(
                            egress.RequestException("timed out waiting for llama runner")))

    assert oc.preload_model(MODEL) is False
    assert "num_gpu" not in oc.gpu_options(), (
        "after a failed load the client still demands the whole card")


def test_a_refused_layout_still_falls_back_and_loads(monkeypatch):
    """The existing behaviour, kept: an explicit refusal drops the pin and the
    preload goes ahead unpinned, so the first real request is not a cold load."""
    seen = []
    monkeypatch.setattr(oc, "loaded_models", lambda: [])
    responses = [_Resp(status_code=500,
                       text='{"error":"memory layout cannot be allocated with num_gpu = 999"}'),
                 _Resp()]

    def _post(url, json=None, timeout=None, **kw):
        seen.append(json)
        return responses[min(len(seen) - 1, 1)]

    monkeypatch.setattr(oc.egress, "post", _post)

    assert oc.preload_model(MODEL) is True
    assert len(seen) == 2, seen
    assert seen[0]["options"]["num_gpu"] == oc.ALL_GPU_LAYERS
    assert "num_gpu" not in seen[1].get("options", {})
