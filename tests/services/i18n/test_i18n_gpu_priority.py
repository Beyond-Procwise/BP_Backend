"""Extraction (and every other foreground model call) goes before background translation.

The shared client counts foreground generations -- waiting for a GPU slot or running --
and background translation batches wait while any are in flight, or while the card is
busy with another process's work. On-screen translation does not wait: a person is
looking at it.
"""
from __future__ import annotations

import src.services.ollama_client as oc
from src.services.i18n.filler import BACKGROUND, SCREEN, Filler
from src.services.i18n.gpu_gate import GpuGate
from src.services.i18n.service import TranslateResult


class _Resp:
    status_code = 200
    text = ""

    def raise_for_status(self):
        pass

    def json(self):
        return {"response": "ok"}


def test_a_foreground_call_is_counted_while_it_runs(monkeypatch):
    seen = []

    def fake_post(url, json=None, timeout=None, **kw):
        seen.append(oc.foreground_busy())
        return _Resp()

    monkeypatch.setattr(oc.egress, "post", fake_post)
    assert not oc.foreground_busy()
    oc.ollama_generate("x", model="m", retries=1)
    oc.ollama_generate("x", model="m", retries=1, background=True)
    assert seen == [True, False] and not oc.foreground_busy()


def test_the_count_is_released_when_the_call_fails(monkeypatch):
    def boom(url, json=None, timeout=None, **kw):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(oc.egress, "post", boom)
    try:
        oc.ollama_generate("x", model="m", retries=1)
    except Exception:
        pass
    assert not oc.foreground_busy()


def test_gate_busy_on_foreground_or_a_busy_card():
    assert GpuGate(foreground=lambda: True, utilisation=lambda: 0, util_threshold=30).busy()
    assert GpuGate(foreground=lambda: False, utilisation=lambda: 85, util_threshold=30).busy()
    assert not GpuGate(foreground=lambda: False, utilisation=lambda: 5, util_threshold=30).busy()
    # threshold 0 = the card check is off; an unreadable card never blocks on its own
    assert not GpuGate(foreground=lambda: False, utilisation=lambda: 99, util_threshold=0).busy()
    assert not GpuGate(foreground=lambda: False, utilisation=lambda: None, util_threshold=30).busy()


class _Svc:
    batch_size = 40

    def __init__(self):
        self.calls = []

    def translate(self, lang, texts, *, lang_name=None):
        self.calls.append(sorted(texts.values()))
        return TranslateResult(translations=dict(texts))


class _Gate:
    def __init__(self, busy):
        self._busy = busy

    def busy(self):
        return self._busy


def test_background_work_waits_while_the_gpu_is_wanted():
    svc = _Svc()
    f = Filler(svc, start_thread=False, gate=_Gate(True))
    f.enqueue("es", {"a": "bg"}, BACKGROUND)
    assert f.run_once() == 0 and svc.calls == [] and f.pending("es") == 1
    f.gate = _Gate(False)
    assert f.run_once() == 1 and svc.calls == [["bg"]]


def test_screen_work_does_not_wait():
    svc = _Svc()
    f = Filler(svc, start_thread=False, gate=_Gate(True))
    f.enqueue("es", {"a": "bg"}, BACKGROUND)
    f.enqueue("es", {"b": "scr"}, SCREEN)
    assert f.run_once() == 1 and svc.calls == [["scr"]]
    assert f.pending("es") == 1  # the background item still waits


def test_translation_calls_never_count_as_foreground(monkeypatch):
    from src.services.i18n.adapters import OllamaAdapter
    from src.services.i18n.settings import load_settings
    seen = {}
    OllamaAdapter(load_settings({}), generate=lambda p, **kw: seen.update(kw) or "{}").complete_json("p", {})
    assert seen["background"] is True
