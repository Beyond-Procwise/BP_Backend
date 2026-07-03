# tests/extraction/test_judge_backend_routing.py
"""The grounded last-resort judge must default to the unified AgentNick model
(Ollama text path), NOT Qwen2.5-VL. Qwen-VL is only used when explicitly opted
into via EXTRACTION_V3_JUDGE_MODEL=qwen.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.services.extraction_v3.judge import grounded_last_resort as glr  # noqa: E402


def _patch_backends(monkeypatch):
    calls = {"ollama": 0, "qwen": 0}
    monkeypatch.setattr(glr, "_call_ollama_grounded",
                        lambda *a, **k: calls.__setitem__("ollama", calls["ollama"] + 1))
    monkeypatch.setattr(glr, "_call_qwen_grounded",
                        lambda *a, **k: calls.__setitem__("qwen", calls["qwen"] + 1))
    return calls


def test_default_routes_to_agentnick_not_qwen(monkeypatch):
    monkeypatch.delenv("EXTRACTION_V3_JUDGE_MODEL", raising=False)
    calls = _patch_backends(monkeypatch)
    glr.call_grounded_last_resort(SimpleNamespace(name="invoice_id"), "some doc text")
    assert calls["ollama"] == 1, "default judge must use the Ollama/AgentNick path"
    assert calls["qwen"] == 0, "Qwen-VLM must not be invoked by default"


def test_explicit_qwen_still_opt_inable(monkeypatch):
    monkeypatch.setenv("EXTRACTION_V3_JUDGE_MODEL", "qwen")
    calls = _patch_backends(monkeypatch)
    glr.call_grounded_last_resort(SimpleNamespace(name="invoice_id"), "some doc text")
    assert calls["qwen"] == 1 and calls["ollama"] == 0
