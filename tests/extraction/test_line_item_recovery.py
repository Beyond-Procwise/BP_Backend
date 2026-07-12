# tests/extraction/test_line_item_recovery.py
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.services.extraction import context_layer  # noqa: E402


FULL_TEXT = (
    "ACME LTD INVOICE INV9\n"
    "Design services - phase 1   600.00\n"
    "Hosting - annual            400.00\n"
    "Subtotal 1000.00\n"
)


def test_recovers_grounded_line_items(monkeypatch):
    monkeypatch.setattr(context_layer, "_call_llm", lambda prompt, **_kw: (
        '[{"description":"Design services - phase 1","quantity":1,"unit_price":600.0,"amount":600.0},'
        '{"description":"Hosting - annual","quantity":1,"unit_price":400.0,"amount":400.0}]'
    ))
    items = context_layer.synthesize_line_items("invoice", FULL_TEXT, header_subtotal=1000.0)
    assert len(items) == 2
    assert items[0]["description"] == "Design services - phase 1"
    assert items[0]["amount"] == 600.0
    assert sum(i["amount"] for i in items) == 1000.0


def test_drops_ungrounded_rows(monkeypatch):
    # "Consulting fee" is NOT in FULL_TEXT -> must be dropped (no fabrication).
    monkeypatch.setattr(context_layer, "_call_llm", lambda prompt, **_kw: (
        '[{"description":"Consulting fee","amount":999.0},'
        '{"description":"Hosting - annual","amount":400.0}]'
    ))
    items = context_layer.synthesize_line_items("invoice", FULL_TEXT, header_subtotal=400.0)
    assert len(items) == 1
    assert items[0]["description"] == "Hosting - annual"


def test_llm_failure_returns_empty(monkeypatch):
    def _boom(prompt, **_kw):
        raise RuntimeError("ollama down")
    monkeypatch.setattr(context_layer, "_call_llm", _boom)
    assert context_layer.synthesize_line_items("invoice", FULL_TEXT) == []


def test_empty_text_returns_empty():
    assert context_layer.synthesize_line_items("invoice", "") == []


def test_non_json_response_returns_empty(monkeypatch):
    monkeypatch.setattr(context_layer, "_call_llm", lambda prompt: "sorry, no tables here")
    assert context_layer.synthesize_line_items("invoice", FULL_TEXT) == []


def test_drops_row_with_ungrounded_amount(monkeypatch):
    # Description IS in the text but the amount (777.0) is NOT -> drop it.
    monkeypatch.setattr(context_layer, "_call_llm", lambda prompt, **_kw: (
        '[{"description":"Hosting - annual","amount":777.0}]'
    ))
    items = context_layer.synthesize_line_items("invoice", FULL_TEXT)
    assert items == []
