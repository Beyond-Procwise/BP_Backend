"""Tests for ModelSyncService prompt-pollution guards.

Covers (1) the supplier-name sanitiser that keeps garbage extractions out of the
Modelfile prompt, and (2) the inject step removing BOTH legacy learned-pattern
section markers so they cannot accumulate.
"""
from __future__ import annotations

import src.services.model_sync_service as ms


# ---------------------------------------------------------------------------
# is_valid_supplier_name
# ---------------------------------------------------------------------------
def test_rejects_phone_number():
    assert ms.is_valid_supplier_name("800-531-8575") is False


def test_rejects_column_headers_and_fragments():
    for bad in ["Description Qty Unit Price", "days Tax", "Delivery Deadline",
                "GOMEZ, GOOD AND CROSS QUOTE TRADING LTD. UNKNOWN", "CONDITIONS"]:
        assert ms.is_valid_supplier_name(bad) is False, bad


def test_rejects_id_like_tokens():
    for bad in ["INV600820", "INV600784", "PO507269", "QUT110500"]:
        assert ms.is_valid_supplier_name(bad) is False, bad


def test_rejects_empty_and_too_long():
    assert ms.is_valid_supplier_name("") is False
    assert ms.is_valid_supplier_name("  ") is False
    assert ms.is_valid_supplier_name("x" * 80) is False


def test_accepts_real_suppliers():
    for ok in ["TechNova Ltd", "Gomez, Good and Cross Trading Ltd", "NexaSpark",
               "Dell Workspace Solutions Ltd", "Duncan LLC", "Infotech"]:
        assert ms.is_valid_supplier_name(ok) is True, ok


# ---------------------------------------------------------------------------
# _inject_vendor_knowledge — both legacy markers removed, single section kept
# ---------------------------------------------------------------------------
def _svc():
    return ms.ModelSyncService(agent_nick=None)


def test_inject_removes_both_legacy_sections():
    modelfile = (
        'SYSTEM """You are AgentNick.\n\n'
        "=== LEARNED VENDOR PATTERNS (auto-updated) ===\n\n"
        "LEARNED VENDOR PATTERNS (from successful extractions):\n- OldGarbage (Invoice)\n\n"
        "=== LEARNED PATTERNS (auto-updated) ===\n\n"
        "LEARNED VENDOR PATTERNS (from successful extractions):\n- AlsoOld (Quote)\n\n"
        "=== QUALITY STANDARDS ===\nstuff\"\"\"\n"
    )
    out = _svc()._inject_vendor_knowledge(modelfile, "LEARNED VENDOR PATTERNS (from successful extractions):\n- NewClean (Invoice), extractions=5")

    # Neither legacy section header survives as a duplicate
    assert out.count("=== LEARNED VENDOR PATTERNS (auto-updated) ===") == 0
    assert out.count("=== LEARNED PATTERNS (auto-updated) ===") == 1
    assert "OldGarbage" not in out
    assert "AlsoOld" not in out
    assert "NewClean" in out
    assert "=== QUALITY STANDARDS ===" in out


def test_inject_noop_when_empty():
    modelfile = 'SYSTEM """x=== QUALITY STANDARDS ==="""'
    assert _svc()._inject_vendor_knowledge(modelfile, "") == modelfile
