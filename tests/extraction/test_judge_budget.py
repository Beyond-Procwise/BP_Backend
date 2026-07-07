"""L3 judge loop is budgeted so a many-gaps document can't storm into minutes."""
from src.services.extraction import judge_gate as JG
from src.services.extraction.pattern_registry import get_registry


class _Parsed:
    full_text = "PURCHASE ORDER PO123 vendor Acme total 100 " * 8


def test_judge_respects_max_calls(monkeypatch):
    monkeypatch.setenv("EXTRACTION_JUDGE_MAX_CALLS", "3")
    monkeypatch.setenv("EXTRACTION_JUDGE_BUDGET_S", "999")
    calls = {"n": 0}

    def fake(field, doc_full_text, file_path=None):
        calls["n"] += 1
        return None  # judge finds nothing → all required fields stay gaps

    monkeypatch.setattr(JG, "call_grounded_last_resort", fake)
    reg = get_registry("purchase_order")
    JG.run_grounded_judge_for_gaps(parsed=_Parsed(), registry=reg,
                                   existing_candidates=[], file_path=None)
    assert calls["n"] <= 3, f"judge exceeded call budget: {calls['n']}"


def test_judge_respects_time_budget(monkeypatch):
    monkeypatch.setenv("EXTRACTION_JUDGE_MAX_CALLS", "999")
    monkeypatch.setenv("EXTRACTION_JUDGE_BUDGET_S", "0")  # zero budget → stop immediately
    calls = {"n": 0}

    def fake(field, doc_full_text, file_path=None):
        calls["n"] += 1
        return None

    monkeypatch.setattr(JG, "call_grounded_last_resort", fake)
    reg = get_registry("purchase_order")
    JG.run_grounded_judge_for_gaps(parsed=_Parsed(), registry=reg,
                                   existing_candidates=[], file_path=None)
    assert calls["n"] == 0, "zero time budget must make zero judge calls"
