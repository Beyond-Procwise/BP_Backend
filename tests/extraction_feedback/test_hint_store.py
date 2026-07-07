"""T3: ExtractionHintStore reads active, scoped hints from bp_prompt."""
from src.services.extraction_feedback.hint_store import HINT_STORE


def test_hints_for_returns_active_scoped(seed_hint):
    seed_hint(doc_type="invoice", vendor_key="NEXASPARK",
              hint_text="VAT appears as 'VAT @20%' in the totals block; capture as tax_amount",
              field_name="tax_amount")
    HINT_STORE.refresh()
    hints = HINT_STORE.hints_for("invoice", "NEXASPARK")
    assert any("VAT @20%" in h for h in hints)


def test_scope_isolation_and_case_insensitive(seed_hint):
    seed_hint(doc_type="invoice", vendor_key="NEXASPARK", hint_text="hint A")
    HINT_STORE.refresh()
    assert HINT_STORE.hints_for("invoice", "nexaspark")  # case-insensitive match
    assert HINT_STORE.hints_for("invoice", "OTHERCO") == []  # different vendor
    assert HINT_STORE.hints_for("quote", "NEXASPARK") == []  # different doc_type
    assert HINT_STORE.hints_for(None, "NEXASPARK") == []
