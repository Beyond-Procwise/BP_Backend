"""Tests for the high-level TemplateService used by the orchestrator."""
from __future__ import annotations

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.services.extraction_v2.template_service import (  # noqa: E402
    TemplateService, configure_template_service,
)
from src.services.extraction_v2.template_store import (  # noqa: E402
    FieldHint, InMemoryTemplateStore, VendorTemplate,
)


def _service() -> TemplateService:
    return TemplateService(InMemoryTemplateStore())


def test_apply_template_overrides_supplier_name_when_template_present():
    svc = _service()
    svc.store.upsert(VendorTemplate(
        fingerprint="fp_x", vendor_name="Aquarius",
        doc_type="Quote",
        field_hints={
            "supplier_name": FieldHint(
                field="supplier_name", value="Aquarius Marketing Ltd",
                confidence=0.99,
            ),
        },
    ))
    header_in = {"supplier_name": "SUP-AuariusMarketing", "invoice_id": "I-1"}
    out, sources = svc.apply_template(header_in, "fp_x")
    assert out["supplier_name"] == "Aquarius Marketing Ltd"
    assert sources["supplier_name"] == "template"
    # invoice_id is not in the override list — left alone
    assert out["invoice_id"] == "I-1"


def test_apply_template_no_op_when_fingerprint_unknown():
    svc = _service()
    out, sources = svc.apply_template({"supplier_name": "X"}, "missing_fp")
    assert out == {"supplier_name": "X"}
    assert sources == {}


def test_apply_template_no_op_when_value_already_matches_template():
    svc = _service()
    svc.store.upsert(VendorTemplate(
        fingerprint="fp_y", vendor_name="ACME",
        doc_type="Invoice",
        field_hints={
            "supplier_name": FieldHint(
                field="supplier_name", value="ACME Co", confidence=0.99,
            ),
        },
    ))
    out, sources = svc.apply_template({"supplier_name": "ACME Co"}, "fp_y")
    # Same value → no source recorded (we didn't override anything)
    assert sources == {}
    assert out["supplier_name"] == "ACME Co"


def test_learn_from_extraction_writes_high_value_fields_only():
    svc = _service()
    learned = svc.learn_from_extraction(
        fingerprint="fp_learn1",
        header={
            "supplier_name": "Perry Manufacturing",
            "buyer_name": "WidgetCo",
            "invoice_id": "INV-100",  # should NOT be learned (not in override list)
            "garbage": "  ",          # too short
        },
        line_items=[{"item_description": "Widget", "quantity": 5}],
        doc_type="Invoice",
        vendor_name_hint="Perry",
    )
    assert learned is True
    tpl = svc.store.get("fp_learn1")
    assert tpl is not None
    assert tpl.vendor_name == "Perry"
    assert "supplier_name" in tpl.field_hints
    assert "buyer_name" in tpl.field_hints
    assert "invoice_id" not in tpl.field_hints  # not in override allow-list
    assert tpl.line_item_hints is not None
    assert "item_description" in tpl.line_item_hints.column_map


def test_learn_from_extraction_skips_garbage_values_url_or_email():
    svc = _service()
    learned = svc.learn_from_extraction(
        fingerprint="fp_learn2",
        header={"supplier_name": "nexasparkideas.com"},
        line_items=[],
        doc_type="Invoice",
    )
    # "nexasparkideas.com" looks like a URL — must not be learned
    tpl = svc.store.get("fp_learn2")
    if tpl is not None:
        assert "supplier_name" not in tpl.field_hints
    assert learned is False


def test_learn_from_extraction_does_not_overwrite_human_corrections():
    svc = _service()
    svc.store.record_correction(
        fingerprint="fp_corrected", field="supplier_name",
        value="Human-Approved Name", confidence=0.99,
        doc_type="Invoice", vendor_name="Trusted Vendor",
    )
    # Now an LLM-derived run tries to learn a different value
    learned = svc.learn_from_extraction(
        fingerprint="fp_corrected",
        header={"supplier_name": "LLM Guess"},
        line_items=[],
        doc_type="Invoice",
    )
    assert learned is False
    tpl = svc.store.get("fp_corrected")
    assert tpl is not None
    assert tpl.field_hints["supplier_name"].value == "Human-Approved Name"


def test_learn_then_apply_round_trip_fixes_supplier_name_for_next_doc():
    """Simulates the production flow: an extraction with filename rescue
    teaches a template. The next doc with the same fingerprint gets the
    correct supplier_name automatically (no LLM hallucination)."""
    svc = _service()

    # Doc 1: extraction produced garbage but filename rescued the value.
    # Orchestrator learns the rescued value as the template.
    svc.learn_from_extraction(
        fingerprint="fp_round_trip",
        header={"supplier_name": "Aquarius"},  # filename-rescued
        line_items=[],
        doc_type="Quote",
        vendor_name_hint="Aquarius",
        rescued_fields={"supplier_name"},
    )

    # Doc 2: LLM hallucinated again. Template applies the learned value.
    out, sources = svc.apply_template(
        {"supplier_name": "SUP-AuariusMarketing"}, "fp_round_trip",
    )
    assert out["supplier_name"] == "Aquarius"
    assert sources["supplier_name"] == "template"


def test_configure_with_get_conn_none_uses_in_memory_backend():
    svc = configure_template_service(get_conn=None)
    assert isinstance(svc.store, InMemoryTemplateStore)


def test_line_items_prompt_hint_returns_none_when_no_template():
    svc = _service()
    assert svc.line_items_prompt_hint("missing_fp") is None


def test_line_items_prompt_hint_returns_none_when_template_has_no_hints():
    from src.services.extraction_v2.template_store import VendorTemplate

    svc = _service()
    svc.store.upsert(VendorTemplate(
        fingerprint="no_lines", vendor_name="X", doc_type="Invoice",
    ))
    assert svc.line_items_prompt_hint("no_lines") is None


def test_learn_refines_line_item_columns_across_multiple_successes():
    """Auto-learn: each successful extraction unions new columns into
    the template's line_item_hints and never shrinks expected_min_rows."""
    svc = _service()

    # First run: observed columns = {item_description, quantity}, 1 line
    svc.learn_from_extraction(
        fingerprint="fp_refine",
        header={"supplier_name": "VendorX"},
        line_items=[{"item_description": "A", "quantity": 1}],
        doc_type="Invoice",
    )
    tpl1 = svc.store.get("fp_refine")
    assert tpl1 is not None and tpl1.line_item_hints is not None
    assert tpl1.line_item_hints.expected_min_rows == 1
    assert "item_description" in tpl1.line_item_hints.column_map
    assert "quantity" in tpl1.line_item_hints.column_map

    # Second run: NEW columns + more lines. Hints must REFINE.
    svc.learn_from_extraction(
        fingerprint="fp_refine",
        header={"supplier_name": "VendorX"},
        line_items=[
            {"item_description": "A", "quantity": 1, "unit_price": 10.0},
            {"item_description": "B", "quantity": 2, "unit_price": 20.0},
            {"item_description": "C", "quantity": 3, "unit_price": 30.0},
        ],
        doc_type="Invoice",
    )
    tpl2 = svc.store.get("fp_refine")
    assert tpl2 is not None and tpl2.line_item_hints is not None
    # expected_min_rows must NOT shrink — and must grow to observed max
    assert tpl2.line_item_hints.expected_min_rows == 3
    # The column map must now include unit_price
    assert "unit_price" in tpl2.line_item_hints.column_map
    # Older columns must still be present
    assert "item_description" in tpl2.line_item_hints.column_map
    assert "quantity" in tpl2.line_item_hints.column_map

    # Third run with FEWER lines must NOT shrink the floor
    svc.learn_from_extraction(
        fingerprint="fp_refine",
        header={"supplier_name": "VendorX"},
        line_items=[{"item_description": "A", "quantity": 1, "unit_price": 10.0}],
        doc_type="Invoice",
    )
    tpl3 = svc.store.get("fp_refine")
    assert tpl3 is not None and tpl3.line_item_hints is not None
    assert tpl3.line_item_hints.expected_min_rows == 3, (
        "expected_min_rows must not shrink; it's a floor for invariant checks"
    )


def test_legacy_extraction_prompt_context_combines_all_known_facts():
    from src.services.extraction_v2.template_store import (
        FieldHint, LineItemHints, VendorTemplate,
    )

    svc = _service()
    svc.store.upsert(VendorTemplate(
        fingerprint="legacy_full",
        vendor_name="Aquarius",
        doc_type="Invoice",
        field_hints={
            "supplier_name": FieldHint(
                field="supplier_name", value="Aquarius Marketing Ltd",
                confidence=0.99,
            ),
            "buyer_name": FieldHint(
                field="buyer_name", value="Assurity Ltd", confidence=0.99,
            ),
        },
        line_item_hints=LineItemHints(
            header_anchors=["Description", "Qty"],
            column_map={"Description": "item_description"},
            expected_min_rows=1,
        ),
    ))
    ctx = svc.legacy_extraction_prompt_context("legacy_full")
    assert ctx is not None
    assert "Aquarius" in ctx
    assert "Aquarius Marketing Ltd" in ctx
    assert "Assurity Ltd" in ctx
    assert "item_description" in ctx


def test_legacy_extraction_prompt_context_returns_none_when_unknown():
    svc = _service()
    assert svc.legacy_extraction_prompt_context("nope") is None


def test_line_items_prompt_hint_includes_vendor_name_and_columns():
    from src.services.extraction_v2.template_store import (
        LineItemHints, VendorTemplate,
    )

    svc = _service()
    svc.store.upsert(VendorTemplate(
        fingerprint="with_hints",
        vendor_name="Aquarius Marketing",
        doc_type="Invoice",
        line_item_hints=LineItemHints(
            header_anchors=["Description", "Qty", "Unit Price"],
            column_map={
                "Description": "item_description",
                "Qty": "quantity",
                "Unit Price": "unit_price",
                "Total": "line_amount",
            },
            expected_min_rows=2,
        ),
    ))
    hint = svc.line_items_prompt_hint("with_hints")
    assert hint is not None
    assert "Aquarius Marketing" in hint
    # All 4 columns should appear in the hint
    for col in ("item_description", "quantity", "unit_price", "line_amount"):
        assert col in hint
    # The header-anchor hints should be there too
    assert "Description" in hint
    assert "at least 2 line item" in hint
