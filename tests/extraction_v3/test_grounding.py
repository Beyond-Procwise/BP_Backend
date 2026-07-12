"""Tests for the extraction_v3 grounding guard.

The guard decides whether a committed field's value is *grounded* in the source
document. It must block genuine hallucinations (values absent from the document)
while preserving correctly-extracted values whose evidence_text was reformatted
(whitespace/newline differences, ISO-normalized dates, decimal-normalized
amounts). See FINDINGS.md F1.
"""
from src.services.extraction_v3.grounding import (
    is_value_grounded,
    ground_committed_fields,
)
from src.services.extraction_v3.schemas.result import CommittedField, ResidualField


def _cf(field_path="invoice_date", value="2024-02-15", evidence_text="2024-02-15",
        model="regex", conf=0.72) -> CommittedField:
    return CommittedField(
        field_path=field_path, value=value, page=1, bbox=(0.0, 0.0, 0.0, 0.0),
        evidence_text=evidence_text, model=model, model_confidence=conf,
        judge_actions=[], final_confidence=conf,
    )


FT = (
    "ACME Industries Ltd\n"
    "Invoice Number: INV-039469\n"
    "Invoice Date: 15 Feb 2024\n"
    "Net (ex-VAT) £27,202.00\n"
    "Grand Total £32,642.40\n"
)


class TestIsValueGrounded:
    def test_exact_evidence_substring_is_grounded(self):
        assert is_value_grounded("INV-039469", "Invoice Number: INV-039469", FT) is True

    def test_newline_in_evidence_still_grounded(self):
        # evidence has a newline the flat full_text join doesn't; normalized match wins
        assert is_value_grounded("£27,202.00", "Net (ex-VAT)\n£27,202.00", FT) is True

    def test_iso_date_grounded_against_textual_doc_date(self):
        # value normalized to ISO; document says "15 Feb 2024"
        assert is_value_grounded("2024-02-15", "2024-02-15", FT) is True

    def test_iso_date_grounded_against_nonpadded_and_dotted_doc_dates(self):
        # single-digit day, no zero-padding, dotted separator
        assert is_value_grounded("2024-02-05", "2024-02-05", "paid on 5 Feb 2024") is True
        assert is_value_grounded("2024-02-05", "2024-02-05", "date: 5/2/2024") is True
        assert is_value_grounded("2024-02-15", "2024-02-15", "Dated 15.02.2024") is True

    def test_noniso_date_value_cross_renders(self):
        # value stored day-first; document writes it textually (and vice-versa)
        assert is_value_grounded("15/02/2024", "15/02/2024", "Invoice Date: 15 Feb 2024") is True
        assert is_value_grounded("3 July 2026", "3 July 2026", "Date 03/07/2026") is True

    def test_decimal_amount_grounded_against_formatted_doc_amount(self):
        # value stripped to plain decimal; doc shows "£27,202.00"
        assert is_value_grounded("27202.0", "27202.0", FT) is True

    def test_value_present_even_if_evidence_missing(self):
        assert is_value_grounded("15 Feb 2024", "", FT) is True

    def test_genuine_fabrication_is_not_grounded(self):
        # A supplier name that appears nowhere in the document
        assert is_value_grounded("Globex Corporation", "Globex Corporation", FT) is False

    def test_synthetic_evidence_is_exempt(self):
        assert is_value_grounded("SYN-ACME-2024-02-ab12cd",
                                 "synthetic_invoice_id:SYN-ACME-2024-02-ab12cd", FT) is True

    def test_pipeline_recovery_derivation_is_exempt(self):
        # deterministic derived value (e.g. subtotal = total/(1+tax)) need not appear verbatim
        assert is_value_grounded("30000.0", "", FT, model="pipeline_recovery") is True

    def test_empty_full_text_cannot_verify_so_not_blocked(self):
        assert is_value_grounded("anything", "anything", "") is True


class TestGroundCommittedFields:
    def test_ungrounded_field_is_demoted_to_residual(self):
        committed = [
            _cf(field_path="supplier_name", value="Globex Corporation",
                evidence_text="Globex Corporation", model="judge"),
            _cf(field_path="invoice_date", value="2024-02-15", evidence_text="2024-02-15"),
        ]
        residuals: list[ResidualField] = []
        kept, resid = ground_committed_fields(committed, residuals, FT)
        kept_paths = {c.field_path for c in kept}
        resid_paths = {r.field_path for r in resid}
        assert "invoice_date" in kept_paths           # correct ISO date preserved
        assert "supplier_name" not in kept_paths       # fabrication blocked
        assert "supplier_name" in resid_paths
        assert all(r.reason == "ungrounded_value" for r in resid)

    def test_all_grounded_fields_are_preserved(self):
        committed = [
            _cf(field_path="invoice_id", value="INV-039469",
                evidence_text="Invoice Number: INV-039469", model="vlm"),
            _cf(field_path="invoice_amount", value="27202.0", evidence_text="27202.0"),
        ]
        kept, resid = ground_committed_fields(committed, [], FT)
        assert len(kept) == 2
        assert resid == []
