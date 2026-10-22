"""One invoice number on two different documents (backlog 1.16, specs/2026-10-20-duplicate-invoice-same-reference-design.md).

The deliberate case: invoice 01-2024-002 arrived as 'MASTER Invoice for PO1.pdf' (Borcelle, 17,065.50) and
'PO1_IT_Invoice_LowerCost.pdf' (Borcelle Fauget, 15,877.50). Two invoices sent by accident that could both
have been paid. _stg/_trgt keep one row per number, so only proc.bp_invoice_raw holds both; the existing
detector also needs an identical total and skips equal numbers, so it could not see this.
"""
from datetime import date

import pytest

from src.services import duplicate_invoice_detector as did
from src.services.duplicate_invoice_detector import (
    DETECTOR_TYPE, find_reference_conflicts, supplier_family_match,
)


def _doc(first_raw, ref="01-2024-002", file="a.pdf", supplier="Borcelle", total=17065.50, text="t1",
         currency="GBP", day=date(2025, 6, 5), po=None, raw=None):
    return {"raw_id": raw or first_raw, "first_raw_id": first_raw, "invoice_id": ref, "source_file": file,
            "text_hash": text, "supplier_name": supplier, "supplier_id": None, "total_amount": total,
            "currency": currency, "invoice_date": day, "po_id": po, "deal_id": None}


MASTER = _doc(38819, file="documents/invoice/MASTER Invoice for PO1.pdf", supplier="Borcelle", total=17065.50, text="m")
LOWER = _doc(38753, file="documents/invoice/PO1_IT_Invoice_LowerCost.pdf", supplier="Borcelle Fauget",
             total=15877.50, text="l")


# ---- supplier agreement ---------------------------------------------------------------------------

def test_a_supplier_name_contained_in_another_is_the_same_supplier():
    assert supplier_family_match("Borcelle", "Borcelle Fauget")
    assert supplier_family_match("BORCELLE FAUGET LTD", "borcelle fauget")


def test_unrelated_suppliers_never_match_even_on_a_shared_suffix():
    assert not supplier_family_match("Acme Ltd", "Borcelle Ltd")
    assert not supplier_family_match("Techworld", "Tech World Services")
    assert not supplier_family_match("", "Borcelle")
    assert not supplier_family_match(None, None)
    assert not supplier_family_match("Ltd", "Ltd")        # nothing but a suffix says nothing
    assert not supplier_family_match("The Co", "The Foods Ltd")   # filler words prove nothing


def test_equal_supplier_ids_match_without_names():
    a, b = dict(MASTER, supplier_name=None, supplier_id="SUP-1"), dict(LOWER, supplier_name=None, supplier_id="SUP-1")
    assert find_reference_conflicts([a, b])


def test_a_contained_name_needs_dates_close_together_but_an_equal_name_does_not():
    far = date(2026, 1, 1)
    assert find_reference_conflicts([MASTER, dict(LOWER, invoice_date=far)]) == []            # 'Borcelle' ~ 'Borcelle Fauget', 7 months apart
    same_name = dict(LOWER, supplier_name="Borcelle", invoice_date=far)
    assert len(find_reference_conflicts([MASTER, same_name])) == 1                           # equal names: the number alone is enough
    assert len(find_reference_conflicts([MASTER, dict(LOWER, invoice_date=None)])) == 1      # absent date is not disagreement


# ---- the rule -------------------------------------------------------------------------------------

def test_the_deliberate_case_is_flagged_and_the_later_document_carries_it():
    out = find_reference_conflicts([MASTER, LOWER])
    assert len(out) == 1
    c = out[0]
    assert c["kind"] == "different_amount"
    assert c["earlier"]["source_file"].endswith("LowerCost.pdf")        # first ingested
    assert c["later"]["source_file"].endswith("MASTER Invoice for PO1.pdf")
    assert c["totals"] == [15877.50, 17065.50]
    assert c["difference"] == pytest.approx(1188.00)
    assert c["amount"] == 15877.50                                       # the least overpaid if both were paid


def test_input_order_does_not_change_the_answer():
    assert find_reference_conflicts([LOWER, MASTER])[0]["later"]["source_file"] == MASTER["source_file"]


def test_the_same_number_and_total_on_two_files_is_flagged_as_a_plain_duplicate():
    a = _doc(10, file="x.pdf", total=500.0, text="p")
    b = _doc(11, file="x-copy.pdf", total=500.0, text="q")
    out = find_reference_conflicts([a, b])
    assert [c["kind"] for c in out] == ["same_amount"]
    assert out[0]["amount"] == 500.0 and out[0]["difference"] == 0


def test_a_document_read_twice_is_not_a_second_document():
    # MASTER was read fourteen times; identical file and content is one document.
    again = dict(MASTER, raw_id=38952, first_raw_id=38819)
    assert find_reference_conflicts([MASTER, again]) == []


def test_the_same_file_read_again_with_different_text_is_still_one_document():
    # A reader fix changes the extracted text of the SAME file. That is one document, read twice;
    # accusing a supplier on the strength of two reads of one file would be a false finding.
    changed = dict(MASTER, raw_id=40000, first_raw_id=40000, text_hash="m2", total_amount=18000.0)
    assert find_reference_conflicts([MASTER, changed]) == []


def test_identical_stored_text_under_two_file_names_is_one_document_sent_twice_by_hand():
    a = _doc(1, file="documents/invoice/a.pdf", total=5.0, text="same text")
    b = _doc(2, file="documents/invoice/a-copy.pdf", total=5.0, text="same text")
    assert find_reference_conflicts([a, b]) == []


def test_two_files_with_no_stored_text_cannot_be_merged_on_content():
    # 99.7% of raw invoices have no stored text: an empty hash must never make two files "identical".
    a = _doc(1, file="a.pdf", total=5.0, text=None)
    b = _doc(2, file="b.pdf", total=5.0, text=None)
    assert [c["kind"] for c in find_reference_conflicts([a, b])] == ["same_amount"]


def test_different_suppliers_reusing_a_number_are_not_duplicates():
    other = _doc(20, file="o.pdf", supplier="Acme Ltd", total=17065.50, text="o")
    assert find_reference_conflicts([MASTER, other]) == []


def test_different_numbers_are_not_this_rule():
    assert find_reference_conflicts([MASTER, dict(LOWER, invoice_id="01-2024-003")]) == []


def test_credit_notes_and_unreadable_totals_are_never_compared():
    assert find_reference_conflicts([MASTER, dict(LOWER, total_amount=-15877.50)]) == []
    assert find_reference_conflicts([MASTER, dict(LOWER, total_amount=None)]) == []
    assert find_reference_conflicts([MASTER, dict(LOWER, total_amount=0)]) == []


def test_the_reference_is_compared_without_case_or_spacing_noise():
    assert len(find_reference_conflicts([MASTER, dict(LOWER, invoice_id=" 01-2024-002 ")])) == 1
    assert len(find_reference_conflicts([_doc(1, ref="INV 100", text="a"), _doc(2, ref="inv100", file="b.pdf", text="b",
                                                                                  total=900.0)])) == 1


def test_three_documents_with_one_number_raise_one_finding_against_the_earliest():
    third = _doc(50000, file="third.pdf", supplier="Borcelle", total=16000.0, text="t3")
    out = find_reference_conflicts([MASTER, LOWER, third])
    assert len(out) == 1
    assert out[0]["earlier"]["source_file"] == LOWER["source_file"] and out[0]["later"]["source_file"] == third["source_file"]
    assert len(out[0]["documents"]) == 3
    assert out[0]["amount"] == 15877.50


def test_totals_in_different_currencies_are_flagged_but_never_subtracted():
    usd = dict(LOWER, currency="USD")
    c = find_reference_conflicts([MASTER, usd])[0]
    assert c["kind"] == "different_amount" and c["difference"] is None
    assert "GBP" in did._conflict_note(c) and "USD" in did._conflict_note(c)


def test_the_reference_is_spelled_as_the_first_document_spelled_it():
    first = _doc(1, ref="INV-1", file="a.pdf", text="a", total=10.0)
    second = _doc(2, ref="inv-1", file="b.pdf", text="b", total=11.0)
    assert find_reference_conflicts([second, first])[0]["ref"] == "INV-1"      # stable: a later arrival can never change it


# ---- what is said and what is claimed -------------------------------------------------------------

def test_the_note_names_both_documents_both_totals_and_says_to_hold_payment():
    note = did._conflict_note(find_reference_conflicts([MASTER, LOWER])[0])
    assert "01-2024-002" in note
    assert "MASTER Invoice for PO1.pdf" in note and "PO1_IT_Invoice_LowerCost.pdf" in note
    assert "17,065.50" in note and "15,877.50" in note and "1,188.00" in note
    assert "hold" in note.lower()


def test_severity_is_critical_for_an_identical_total_and_warning_for_a_conflict():
    same = find_reference_conflicts([_doc(1, total=5.0, text="a"), _doc(2, file="b.pdf", total=5.0, text="b")])[0]
    diff = find_reference_conflicts([MASTER, LOWER])[0]
    assert did._conflict_severity(same) == "critical"
    assert did._conflict_severity(diff) == "warning"


def test_the_recovery_uses_the_lower_total_and_is_anchored_to_the_flagged_document():
    c = find_reference_conflicts([MASTER, LOWER])[0]
    rec = did._conflict_opportunity_record(c, {"USD": 1.0, "GBP": 1.0})
    assert rec["detector_type"] == DETECTOR_TYPE
    assert rec["opportunity_id"] == "dupref:01-2024-002"
    assert rec["invoice_id"] == "01-2024-002"
    assert rec["financial_impact_gbp"] == 15877.50
    d = rec["calculation_details"]
    assert d["kind"] == "different_amount" and d["higher_total"] == 17065.50 and d["lower_total"] == 15877.50
    assert d["payment_confirmed"] is False                              # never claims a payment it cannot see
    assert "if it was paid" in d["payment_note"]
    assert len(rec["source_records"]) >= 2


def test_a_payment_shown_on_either_document_counts():
    paid = dict(LOWER, invoice_paid_date=date(2025, 7, 1))
    rec = did._conflict_opportunity_record(find_reference_conflicts([MASTER, paid])[0], {"USD": 1.0, "GBP": 1.0})
    assert rec["calculation_details"]["payment_confirmed"] is True
    assert "2025-07-01" in rec["calculation_details"]["payment_note"]


def test_mixed_currencies_are_worded_as_up_to_never_as_a_difference():
    c = find_reference_conflicts([MASTER, dict(LOWER, currency="USD")])[0]
    rec = did._conflict_opportunity_record(c, {"USD": 1.0, "GBP": 0.8})
    assert rec["item_description"].startswith("Recover up to ")


def test_an_unconvertible_currency_gives_no_gbp_claim():
    c = find_reference_conflicts([dict(MASTER, currency="ZWL"), dict(LOWER, currency="ZWL")])[0]
    rec = did._conflict_opportunity_record(c, {"USD": 1.0, "GBP": 0.8})
    assert rec["financial_impact_gbp"] is None


# ---- the run: obeys the same idempotency, writes through the same tables ------------------------------

class _Cur:
    def __init__(self, ref_docs, already=False, pair_finding=False):
        self.ref_docs, self.already, self.pair_finding = ref_docs, already, pair_finding
        self.inserts, self._next = [], []

    def execute(self, sql, params=None):
        s = " ".join(sql.split())
        if "FROM proc.bp_invoice_trgt i" in s or "FROM proc.bp_invoice_line_items_trgt" in s:
            self._next = []
        elif "FROM proc.bp_invoice_raw" in s and "text_hash" in s:
            self._next = [tuple(d.get(k) for k in did._REF_DOC_COLUMNS) for d in self.ref_docs]
        elif "FROM proc.bp_extraction_discrepancy" in s:
            # the conflict kind asks by its own wording (LIKE); the pair scan asks by id alone
            raised = self.already if "LIKE" in s else self.pair_finding
            self._next = [(1,)] if raised else []
        elif "SELECT raw_id, source_file FROM proc.bp_invoice_raw" in s:
            self._next = [(38819, "documents/invoice/MASTER Invoice for PO1.pdf")]
        elif s.startswith("INSERT INTO proc.bp_extraction_discrepancy"):
            self.inserts.append(params)
        else:
            raise AssertionError("unexpected SQL: " + s[:90])

    def fetchall(self):
        return list(self._next)

    def fetchone(self):
        return self._next[0] if self._next else None


class _Conn:
    def __init__(self, cur):
        self._cur, self.committed = cur, False

    def cursor(self):
        return self._cur

    def commit(self):
        self.committed = True


@pytest.fixture()
def run(monkeypatch):
    upserts = []
    monkeypatch.setattr("src.services.opportunity_store.upsert_opportunity", lambda cur, rec: upserts.append(rec))
    monkeypatch.setattr(did, "_get_rates", lambda: {"USD": 1.0, "GBP": 1.0})

    def go(docs, already=False, pair_finding=False):
        cur = _Cur(docs, already, pair_finding)
        written = did._run(_Conn(cur))
        return written, cur.inserts, upserts
    return go


def test_the_run_writes_one_finding_and_one_recovery_for_the_deliberate_case(run):
    written, inserts, upserts = run([MASTER, LOWER])
    assert written == 1 and len(inserts) == 1
    raw_id, source_file, doc_pk, severity, raw_value, computed, note = inserts[0]
    assert doc_pk == "01-2024-002" and severity == "warning"
    assert source_file.endswith("MASTER Invoice for PO1.pdf")
    assert "17,065.50" in note
    assert [u["opportunity_id"] for u in upserts] == ["dupref:01-2024-002"]


def test_the_run_is_idempotent_and_never_re_raises_a_dismissed_finding(run):
    written, inserts, upserts = run([MASTER, LOWER], already=True)
    assert written == 0 and inserts == []
    assert len(upserts) == 1                      # the recovery is kept in step, as for every duplicate


def test_nothing_is_written_when_there_is_no_conflict(run):
    written, inserts, upserts = run([MASTER])
    assert written == 0 and inserts == [] and upserts == []


def test_a_pair_scan_finding_on_the_same_id_does_not_hide_the_same_number_finding(run):
    # They say different things; one must not silently suppress the other.
    written, inserts, _ = run([MASTER, LOWER], pair_finding=True)
    assert written == 1 and len(inserts) == 1


def test_the_flagged_document_s_own_read_is_what_the_finding_points_at(run):
    _w, inserts, _u = run([MASTER, LOWER])
    raw_id, source_file = inserts[0][0], inserts[0][1]
    assert raw_id == MASTER["raw_id"] and source_file == MASTER["source_file"]
