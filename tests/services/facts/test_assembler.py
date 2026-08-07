"""The Fact Assembler: _trgt rows in, provenanced facts out, nothing invented.

Per F1 the assembler drives from the _trgt line row and looks provenance up by
(doc_type, doc_pk, field_path). It never enumerates provenance and joins
forward — that direction looks catastrophic only because provenance also
records failed extraction attempts whose doc_pk never promoted.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.services.facts.assembler import (  # noqa: E402
    BASIS_UOM_UNSTATED,
    UOM_ABSENT,
    assemble_line_facts,
)
from src.services.facts.models import ArithmeticState, MeasureRole  # noqa: E402

D = Decimal
_T = datetime(2026, 5, 12, 10, 4, 28, tzinfo=timezone.utc)

# The real column list of proc.bp_invoice_line_items_trgt, trimmed to what the
# assembler reads. SELECT * drives off cursor.description, so the fake mirrors
# that contract.
LINE_COLS = ["invoice_id", "line_no", "item_id", "item_description", "quantity",
             "unit_of_measure", "unit_price", "line_amount", "tax_amount",
             "document_id"]

PROV_COLS = ["provenance_id", "doc_type", "doc_pk", "field_path", "value", "page",
             "bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1", "evidence_text",
             "model", "final_confidence", "extracted_at"]

HEADER_COLS = ["invoice_id", "currency", "supplier_id", "buyer_id", "contract_id",
               "deal_id", "document_id"]

FX_COLS = ["rate", "fetched_at", "base_currency"]


def _line(line_no=1, qty="2", price="86.94", amount="173.88", uom=None,
          invoice_id="INV-1"):
    return [invoice_id, line_no, "ITEM-1", "Laptop",
            None if qty is None else D(qty),
            uom,
            None if price is None else D(price),
            None if amount is None else D(amount),
            None, "DOC-1"]


def _prov_row(idx, field, value, pid=1):
    return [pid, "invoice", "INV-1", f"line_items[{idx}].{field}", value, 0,
            10.0, 20.0, 30.0, 40.0, value, "table_transformer", 0.85, _T]


class FakeCursor:
    """Dispatches on the SQL text, the way the real tables would."""

    def __init__(self, lines, provenance, header=None, fx=None):
        self._lines, self._prov = lines, provenance
        self._header = header if header is not None else ["INV-1", "GBP", "SUP-1", "BUY-1", None, None, "DOC-1"]
        self._fx = fx
        self.description, self._rows = [], []
        self.executed = []

    def execute(self, sql, params=None):
        self.executed.append((" ".join(sql.split()), params))
        s = sql.lower()
        if "bp_extraction_provenance_v3" in s:
            self.description = [(c,) for c in PROV_COLS]
            self._rows = list(self._prov)
        elif "line_items_trgt" in s:
            self.description = [(c,) for c in LINE_COLS]
            self._rows = list(self._lines)
        elif "bp_fx_rates" in s:
            self.description = [(c,) for c in FX_COLS]
            self._rows = list(self._fx or [])
        else:  # header
            self.description = [(c,) for c in HEADER_COLS]
            self._rows = [self._header] if self._header else []

    def fetchall(self):
        return list(self._rows)

    def fetchone(self):
        return self._rows[0] if self._rows else None


def _assemble(lines, prov, **kw):
    return assemble_line_facts(FakeCursor(lines, prov, **kw), "invoice", "INV-1")


def test_a_line_with_provenance_becomes_a_fact():
    facts = _assemble([_line()], [_prov_row(0, "unit_price", "86.94")])
    assert len(facts) == 1
    f = facts[0]
    assert f.provenance, "provenance must be non-empty by construction"
    assert f.unit_price == D("86.94")
    assert f.quantity == D("2")
    assert f.currency == "GBP"


def test_a_line_with_no_provenance_produces_no_fact(caplog):
    """F2's fail-closed behaviour. This must assert ABSENCE — a fact with empty
    provenance would satisfy a weaker version of this test while destroying the
    guarantee the whole phase rests on."""
    import logging
    with caplog.at_level(logging.INFO):
        facts = _assemble([_line()], [])
    assert facts == []
    assert any("provenance" in r.message.lower() for r in caplog.records), \
        "skipping a line for want of provenance must be logged, not silent"


def test_the_line_index_convention_is_zero_based_against_one_based_line_no():
    """Provenance writes line_items[0] for the first line; the _trgt tables
    number lines from 1. Verified against bp_sqldb: matching on line_no - 1
    returns the same unit_price as the _trgt row, while matching on line_no
    returns a different line's price. An off-by-one here silently attaches the
    wrong line's evidence to a price, which is worse than no evidence."""
    facts = _assemble([_line(line_no=3, price="55.00")],
                      [_prov_row(2, "unit_price", "55.00"),
                       _prov_row(3, "unit_price", "999.00", pid=2)])
    assert len(facts) == 1
    paths = {p.field_path for p in facts[0].provenance}
    assert "line_items[2].unit_price" in paths
    assert "line_items[3].unit_price" not in paths
    assert all(p.verbatim_snippet != "999.00" for p in facts[0].provenance)


def test_an_unmappable_uom_carries_forward_raw():
    facts = _assemble([_line(uom="30 days from quote date")],
                      [_prov_row(0, "unit_price", "86.94")])
    f = facts[0]
    assert f.uom == "30 days from quote date"
    assert f.uom_normalised is None
    assert "UOM_UNMAPPED" in f.reason_codes
    # never a normalised guess, and never NULL -- a NULL would make the model
    # reject a fact that genuinely exists
    assert f.basis_uom == "30 days from quote date"


def test_a_mappable_uom_is_normalised():
    facts = _assemble([_line(uom="EACH")], [_prov_row(0, "unit_price", "86.94")])
    f = facts[0]
    assert f.uom_normalised == "each"
    assert f.basis_uom == "each"
    assert "UOM_UNMAPPED" not in f.reason_codes


def test_an_absent_uom_is_recorded_as_unstated_not_guessed():
    """Measured on bp_sqldb: unit_of_measure is NULL on 152/152 invoice lines
    and 171/171 PO lines. Defaulting those to 'each' would fabricate a unit for
    the entire corpus; dropping them would discard every priced invoice fact.
    The honest record is an explicit sentinel that cannot be mistaken for a
    real unit, plus a reason code."""
    facts = _assemble([_line(uom=None)], [_prov_row(0, "unit_price", "86.94")])
    f = facts[0]
    assert f.uom is None
    assert f.uom_normalised is None
    assert f.basis_uom == BASIS_UOM_UNSTATED
    assert UOM_ABSENT in f.reason_codes

    from src.services.facts.uom import normalise_uom
    assert normalise_uom(BASIS_UOM_UNSTATED).canonical is None, \
        "the sentinel must not collide with a real unit"


def test_a_non_gbp_line_stamps_the_rate_date_and_source():
    facts = _assemble([_line()], [_prov_row(0, "unit_price", "86.94")],
                      header=["INV-1", "USD", "SUP-1", "BUY-1", None, None, "DOC-1"],
                      fx=[[D("0.743043"), _T, "USD"]])
    f = facts[0]
    assert f.currency == "USD"
    assert f.fx_rate == D("0.743043")
    assert f.fx_rate_date == _T
    assert f.fx_rate_source and "bp_fx_rates" in f.fx_rate_source


def test_an_unavailable_rate_is_null_not_a_guessed_parity():
    facts = _assemble([_line()], [_prov_row(0, "unit_price", "86.94")],
                      header=["INV-1", "XYZ", "SUP-1", "BUY-1", None, None, "DOC-1"],
                      fx=[])
    f = facts[0]
    assert f.fx_rate is None
    assert "FX_UNAVAILABLE" in f.reason_codes


def test_a_priced_line_is_a_unit_rate_with_a_checked_arithmetic_state():
    facts = _assemble([_line(qty="2", price="86.94", amount="173.88", uom="each")],
                      [_prov_row(0, "unit_price", "86.94")])
    f = facts[0]
    assert f.measure_role is MeasureRole.UNIT_RATE
    assert f.basis_uom == "each"
    assert f.arithmetic_state is ArithmeticState.CONSISTENT


def test_a_lump_sum_line_with_only_an_amount_is_an_extended_line():
    facts = _assemble([_line(qty=None, price=None, amount="58000.00")],
                      [_prov_row(0, "line_amount", "58000.00")])
    f = facts[0]
    assert f.measure_role is MeasureRole.EXTENDED_LINE
    assert f.unit_price is None
    assert f.extended_value == D("58000.00")


def test_an_inconsistent_line_still_produces_a_fact():
    """The whole point of the state is that the record carries its own
    reliability. Suppressing inconsistent lines would hide 7-22% of the corpus
    and quietly improve the apparent numbers."""
    facts = _assemble([_line(qty="40", price="4586.65", amount="4586.65")],
                      [_prov_row(0, "unit_price", "4586.65")])
    assert len(facts) == 1
    assert facts[0].arithmetic_state is ArithmeticState.INCONSISTENT


def test_a_quantity_one_line_is_recorded_as_untestable():
    facts = _assemble([_line(qty="1", price="500", amount="500")],
                      [_prov_row(0, "unit_price", "500")])
    assert facts[0].arithmetic_state is ArithmeticState.UNTESTABLE_QUANTITY_ONE


def test_provenance_carries_the_document_span_not_just_the_id():
    facts = _assemble([_line()], [_prov_row(0, "unit_price", "86.94")])
    p = facts[0].provenance[0]
    assert p.document_id
    assert p.locator.startswith("bbox:")
    assert p.verbatim_snippet == "86.94"
    assert p.page == 0
    assert p.extraction_id == "1"


def test_every_economic_field_with_evidence_is_attached():
    facts = _assemble([_line()],
                      [_prov_row(0, "unit_price", "86.94", pid=1),
                       _prov_row(0, "quantity", "2", pid=2),
                       _prov_row(0, "line_amount", "173.88", pid=3)])
    paths = {p.field_path for p in facts[0].provenance}
    assert paths == {"line_items[0].unit_price", "line_items[0].quantity",
                     "line_items[0].line_amount"}


def test_duplicate_provenance_attempts_resolve_to_one_row_deterministically():
    """The same (doc_pk, field_path) appears once per extraction attempt on
    bp_sqldb. Without a deterministic pick, two runs could stamp different
    evidence onto the same fact."""
    dup_low = _prov_row(0, "unit_price", "86.94", pid=1)
    dup_high = _prov_row(0, "unit_price", "86.94", pid=2)
    dup_high[12] = 0.99  # final_confidence
    facts = _assemble([_line()], [dup_low, dup_high])
    prov = [p for p in facts[0].provenance if p.field_path.endswith("unit_price")]
    assert len(prov) == 1
    assert prov[0].extraction_id == "2", "the highest-confidence attempt must win"


def test_the_assembler_never_enumerates_provenance_and_joins_forward():
    """F1: joining provenance -> _trgt drops most rows, because provenance also
    records failed attempts whose doc_pk was garbage. Every lookup must be
    keyed by the _trgt row's own (doc_type, doc_pk)."""
    cur = FakeCursor([_line()], [_prov_row(0, "unit_price", "86.94")])
    assemble_line_facts(cur, "invoice", "INV-1")
    prov_queries = [(s, p) for s, p in cur.executed if "bp_extraction_provenance_v3" in s]
    assert prov_queries, "provenance must be looked up"
    for sql, params in prov_queries:
        assert "doc_pk" in sql and "doc_type" in sql
        assert params and "INV-1" in params


def test_facts_are_deterministically_identified_so_a_rerun_does_not_duplicate():
    a = _assemble([_line()], [_prov_row(0, "unit_price", "86.94")])
    b = _assemble([_line()], [_prov_row(0, "unit_price", "86.94")])
    assert a[0].fact_id == b[0].fact_id


def test_an_unsupported_doc_type_is_refused_rather_than_guessed():
    with pytest.raises(ValueError):
        assemble_line_facts(FakeCursor([], []), "brochure", "X-1")
