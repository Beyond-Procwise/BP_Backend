import datetime as dt
from src.services import deal_assignment_service as das


def test_basename_match_ignores_directory_and_case():
    assert das.basename_match("documents/po/DUNCAN PO526702.pdf", "/tmp/x/duncan po526702.PDF")
    assert not das.basename_match("a/INV1.pdf", "a/INV2.pdf")


def test_mint_document_id_is_deterministic_and_typed():
    a = das.mint_document_id("DEAL_A2026052891", "invoice", "INV610366")
    b = das.mint_document_id("DEAL_A2026052891", "invoice", "INV610366")
    assert a == b == "DEAL_A2026052891::invoice::INV610366"


def test_resolve_deal_date_prefers_po_expected_delivery():
    po = {"expected_delivery_date": dt.date(2024, 10, 9)}
    assert das.resolve_deal_date(po, inv_line_delivery=dt.date(2024, 11, 1)) == dt.date(2024, 10, 9)


def test_resolve_deal_date_falls_back_to_invoice_line_then_none():
    assert das.resolve_deal_date(None, inv_line_delivery=dt.date(2024, 11, 1)) == dt.date(2024, 11, 1)
    assert das.resolve_deal_date(None, inv_line_delivery=None) is None


def test_lookback_deal_identity_from_canonical_po():
    assert das.lookback_deal_id("526702") == "DEALV2-526702"
    assert das.lookback_deal_name("Duncan LLC", "526702") == "Duncan LLC — PO 526702"


class _RecCursor:
    def __init__(self, columns):
        self._columns = columns  # {table: [col,...]}
        self.executed = []       # (sql, params)
        self._result = []
        self.description = None
    def execute(self, sql, params=()):
        self.executed.append((" ".join(sql.split()), params))
        s = sql.lower()
        if "information_schema.columns" in s:
            tbl = params[1]
            self._result = [(c,) for c in self._columns.get(tbl, [])]
            self.description = [("column_name",)]
        else:
            self._result = []
            self.description = None
    def fetchall(self): return list(self._result)
    def fetchone(self): return self._result[0] if self._result else None

class _RecConn:
    def __init__(self, cur): self._cur = cur; self.autocommit = True
    def cursor(self): return self._cur
    def commit(self): pass
    def rollback(self): pass
    def close(self): pass


def test_persist_deal_writes_deal_cols_to_stg_and_trgt():
    cols = ["invoice_id", "deal_id", "deal_name", "document_id", "deal_date"]
    cur = _RecCursor({"bp_invoice_stg": cols, "bp_invoice_trgt": cols})
    das._persist_deal(cur, "invoice", "INV610366",
                      deal_id="DEAL_A2026052891", deal_name="deal_a",
                      document_id="DEAL_A2026052891::invoice::INV610366",
                      deal_date=None)
    updates = [e for e in cur.executed if e[0].lower().startswith("update")]
    assert any("bp_invoice_trgt" in e[0] and "deal_id" in e[0] for e in updates)
    assert any("bp_invoice_stg" in e[0] and "deal_id" in e[0] for e in updates)
