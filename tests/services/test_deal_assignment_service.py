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


class _ScriptCursor:
    """Returns canned rows per matched SQL fragment; records writes & status updates."""
    def __init__(self, script, columns):
        self.script = script            # list of (substr, rows[list[dict]])
        self.columns = columns
        self.executed = []
        self._rows = []; self.description = None
    def execute(self, sql, params=()):
        s = " ".join(sql.split())
        self.executed.append((s, params))
        low = s.lower()
        if "information_schema.columns" in low:
            tbl = params[1]; self._rows = [(c,) for c in self.columns.get(tbl, [])]
            self.description = [("column_name",)]; return
        for substr, rows in self.script:
            if substr.lower() in low:
                self._rows = [tuple(r.values()) for r in rows]
                self.description = [(k,) for k in (rows[0].keys() if rows else [])]
                return
        self._rows = []; self.description = None
    def fetchall(self): return list(self._rows)
    def fetchone(self): return self._rows[0] if self._rows else None


def test_look_forward_links_monitor_deal_to_matching_invoice(monkeypatch):
    cols = ["invoice_id", "deal_id", "deal_name", "document_id", "deal_date"]
    monitor = [{"id": 720, "file_path": "documents/Invoice/THRIVE INV103404 for PO502001.pdf",
                "deal_id": "TEST00120260610104", "deal_name": "Test001",
                "category": "Invoice", "document_type": "pdf"}]
    inv = [{"invoice_id": "103404", "source_file": "x/THRIVE INV103404 for PO502001.pdf"}]
    cur = _ScriptCursor(
        script=[("from proc.process_monitor", monitor),
                ("from proc.bp_invoice_raw", inv),
                ("from proc.bp_invoice_trgt", inv)],
        columns={"bp_invoice_stg": cols, "bp_invoice_trgt": cols})
    conn = _RecConn(cur)
    n = das._look_forward(cur)
    assert n >= 1
    # deal columns written to trgt
    assert any("update proc.bp_invoice_trgt" in e[0].lower() and "deal_id" in e[0].lower()
               for e in cur.executed)
    # status advanced to Deal_Linked
    assert any("update proc.process_monitor set status" in e[0].lower()
               and e[1] and "Deal_Linked" in e[1] for e in cur.executed)


def test_look_forward_matches_by_process_monitor_id_over_filename():
    # raw row carries process_monitor_id=555 but its source_file basename does
    # NOT match the monitor's file_path — the exact pmid match must still win.
    monitor = [{"id": 555, "file_path": "documents/Invoice/renamed.pdf",
                "deal_id": "DEALX", "deal_name": "dx",
                "category": "Invoice", "document_type": "pdf"}]
    inv = [{"invoice_id": "INVX", "source_file": "store/totally-different.pdf",
            "process_monitor_id": 555}]
    cols = ["invoice_id", "deal_id", "deal_name", "document_id", "deal_date"]
    cur = _ScriptCursor(
        script=[("from proc.process_monitor", monitor),
                ("from proc.bp_invoice_raw", inv)],
        columns={"bp_invoice_stg": cols, "bp_invoice_trgt": cols})
    n = das._look_forward(cur)
    assert n == 1
    assert any("update proc.bp_invoice_trgt" in e[0].lower()
               and e[1] and "INVX" in str(e[1]) and "DEALX" in str(e[1])
               for e in cur.executed)


def test_look_back_joins_po_existing_deal_without_overwriting(monkeypatch):
    # A PO already carries an authoritative deal_id; an unlinked invoice on it
    # must JOIN that deal, and the PO must NOT be re-stamped with DEALV2-.
    inv = [{"invoice_id": "INV9", "po_id": "PO526702", "supplier_id": "SUP-Duncan", "deal_id": None}]
    po = [{"po_id": "526702", "supplier_id": "SUP-Duncan", "supplier_name": "Duncan LLC",
           "expected_delivery_date": None, "deal_id": "DEAL_A2026052891", "deal_name": "deal_a"}]
    cols = ["invoice_id", "deal_id", "deal_name", "document_id", "deal_date"]
    cur = _ScriptCursor(
        script=[("from proc.bp_invoice_trgt", inv),
                ("from proc.bp_purchase_order_trgt", po)],
        columns={"bp_invoice_trgt": cols, "bp_invoice_stg": cols})
    monkeypatch.setattr(das, "score_link", lambda *a, **k: {"F": 95.0})
    n = das._look_back(cur)
    assert n == 1
    # child invoice joined the PO's existing authoritative deal
    assert any("update proc.bp_invoice_trgt" in e[0].lower()
               and e[1] and "DEAL_A2026052891" in str(e[1]) for e in cur.executed)
    # the PO was NOT re-stamped, and no DEALV2 id was minted anywhere
    assert not any("update proc.bp_purchase_order_trgt" in e[0].lower() for e in cur.executed)
    assert not any(e[1] and "DEALV2-" in str(e[1]) for e in cur.executed)


def test_look_forward_deal_tagged_but_unmatched_is_deal_linked():
    # Monitor row carries a deal_id but no extracted doc matches by filename.
    # It must still be Deal_Linked (the deal is known), never Deal_Unassigned_Review.
    monitor = [{"id": 707, "file_path": "documents/po/DUNCAN PO526702.pdf",
                "deal_id": "DEAL_A2026052891", "deal_name": "deal_a",
                "category": "po", "document_type": "pdf"}]
    cur = _ScriptCursor(
        script=[("from proc.process_monitor", monitor),
                ("from proc.bp_purchase_order_raw", [])],  # no raw rows -> no match
        columns={})
    n = das._look_forward(cur)
    assert n == 0  # nothing stamped to _trgt
    status_writes = [e for e in cur.executed
                     if "update proc.process_monitor set status" in e[0].lower()]
    assert status_writes and status_writes[0][1] and status_writes[0][1][0] == "Deal_Linked"
    assert not any(e[1] and "Deal_Unassigned_Review" in str(e[1]) for e in status_writes)


def test_look_back_groups_invoice_under_canonical_po_deal(monkeypatch):
    # invoice with no deal, references PO 502001; a PO exists in trgt
    inv = [{"invoice_id": "103404", "po_id": "PO502001", "supplier_id": "SUP-Thrive",
            "deal_id": None}]
    po = [{"po_id": "502001", "supplier_id": "SUP-Thrive", "supplier_name": "Thrive Ltd",
           "expected_delivery_date": None, "deal_id": None}]
    columns = {"bp_invoice_stg": ["invoice_id", "deal_id", "deal_name", "document_id", "deal_date"],
               "bp_invoice_trgt": ["invoice_id", "deal_id", "deal_name", "document_id", "deal_date"]}
    cur = _ScriptCursor(
        script=[("from proc.bp_invoice_trgt", inv),
                ("from proc.bp_purchase_order_trgt", po)],
        columns=columns)
    # force the link score to pass
    monkeypatch.setattr(das, "score_link", lambda *a, **k: {"F": 95.0})
    n = das._look_back(cur)
    assert n >= 1
    assert any("dealv2-502001" in (str(e[1]).lower() if e[1] else "") for e in cur.executed)


def test_reconcile_legacy_binds_like_pattern_and_rewrites_to_dealv2():
    # Regression: a bare '%' in "LIKE 'DEAL-%'" is misread by psycopg2 as a
    # parameter placeholder. The pattern must be passed as a bind parameter.
    legacy = [{"invoice_id": "INV1", "deal_id": "DEAL-526702", "po_id": "PO526702"}]
    cur = _ScriptCursor(script=[("from proc.bp_invoice_trgt", legacy)], columns={})
    n = das._reconcile_legacy(cur)
    selects = [e for e in cur.executed
               if e[0].lower().startswith("select") and "like" in e[0].lower()]
    assert selects, "reconcile should issue a LIKE select"
    assert selects[0][1] == ("DEAL-%",), "LIKE pattern must be a bind param, not inline"
    assert any(e[0].lower().startswith("update") and e[1] and "DEALV2-526702" in str(e[1])
               for e in cur.executed)
    assert n >= 1


def test_backfill_deal_metadata_stamps_document_id_for_legacy_rows():
    # A reconciled row: has deal_id, but document_id is empty.
    legacy = [{"po_id": "526702", "deal_id": "DEALV2-526702", "deal_name": "Duncan LLC — PO 526702"}]
    cols = ["po_id", "deal_id", "deal_name", "document_id", "deal_date"]
    cur = _ScriptCursor(
        script=[("from proc.bp_purchase_order_trgt", legacy)],
        columns={"bp_purchase_order_trgt": cols})
    n = das._backfill_deal_metadata(cur)
    assert n >= 1
    # document_id stamped as <deal_id>::po::<pk>
    assert any("update proc.bp_purchase_order_trgt" in e[0].lower()
               and e[1] and "DEALV2-526702::po::526702" in str(e[1]) for e in cur.executed)
    # and recorded in the deal_document_map
    assert any("insert into proc.bp_deal_document_map" in e[0].lower() for e in cur.executed)


def test_propagate_deal_along_po_spreads_to_siblings_without_deal():
    # PO 888 carries a deal; the invoice on PO 888 has none -> it must inherit
    # the PO's deal. (The PO row's own deal_id is the known one.)
    cols = ["invoice_id", "deal_id", "deal_name", "document_id", "deal_date"]
    inv = [{"invoice_id": "INV1", "po_id": "PO888", "deal_id": None, "deal_name": None}]
    po = [{"po_id": "888", "deal_id": "DEALSIM", "deal_name": "Sim"}]
    cur = _ScriptCursor(
        script=[("select invoice_id, po_id, deal_id, deal_name from proc.bp_invoice_trgt", inv),
                ("select po_id, deal_id, deal_name from proc.bp_purchase_order_trgt", po)],
        columns={"bp_invoice_trgt": cols, "bp_invoice_stg": cols})
    n = das._propagate_deal_along_po(cur)
    assert n == 1
    # the invoice inherits DEALSIM
    assert any("update proc.bp_invoice_trgt" in e[0].lower()
               and e[1] and "DEALSIM" in str(e[1]) and "INV1" in str(e[1]) for e in cur.executed)


def test_propagate_deal_skips_conflicting_po_chain():
    # Two docs on the same PO carry DIFFERENT deals -> conflict, propagate nothing.
    cols = ["invoice_id", "deal_id", "deal_name", "document_id", "deal_date"]
    inv = [{"invoice_id": "INV1", "po_id": "PO888", "deal_id": "DEAL_A", "deal_name": "A"}]
    po = [{"po_id": "888", "deal_id": "DEAL_B", "deal_name": "B"}]
    cur = _ScriptCursor(
        script=[("select invoice_id, po_id, deal_id, deal_name from proc.bp_invoice_trgt", inv),
                ("select po_id, deal_id, deal_name from proc.bp_purchase_order_trgt", po)],
        columns={"bp_invoice_trgt": cols})
    n = das._propagate_deal_along_po(cur)
    assert n == 0
    assert not any(e[0].lower().startswith("update") for e in cur.executed)


def test_assign_deals_runs_all_passes_and_returns_counts(monkeypatch):
    calls = []
    monkeypatch.setattr(das, "_look_forward", lambda cur: calls.append("fwd") or 2)
    monkeypatch.setattr(das, "_look_back", lambda cur: calls.append("back") or 1)
    monkeypatch.setattr(das, "_reconcile_legacy", lambda cur: calls.append("rec") or 3)
    monkeypatch.setattr(das, "_propagate_deal_along_po", lambda cur: calls.append("prop") or 6)
    monkeypatch.setattr(das, "_backfill_deal_metadata", lambda cur: calls.append("meta") or 5)
    monkeypatch.setattr(das, "_flag_unassigned", lambda cur: calls.append("flag") or 4)
    cur = _ScriptCursor(script=[], columns={})
    conn = _RecConn(cur)
    result = das.assign_deals(conn=conn)
    assert result == {"forward_linked": 2, "backward_linked": 1, "reconciled": 3,
                      "propagated": 6, "metadata_filled": 5, "unassigned_review": 4}
    assert calls == ["fwd", "back", "rec", "prop", "meta", "flag"]
