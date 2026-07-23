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
        # Amend-mode: TEST00120260610104 is an established deal (bp_deal row), so
        # the monitor's deal is authoritative and must still be stamped.
        script=[("from proc.bp_deal where deal_id", [{"deal_id": "TEST00120260610104"}]),
                ("from proc.process_monitor", monitor),
                ("from proc.bp_invoice_raw", inv),
                ("from proc.bp_invoice_trgt", inv)],
        columns={"bp_invoice_stg": cols, "bp_invoice_trgt": cols})
    conn = _RecConn(cur)
    n = das._look_forward(cur)
    assert n >= 1
    # deal columns written to trgt
    assert any("update proc.bp_invoice_trgt" in e[0].lower() and "deal_id" in e[0].lower()
               for e in cur.executed)
    # status is owned by reconcile_status now — look_forward must NOT write status
    assert not any("update proc.process_monitor set status" in e[0].lower()
                   for e in cur.executed)


def test_ensure_in_trgt_copies_staged_doc_when_absent():
    # A PO-less quote is in _stg but not _trgt -> deal-path promotion copies it
    # into _trgt (without the deal columns, which _look_forward stamps later).
    cols = ["quote_id", "po_id", "supplier_id", "total_amount",
            "deal_id", "deal_name", "document_id", "deal_date"]
    staged = [{"quote_id": "Q1", "po_id": None, "supplier_id": "SUP-X", "total_amount": 100}]
    cur = _ScriptCursor(
        script=[("select * from proc.bp_quote_stg", staged)],
        columns={"bp_quote_stg": cols, "bp_quote_trgt": cols})
    assert das._ensure_in_trgt(cur, "quote", "Q1") is True
    inserts = [e for e in cur.executed if e[0].lower().startswith("insert into proc.bp_quote_trgt")]
    assert inserts, "staged quote should be inserted into _trgt"
    assert "deal_id" not in inserts[0][0].lower()  # deal cols excluded from the copy


def test_ensure_in_trgt_noops_when_not_staged():
    # Doc not in _trgt and not in _stg (e.g. held at raw by a discrepancy) -> no copy.
    cur = _ScriptCursor(script=[], columns={"bp_invoice_stg": ["invoice_id"], "bp_invoice_trgt": ["invoice_id"]})
    assert das._ensure_in_trgt(cur, "invoice", "INV404") is False
    assert not any(e[0].lower().startswith("insert into proc.bp_invoice_trgt") for e in cur.executed)


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
        # Amend-mode: DEALX is an established deal (bp_deal row).
        script=[("from proc.bp_deal where deal_id", [{"deal_id": "DEALX"}]),
                ("from proc.process_monitor", monitor),
                ("from proc.bp_invoice_raw", inv)],
        columns={"bp_invoice_stg": cols, "bp_invoice_trgt": cols})
    n = das._look_forward(cur)
    assert n == 1
    assert any("update proc.bp_invoice_trgt" in e[0].lower()
               and e[1] and "INVX" in str(e[1]) and "DEALX" in str(e[1])
               for e in cur.executed)


def test_look_forward_does_not_stamp_unestablished_batch_label():
    # The monitor's deal_id is a freshly-typed upload-batch label -- not yet a
    # bp_deal row, not a confirmed proposal. The matching doc must be left
    # UNLINKED (no deal_id stamp) so it flows into the clustering/proposal
    # pipeline instead of being auto-grouped under the batch label.
    monitor = [{"id": 900, "file_path": "documents/Invoice/BATCH INV1.pdf",
                "deal_id": "ANALYSISSET_19072620260719339", "deal_name": "New upload batch",
                "category": "Invoice", "document_type": "pdf"}]
    inv = [{"invoice_id": "INV1", "source_file": "x/BATCH INV1.pdf"}]
    cols = ["invoice_id", "deal_id", "deal_name", "document_id", "deal_date"]
    cur = _ScriptCursor(
        # No "from proc.bp_deal where deal_id" / confirmed-proposal rows scripted
        # -> is_established_deal(cur, "ANALYSISSET_...") is False (the default).
        script=[("from proc.process_monitor", monitor),
                ("from proc.bp_invoice_raw", inv),
                ("from proc.bp_invoice_trgt", inv)],
        columns={"bp_invoice_stg": cols, "bp_invoice_trgt": cols})
    n = das._look_forward(cur)
    assert n == 0  # nothing stamped -- doc left unlinked for clustering
    assert not any("update proc.bp_invoice_trgt" in e[0].lower() and e[1]
                   and "ANALYSISSET_19072620260719339" in str(e[1])
                   for e in cur.executed)


_QA_COLS = ["po_id", "quote_id", "invoice_id", "deal_id", "deal_name", "document_id", "deal_date"]
_QA_COLMAP = {k: _QA_COLS for k in ("bp_quote_trgt", "bp_quote_stg", "bp_invoice_trgt",
                                    "bp_invoice_stg", "bp_purchase_order_trgt",
                                    "bp_purchase_order_stg")}


def test_look_back_joins_existing_po_deal_when_quote_anchors():
    # PO carries an authoritative deal; a quote (explicit po_id ref) anchors it.
    # The quote + invoice JOIN the PO's deal; the PO is not re-stamped; no DEALV2 minted.
    po = [{"po_id": "526702", "supplier_id": "SUP-Duncan", "supplier_name": "Duncan LLC",
           "expected_delivery_date": None, "deal_id": "DEAL_A2026052891", "deal_name": "deal_a"}]
    quote = [{"quote_id": "QUT1", "po_id": "526702", "supplier_id": "SUP-Duncan",
              "deal_id": None, "deal_name": None}]
    inv = [{"invoice_id": "INV9", "deal_id": None}]
    cur = _ScriptCursor(
        script=[("from proc.bp_purchase_order_trgt", po),
                ("from proc.bp_quote_trgt", quote),
                ("from proc.bp_invoice_trgt", inv)],
        columns=_QA_COLMAP)
    das._look_back(cur)
    assert any("update proc.bp_quote_trgt" in e[0].lower() and "DEAL_A2026052891" in str(e[1])
               for e in cur.executed)
    assert any("update proc.bp_invoice_trgt" in e[0].lower() and "DEAL_A2026052891" in str(e[1])
               for e in cur.executed)
    assert not any("update proc.bp_purchase_order_trgt" in e[0].lower() for e in cur.executed)
    assert not any(e[1] and "DEALV2-" in str(e[1]) for e in cur.executed)


def test_look_forward_does_not_cross_claim_pmid_raw_by_basename():
    # Same filename, two monitor rows, different deals; raw points to monitor 100
    # via process_monitor_id. Monitor 200 must NOT claim it by basename.
    monitors = [
        {"id": 100, "file_path": "d/SAME.pdf", "deal_id": "DEAL_RIGHT", "deal_name": "right",
         "category": "Invoice", "document_type": "pdf"},
        {"id": 200, "file_path": "d/SAME.pdf", "deal_id": "DEAL_WRONG", "deal_name": "wrong",
         "category": "Invoice", "document_type": "pdf"},
    ]
    inv = [{"invoice_id": "INV1", "source_file": "x/SAME.pdf", "process_monitor_id": 100}]
    cols = ["invoice_id", "deal_id", "deal_name", "document_id", "deal_date"]
    cur = _ScriptCursor(
        # Amend-mode: DEAL_RIGHT is an established deal (bp_deal row). DEAL_WRONG
        # never reaches the established check (it doesn't claim any doc), so it
        # doesn't need a bp_deal row.
        script=[("from proc.bp_deal where deal_id", [{"deal_id": "DEAL_RIGHT"}]),
                ("from proc.process_monitor", monitors),
                ("from proc.bp_invoice_raw", inv)],
        columns={"bp_invoice_stg": cols, "bp_invoice_trgt": cols})
    das._look_forward(cur)
    inv_updates = [e for e in cur.executed
                   if "update proc.bp_invoice_trgt" in e[0].lower() and e[1]]
    assert any("DEAL_RIGHT" in str(e[1]) for e in inv_updates)
    assert not any("DEAL_WRONG" in str(e[1]) for e in inv_updates)


def test_look_forward_unmatched_is_noop():
    # Monitor row carries a deal_id but no extracted doc matches -> look_forward
    # does nothing (no stamp, no status write). Status is reconcile_status's job;
    # the unmatched doc just isn't in _trgt yet, so it stays at its raw/stg stage.
    monitor = [{"id": 707, "file_path": "documents/po/DUNCAN PO526702.pdf",
                "deal_id": "DEAL_A2026052891", "deal_name": "deal_a",
                "category": "po", "document_type": "pdf"}]
    cur = _ScriptCursor(
        script=[("from proc.process_monitor", monitor),
                ("from proc.bp_purchase_order_raw", [])],  # no raw rows -> no match
        columns={})
    n = das._look_forward(cur)
    assert n == 0  # nothing stamped to _trgt
    assert not any("update proc.process_monitor set status" in e[0].lower()
                   for e in cur.executed)


def test_look_back_forms_dealv2_when_quote_anchors():
    # No existing deal; an anchoring quote forms DEALV2-<po> for quote + PO + invoice.
    po = [{"po_id": "502001", "supplier_id": "SUP-Thrive", "supplier_name": "Thrive Ltd",
           "expected_delivery_date": None, "deal_id": None, "deal_name": None}]
    quote = [{"quote_id": "Q41", "po_id": "502001", "supplier_id": "SUP-Thrive",
              "deal_id": None, "deal_name": None}]
    inv = [{"invoice_id": "103404", "deal_id": None}]
    cur = _ScriptCursor(
        script=[("from proc.bp_purchase_order_trgt", po),
                ("from proc.bp_quote_trgt", quote),
                ("from proc.bp_invoice_trgt", inv)],
        columns=_QA_COLMAP)
    n = das._look_back(cur)
    assert n == 3   # quote + PO + invoice
    for tbl in ("bp_quote_trgt", "bp_purchase_order_trgt", "bp_invoice_trgt"):
        assert any(f"update proc.{tbl}" in e[0].lower() and "DEALV2-502001" in str(e[1])
                   for e in cur.executed), tbl


def test_look_back_no_deal_without_quote_anchor(monkeypatch):
    # PO present but NO quote anchors it (no explicit ref, score below bar) ->
    # no deal minted; the PO + its invoices stay orphaned.
    po = [{"po_id": "519829", "supplier_id": "SUP-X", "supplier_name": "X",
           "expected_delivery_date": None, "deal_id": None, "deal_name": None}]
    cur = _ScriptCursor(
        script=[("from proc.bp_purchase_order_trgt", po),
                ("where supplier_id", [{"quote_id": "Qz", "supplier_id": "SUP-X", "po_id": None}])],
        columns=_QA_COLMAP)
    monkeypatch.setattr(das, "score_link", lambda *a, **k: {"F": 10.0})  # below QUOTE_ANCHOR_MIN_SCORE
    assert das._look_back(cur) == 0
    assert not any(e[1] and "DEALV2-" in str(e[1]) for e in cur.executed)


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


def test_flag_conflict_po_chains_sets_review_status_on_mismatched_deals():
    # PO 888 has an invoice tagged DEAL_A and the PO tagged DEAL_B -> conflict.
    inv = [{"invoice_id": "INV1", "po_id": "PO888", "deal_id": "DEAL_A", "deal_name": "A"}]
    po = [{"po_id": "888", "deal_id": "DEAL_B", "deal_name": "B"}]
    inv_raw = [{"invoice_id": "INV1", "process_monitor_id": 11}]
    po_raw = [{"po_id": "888", "process_monitor_id": 22}]
    cur = _ScriptCursor(
        script=[("select invoice_id, po_id, deal_id, deal_name from proc.bp_invoice_trgt", inv),
                ("select po_id, deal_id, deal_name from proc.bp_purchase_order_trgt", po),
                ("from proc.bp_invoice_raw", inv_raw),
                ("from proc.bp_purchase_order_raw", po_raw)],
        columns={})
    n = das._flag_conflict_po_chains(cur)
    assert n == 2  # both monitor rows flagged
    statuses = [e for e in cur.executed
                if "update proc.process_monitor set status" in e[0].lower()]
    assert all("Deal_Conflict_Review" in str(e[1]) for e in statuses)
    flagged_ids = {e[1][1] for e in statuses}
    assert flagged_ids == {11, 22}


def test_flag_conflict_po_chains_noop_when_single_deal():
    inv = [{"invoice_id": "INV1", "po_id": "PO888", "deal_id": "DEAL_A", "deal_name": "A"}]
    po = [{"po_id": "888", "deal_id": "DEAL_A", "deal_name": "A"}]
    cur = _ScriptCursor(
        script=[("select invoice_id, po_id, deal_id, deal_name from proc.bp_invoice_trgt", inv),
                ("select po_id, deal_id, deal_name from proc.bp_purchase_order_trgt", po)],
        columns={})
    assert das._flag_conflict_po_chains(cur) == 0
    assert not any("update proc.process_monitor set status" in e[0].lower() for e in cur.executed)


def test_reconcile_status_emits_stage_based_updates():
    class _Cur:
        def __init__(self):
            self.sql = []
            self.rowcount = 0
        def execute(self, sql, params=()):
            self.sql.append(" ".join(sql.split()))
    cur = _Cur()
    das.reconcile_status(cur)
    updates = [s for s in cur.sql if s.lower().startswith("update proc.process_monitor")]
    assert len(updates) == 3                      # one per doc type
    joined = " ".join(updates).lower()
    for st in ("deal_linked", "orphaned_awaiting_quote", "deal_unassigned_review",
               "staged", "discrepancy_review", "extracted"):
        assert st in joined                       # full quote-anchored lifecycle covered
    assert "extraction_failed" in joined and "deal_conflict_review" in joined  # preserved
    # orphan rule: a PO/invoice is complete only when its deal contains a quote
    assert "bp_quote_trgt q where q.deal_id" in joined


def test_upsert_document_map_drops_stale_same_doc_entries():
    class _Cur:
        def __init__(self):
            self.sql = []
        def execute(self, sql, params=()):
            self.sql.append((" ".join(sql.split()), params))
    cur = _Cur()
    das._upsert_document_map(cur, "DEAL_X", "x", "po", "PO1", "DEAL_X::po::PO1", None)
    dels = [e for e in cur.sql if e[0].lower().startswith("delete from proc.bp_deal_document_map")]
    assert dels and dels[0][1] == ("po", "PO1", "DEAL_X::po::PO1")


def test_prune_deal_document_map_checks_all_trgt_tables():
    class _Cur:
        def __init__(self):
            self.sql = []
            self.rowcount = 3
        def execute(self, sql, params=()):
            self.sql.append(" ".join(sql.split()))
    cur = _Cur()
    assert das._prune_deal_document_map(cur) == 3
    s = cur.sql[0].lower()
    assert "delete from proc.bp_deal_document_map" in s
    for t in ("bp_invoice_trgt", "bp_quote_trgt", "bp_purchase_order_trgt"):
        assert t in s


def test_mirror_deal_to_raw_and_stg_targets_every_tier():
    deal_cols = ["deal_id", "deal_name", "deal_date"]
    cols = {
        "bp_invoice_raw": ["invoice_id", "raw_id"] + deal_cols,
        "bp_invoice_stg": ["invoice_id"] + deal_cols,
        "bp_invoice_trgt": ["invoice_id"] + deal_cols,
        "bp_invoice_line_items_raw": ["raw_id", "deal_id", "deal_name"],
        "bp_quote_raw": ["quote_id", "raw_id"] + deal_cols,
        "bp_quote_stg": ["quote_id"] + deal_cols,
        "bp_quote_trgt": ["quote_id"] + deal_cols,
        "bp_quote_line_items_raw": ["raw_id", "deal_id", "deal_name"],
        "bp_purchase_order_raw": ["po_id", "raw_id"] + deal_cols,
        "bp_purchase_order_stg": ["po_id"] + deal_cols,
        "bp_purchase_order_trgt": ["po_id"] + deal_cols,
        "bp_po_line_items_raw": ["raw_id", "deal_id", "deal_name"],
    }
    cur = _RecCursor(cols)
    cur.rowcount = 1
    das._mirror_deal_to_raw_and_stg(cur)
    updates = [e[0].lower() for e in cur.executed if e[0].lower().startswith("update")]
    # _raw and _stg mirrored FROM _trgt for each of the 3 doc types (6 updates)…
    assert any("update proc.bp_invoice_raw" in u and "from proc.bp_invoice_trgt" in u for u in updates)
    assert any("update proc.bp_invoice_stg" in u and "from proc.bp_invoice_trgt" in u for u in updates)
    assert any("update proc.bp_purchase_order_raw" in u and "from proc.bp_purchase_order_trgt" in u for u in updates)
    # …plus line-item _raw inheriting the deal from the parent doc _raw via raw_id.
    assert any("update proc.bp_po_line_items_raw" in u and "l.raw_id=p.raw_id" in u for u in updates)
    # idempotency guard: every mirror is gated on a value actually differing.
    assert all("is distinct from" in u for u in updates)


def test_propagate_deal_date_stamps_every_doc_in_deal():
    import datetime as dt

    class _Cur:
        def __init__(self):
            self.sql = []
            self.rowcount = 1
            self._rows = []
            self.description = None
        def execute(self, sql, params=()):
            self.sql.append((" ".join(sql.split()), params))
            if "max(expected_delivery_date)" in sql:
                self._rows = [("DEAL_X", dt.date(2024, 1, 1))]
                self.description = [("deal_id",), ("dd",)]
            else:
                self._rows = []
        def fetchall(self):
            return list(self._rows)
    cur = _Cur()
    das._propagate_deal_date(cur)
    updates = [e for e in cur.sql if e[0].lower().startswith("update")]
    assert len(updates) == 3   # invoice, quote, po all stamped with the deal date
    assert all("deal_date=%s" in e[0].lower() and "DEAL_X" in str(e[1]) for e in updates)


def test_assign_deals_runs_all_passes_and_returns_counts(monkeypatch):
    calls = []
    monkeypatch.setattr(das, "_look_forward", lambda cur: calls.append("fwd") or 2)
    monkeypatch.setattr(das, "_look_back", lambda cur: calls.append("back") or 1)
    monkeypatch.setattr(das, "_reconcile_legacy", lambda cur: calls.append("rec") or 3)
    monkeypatch.setattr(das, "_propagate_deal_along_po", lambda cur: calls.append("prop") or 6)
    monkeypatch.setattr(das, "_flag_conflict_po_chains", lambda cur: calls.append("conf") or 7)
    monkeypatch.setattr(das, "_backfill_deal_metadata", lambda cur: calls.append("meta") or 5)
    monkeypatch.setattr(das, "_propagate_deal_date", lambda cur: calls.append("dates") or 8)
    monkeypatch.setattr(das, "_mirror_deal_to_raw_and_stg", lambda cur: calls.append("mirror") or 10)
    monkeypatch.setattr(das, "_prune_deal_document_map", lambda cur: calls.append("prune") or 9)
    monkeypatch.setattr(das, "reconcile_status", lambda cur: calls.append("status") or 4)
    cur = _ScriptCursor(script=[], columns={})
    conn = _RecConn(cur)
    result = das.assign_deals(conn=conn)
    assert result == {"forward_linked": 2, "backward_linked": 1, "reconciled": 3,
                      "propagated": 6, "conflicts_flagged": 7, "metadata_filled": 5,
                      "deal_dates_set": 8, "tiers_mirrored": 10, "map_pruned": 9,
                      "status_reconciled": 4}
    assert calls == ["fwd", "back", "rec", "prop", "conf", "meta", "dates",
                     "mirror", "prune", "status"]
