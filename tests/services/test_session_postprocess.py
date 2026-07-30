"""Session-completion housekeeping: stale-PO reconcile + proposal generation.

After ses-20260730-UJF3, two gaps remained even with the batch-label gate
fixed: (1) po_not_found rows stayed open forever after the cited PO finally
promoted, and (2) nothing ever ran clustering, so the '3 deals in one batch'
proposal the design promises never appeared. Both now run when a session's
fast-promotion pass completes.
"""
import src.services.session_postprocess as spp


class _Cur:
    def __init__(self, labels, established=frozenset(), reconciled=3):
        self._labels = labels
        self._established = established
        self._reconciled = reconciled
        self.rowcount = 0
        self._rows = []
        self.executed = []

    def execute(self, sql, params=()):
        s = " ".join(sql.lower().split())
        self.executed.append(s)
        if "update proc.bp_extraction_discrepancy" in s:
            self.rowcount = self._reconciled
            self._rows = []
        elif "from proc.process_monitor" in s and "session_id" in s:
            self._rows = [(l,) for l in self._labels]
        elif "from proc.bp_deal " in s:
            hit = params[0] in self._established
            self._rows = [(params[0],)] if hit else []
        elif "bp_deal_proposal" in s:
            self._rows = []
        else:
            self._rows = []

    def fetchall(self): return self._rows
    def fetchone(self): return self._rows[0] if self._rows else None


class _Conn:
    def __init__(self, cur): self._cur = cur; self.commits = 0
    def cursor(self): return self._cur
    def commit(self): self.commits += 1
    def rollback(self): pass


def _patch_conn(monkeypatch, cur):
    from contextlib import contextmanager

    @contextmanager
    def fake_conn():
        yield _Conn(cur)

    monkeypatch.setattr(spp, "get_conn", fake_conn)


def test_reconcile_resolves_open_po_rows_whose_po_now_exists():
    cur = _Cur(labels=[], reconciled=5)
    assert spp.reconcile_po_discrepancies(cur) == 5
    sql = cur.executed[0]
    assert "po_not_found" in sql and "po_pending_review" in sql
    assert "resolved" in sql
    # resolution_action is constrained to apply_value/keep_null/dismiss
    assert "'dismiss'" in sql


def test_postprocess_generates_proposals_for_unestablished_labels(monkeypatch):
    cur = _Cur(labels=["TESTDATA_3007262026073025"])
    _patch_conn(monkeypatch, cur)
    calls = []
    monkeypatch.setattr(
        spp, "_generate_proposals",
        lambda label, session_id: calls.append((label, session_id))
        or {"proposal_ids": [1, 2, 3], "ungrouped": []})
    out = spp.postprocess_session("ses-20260730-UJF3")
    assert calls == [("TESTDATA_3007262026073025", "ses-20260730-UJF3")]
    assert out["proposals"]["TESTDATA_3007262026073025"]["proposal_ids"] == [1, 2, 3]
    assert out["po_discrepancies_resolved"] == 3


def test_postprocess_skips_established_deals(monkeypatch):
    # Amend-mode: the user explicitly targeted a real, tracked deal —
    # clustering must not second-guess that grouping (spec §Upload path change).
    cur = _Cur(labels=["DEALV2-PO99"], established={"DEALV2-PO99"})
    _patch_conn(monkeypatch, cur)
    monkeypatch.setattr(
        spp, "_generate_proposals",
        lambda *a: (_ for _ in ()).throw(AssertionError("must not run")))
    out = spp.postprocess_session("ses-x")
    assert out["proposals"] == {}


def test_postprocess_generation_failure_is_isolated(monkeypatch):
    cur = _Cur(labels=["BATCH_A"])
    _patch_conn(monkeypatch, cur)
    monkeypatch.setattr(
        spp, "_generate_proposals",
        lambda *a: (_ for _ in ()).throw(RuntimeError("clustering blew up")))
    out = spp.postprocess_session("ses-x")
    assert out["proposals"]["BATCH_A"] == {"error": "clustering blew up"}
