import json
import re
import pytest
from src.services import proposal_store as ps

class _Cur:
    def __init__(self): self.exec=[]; self._id=100; self.description=None; self._one=None; self._rows=[]
    def execute(self, sql, params=()):
        self.exec.append((" ".join(sql.split()), params))
        s=sql.lower()
        if "insert into proc.bp_deal_proposal " in s and "returning proposal_id" in s:
            self._id+=1; self._one=(self._id,)
        elif "select" in s and "bp_deal_proposal" in s:
            self.description=[("proposal_id",),("proposed_name",),("confidence",),("status",)]
            self._rows=[(101,"Freight — 3 bidders",93.8,"proposed")]
    def fetchone(self): return self._one
    def fetchall(self): return self._rows

_RESULT = {"proposals": [
    {"proposed_name": "Freight — 3 bidders", "confidence": 93.8, "declared": False,
     "review_required": False, "review_reasons": [],
     "members": [
        {"doc_type": "quote", "doc_pk": "MFS-Q-3391 (V3 (BAFO))", "base_reference": "MFS-Q-3391",
         "role": "anchor_quote", "match_score": None, "match_evidence": None},
        {"doc_type": "quote", "doc_pk": "CL-2024-0771 (V3 (BAFO))", "base_reference": "CL-2024-0771",
         "role": "competing_quote", "match_score": 100.0,
         "match_evidence": {"signals": [{"id": "desc", "s": 1.0}]}}]}],
    "ungrouped": [], "members_with_lines": 2, "members_total": 2}

def test_store_inserts_proposal_and_members_with_json_evidence():
    cur=_Cur()
    ids = ps.store_proposals(cur, "BATCH1", "sess-1", _RESULT)
    assert ids == [101]
    joined = " || ".join(sql for sql,_ in cur.exec)
    assert "insert into proc.bp_deal_proposal" in joined.lower()
    assert "insert into proc.bp_deal_proposal_member" in joined.lower()
    # evidence serialised as JSON text somewhere in the member insert params
    assert any(isinstance(p, (list, tuple)) and any(isinstance(v, str) and "signals" in v for v in p)
               for _s, p in cur.exec)

def test_store_never_writes_deal_id():
    cur=_Cur(); ps.store_proposals(cur, "BATCH1", "s", _RESULT)
    for sql, params in cur.exec:
        if "insert into proc.bp_deal_proposal " in sql.lower():
            # standalone "deal_id" column, not the "batch_deal_id" column (which is required)
            assert not re.search(r"(?<!batch_)deal_id", sql.lower())   # deal_id minted only at confirm

def test_reject_sets_status_rejected():
    cur=_Cur(); ps.reject_proposal(cur, 101, "nick")
    assert any("status='rejected'" in sql.replace(" ","").lower() or
               "status = 'rejected'" in sql.lower() for sql,_ in cur.exec)


# --- FIX C1(a): regenerate replaces prior un-actioned proposals, never stacks -------

def test_delete_proposed_then_store_deletes_first_and_targets_only_proposed():
    cur = _Cur()
    ps.delete_proposed(cur, "BATCH1")
    ps.store_proposals(cur, "BATCH1", "sess-1", _RESULT)

    delete_calls = [(sql, p) for sql, p in cur.exec if sql.lower().startswith("delete")]
    assert len(delete_calls) == 1
    sql, params = delete_calls[0]
    # only 'proposed' rows are targeted -- confirmed/rejected/superseded are preserved
    assert "status='proposed'" in sql.replace(" ", "").lower()
    assert "batch_deal_id" in sql.lower()
    assert params == ("BATCH1",)

    # the DELETE precedes the proposal INSERT (regenerate replaces, doesn't stack)
    idx_delete = next(i for i, (sql, _) in enumerate(cur.exec) if sql.lower().startswith("delete"))
    idx_insert = next(i for i, (sql, _) in enumerate(cur.exec)
                       if sql.lower().startswith("insert into proc.bp_deal_proposal "))
    assert idx_delete < idx_insert


def test_delete_proposed_does_not_touch_confirmed_or_rejected_rows():
    cur = _Cur()
    ps.delete_proposed(cur, "BATCH1")
    sql, params = cur.exec[0]
    # the WHERE clause is a status filter, not a blanket delete of the batch
    assert "status" in sql.lower()
    assert "'confirmed'" not in sql.lower()
    assert "'rejected'" not in sql.lower()
    assert "'superseded'" not in sql.lower()


# --- FIX C1(b): declared (already-confirmed) proposals are never re-persisted ------

_RESULT_WITH_DECLARED = {"proposals": [
    {"proposed_name": "Declared grouping", "confidence": 100.0, "declared": True,
     "review_required": False, "review_reasons": [], "flags": [],
     "members": [{"doc_type": "quote", "doc_pk": "Q9-ALREADY-CONFIRMED", "base_reference": None,
                  "role": "competing_quote", "match_score": None, "match_evidence": None}]},
    {"proposed_name": "Freight — 3 bidders", "confidence": 93.8, "declared": False,
     "review_required": False, "review_reasons": [], "flags": [],
     "members": [{"doc_type": "quote", "doc_pk": "MFS-Q-3391", "base_reference": "MFS-Q-3391",
                  "role": "anchor_quote", "match_score": None, "match_evidence": None}]},
], "ungrouped": [], "members_with_lines": 2, "members_total": 2}


def test_store_skips_declared_proposals_only_persists_inferred():
    cur = _Cur()
    ids = ps.store_proposals(cur, "BATCH1", "s", _RESULT_WITH_DECLARED)
    assert len(ids) == 1   # only the inferred (declared=False) proposal was persisted
    joined = " || ".join(str(p) for _sql, p in cur.exec)
    assert "Q9-ALREADY-CONFIRMED" not in joined
    assert "MFS-Q-3391" in joined


# --- FIX I1: proposal-level HITL flags are persisted and returned ------------------

_RESULT_WITH_FLAGS = {"proposals": [
    {"proposed_name": "Freight — 3 bidders", "confidence": 60.0, "declared": False,
     "review_required": False, "review_reasons": [],
     "flags": ["unresolved supplier on SDP-Q-44120"],
     "members": [{"doc_type": "quote", "doc_pk": "SDP-Q-44120", "base_reference": None,
                  "role": "anchor_quote", "match_score": None, "match_evidence": None}]},
], "ungrouped": [], "members_with_lines": 1, "members_total": 1}


def test_store_persists_flags_json_on_proposal_insert():
    cur = _Cur()
    ps.store_proposals(cur, "BATCH1", "s", _RESULT_WITH_FLAGS)
    insert_calls = [(sql, p) for sql, p in cur.exec
                    if sql.lower().startswith("insert into proc.bp_deal_proposal ")]
    assert len(insert_calls) == 1
    sql, params = insert_calls[0]
    assert "flags" in sql.lower()
    assert any(isinstance(v, str) and "unresolved supplier on SDP-Q-44120" in v for v in params)


class _ListCur:
    """Fake cursor for list_proposals: distinguishes the header select from the
    per-proposal member select by table name."""
    def __init__(self):
        self.description = None
        self._rows = []
    def execute(self, sql, params=()):
        s = sql.lower()
        if "bp_deal_proposal_member" in s:
            self.description = [("doc_type",), ("doc_pk",), ("base_reference",), ("role",),
                                ("match_score",), ("match_evidence",), ("review_required",),
                                ("review_reasons",)]
            self._rows = []
        else:
            self.description = [("proposal_id",), ("batch_deal_id",), ("proposed_name",),
                                ("confidence",), ("status",), ("deal_id",), ("flags",),
                                ("created_at",), ("confirmed_at",), ("confirmed_by",)]
            self._rows = [(101, "BATCH1", "Freight — 3 bidders", 60.0, "proposed", None,
                          '["unresolved supplier on SDP-Q-44120"]', None, None, None)]
    def fetchall(self):
        return self._rows


def test_list_proposals_returns_flags_column():
    cur = _ListCur()
    out = ps.list_proposals(cur, "BATCH1")
    assert len(out) == 1
    assert out[0]["flags"] == '["unresolved supplier on SDP-Q-44120"]'


# --- FIX I2: rejected_member_sets helper (quote doc_pks only, grouped per proposal) -

class _RejectedSetsCur:
    def __init__(self, rows):
        self._rows = rows
        self.description = None
        self.exec = []
    def execute(self, sql, params=()):
        self.exec.append((" ".join(sql.split()), params))
        self.description = [("proposal_id",), ("doc_pk",)]
    def fetchall(self):
        return self._rows


def test_rejected_member_sets_groups_quote_pks_by_proposal():
    cur = _RejectedSetsCur([(5, "Q1"), (5, "Q2")])
    sets = ps.rejected_member_sets(cur, "BATCH1")
    assert sets == [frozenset({"Q1", "Q2"})]
    sql, params = cur.exec[0]
    assert "status = 'rejected'" in sql.lower()
    assert "doc_type = 'quote'" in sql.lower()
    assert params == ("BATCH1",)


def test_rejected_member_sets_empty_when_no_rejections():
    cur = _RejectedSetsCur([])
    assert ps.rejected_member_sets(cur, "BATCH1") == []


class _FakeConn:
    """Minimal stand-in for a psycopg2 connection: record_action() only needs
    .autocommit (so it skips the SAVEPOINT dance) and .cursor().execute()."""
    def __init__(self):
        self.autocommit = True
        self.execs = []
    def cursor(self):
        outer = self
        class _AuditCur:
            def execute(self, sql, params=()):
                outer.execs.append((" ".join(sql.split()), params))
        return _AuditCur()


class _ConfirmCur(_Cur):
    def __init__(self, members):
        super().__init__(); self._members=members
        self.connection = _FakeConn()   # record_action(conn=cur.connection) needs this
    def execute(self, sql, params=()):
        s=sql.lower()
        self.exec.append((" ".join(sql.split()), params))
        if "from proc.bp_deal_proposal_member" in s and "select" in s:
            self.description=[("doc_type",),("doc_pk",),("base_reference",),("role",)]
            self._rows=self._members
        elif "from proc.bp_deal_proposal" in s and "select" in s:
            self.description=[("proposal_id",),("proposed_name",),("status",)]
            self._rows=[(101,"Freight — 3 bidders","proposed")]
        elif "information_schema.columns" in s:
            self.description=[("column_name",)]; self._rows=[("deal_id",),("deal_name",),("document_id",),("deal_date",)]
    def fetchall(self): return self._rows

def test_confirm_mints_deal_and_marks_confirmed():
    members=[("quote","MFS-Q-3391 (V3 (BAFO))","MFS-Q-3391","anchor_quote"),
             ("quote","CL-2024-0771 (V3 (BAFO))","CL-2024-0771","competing_quote"),
             ("po","PO-2024-0091",None,"po")]
    cur=_ConfirmCur(members)
    out = ps.confirm_proposal(cur, 101, "nick")
    assert out["status"] == "confirmed"
    assert out["deal_id"].startswith("DEALV3-")
    joined=" || ".join(sql for sql,_ in cur.exec).lower()
    assert "insert into proc.bp_deal" in joined         # draft header row
    assert "update proc.bp_deal_proposal set status='confirmed'" in joined.replace("  "," ") or \
           "status = 'confirmed'" in joined
    # audit row landed on the fake connection's cursor, not the transaction cursor
    assert any("insert into proc.bp_agent_actions" in sql.lower() for sql, _ in cur.connection.execs)

def test_confirm_conflict_on_stale_members():
    members=[("quote","MFS-Q-3391 (V3 (BAFO))","MFS-Q-3391","anchor_quote")]
    cur=_ConfirmCur(members)
    with pytest.raises(ps.StaleProposalError):
        ps.confirm_proposal(cur, 101, "nick", expected_member_pks=["SOMETHING-ELSE"])
    # only the two read-back SELECTs happened — no insert/update, no minted deal, no audit row
    assert all(not sql.lower().startswith(("insert", "update")) for sql, _ in cur.exec)
    assert cur.connection.execs == []


# --- FIX C1(c): confirm is status-gated -- refuses a non-'proposed' proposal -------

class _NonProposedCur(_ConfirmCur):
    """Like _ConfirmCur, but the proposal header row reports an already-actioned
    status (rejected / confirmed / superseded)."""
    def __init__(self, members, status):
        super().__init__(members)
        self._status = status
    def execute(self, sql, params=()):
        s = sql.lower()
        self.exec.append((" ".join(sql.split()), params))
        if "from proc.bp_deal_proposal_member" in s and "select" in s:
            self.description = [("doc_type",), ("doc_pk",), ("base_reference",), ("role",)]
            self._rows = self._members
        elif "from proc.bp_deal_proposal" in s and "select" in s:
            self.description = [("proposal_id",), ("proposed_name",), ("status",)]
            self._rows = [(101, "Freight — 3 bidders", self._status)]
        elif "information_schema.columns" in s:
            self.description = [("column_name",)]
            self._rows = [("deal_id",), ("deal_name",), ("document_id",), ("deal_date",)]


def test_confirm_refuses_rejected_proposal_writes_nothing():
    members = [("quote", "MFS-Q-3391", "MFS-Q-3391", "anchor_quote")]
    cur = _NonProposedCur(members, "rejected")
    out = ps.confirm_proposal(cur, 101, "nick")
    assert out == {"status": "not_proposed", "current_status": "rejected", "proposal_id": 101}
    # no INSERT/UPDATE, no minted deal, no audit row -- confirm on a rejected proposal
    # must not double-confirm or resurrect it
    assert all(not sql.lower().startswith(("insert", "update")) for sql, _ in cur.exec)
    assert cur.connection.execs == []


def test_confirm_refuses_already_confirmed_proposal_writes_nothing():
    members = [("quote", "MFS-Q-3391", "MFS-Q-3391", "anchor_quote")]
    cur = _NonProposedCur(members, "confirmed")
    out = ps.confirm_proposal(cur, 101, "nick")
    assert out == {"status": "not_proposed", "current_status": "confirmed", "proposal_id": 101}
    assert all(not sql.lower().startswith(("insert", "update")) for sql, _ in cur.exec)
    assert cur.connection.execs == []
