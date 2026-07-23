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
