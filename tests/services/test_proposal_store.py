import json
import re
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
