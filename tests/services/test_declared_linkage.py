from src.services import declared_linkage as dl

class _Cur:
    def __init__(self, confirmed_members):
        self._confirmed = confirmed_members  # list of (doc_type, doc_pk)
        self.description = None; self._rows = []
    def execute(self, sql, params=()):
        s = sql.lower()
        if "bp_deal_proposal_member" in s and "confirmed" in s:
            self.description = [("doc_type",), ("doc_pk",)]
            self._rows = list(self._confirmed)
        elif "explicit_group" in s or "process_monitor" in s:
            self.description = [("a",), ("b",)]; self._rows = []
        else:
            self.description = None; self._rows = []
    def fetchall(self): return self._rows

def test_confirmed_proposal_members_are_declared():
    cur = _Cur([("quote", "Q1"), ("quote", "Q2"), ("po", "PO1")])
    groups = dl.declared_groups(cur, "BATCH1")
    flat = {pk for g in groups for (_dt, pk) in g}
    assert {"Q1", "Q2", "PO1"} <= flat

def test_no_confirmed_no_declared():
    assert dl.declared_groups(_Cur([]), "BATCH1") == []
