from src.services import declared_linkage as dl

class _Cur:
    def __init__(self, confirmed_members):
        self._confirmed = confirmed_members  # list of (proposal_id, doc_type, doc_pk)
        self.description = None; self._rows = []
    def execute(self, sql, params=()):
        s = sql.lower()
        if "bp_deal_proposal_member" in s and "confirmed" in s:
            self.description = [("proposal_id",), ("doc_type",), ("doc_pk",)]
            self._rows = list(self._confirmed)
        elif "explicit_group" in s or "process_monitor" in s:
            self.description = [("a",), ("b",)]; self._rows = []
        else:
            self.description = None; self._rows = []
    def fetchall(self): return self._rows

def test_confirmed_proposal_members_are_declared():
    cur = _Cur([(1, "quote", "Q1"), (1, "quote", "Q2"), (1, "po", "PO1")])
    groups = dl.declared_groups(cur, "BATCH1")
    flat = {pk for g in groups for (_dt, pk) in g}
    assert {"Q1", "Q2", "PO1"} <= flat

def test_no_confirmed_no_declared():
    assert dl.declared_groups(_Cur([]), "BATCH1") == []

def test_two_confirmed_proposals_stay_partitioned():
    # Two distinct confirmed proposals in the same batch must NOT merge into one
    # declared group -- each proposal is its own already-decided cluster.
    cur = _Cur([
        (1, "quote", "Q1"), (1, "quote", "Q2"),
        (2, "quote", "Q3"), (2, "po", "PO1"),
    ])
    groups = dl.declared_groups(cur, "BATCH1")
    assert len(groups) == 2

    group1 = frozenset({("quote", "Q1"), ("quote", "Q2")})
    group2 = frozenset({("quote", "Q3"), ("po", "PO1")})
    assert group1 in groups
    assert group2 in groups
