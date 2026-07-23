from src.services import deal_assignment_service as das


class _Cur:
    def __init__(self, established, confirmed): self.e=established; self.c=confirmed; self.description=None; self._rows=[]
    def execute(self, sql, params=()):
        s=sql.lower()
        if "from proc.bp_deal " in s and "where deal_id" in s:
            self.description=[("deal_id",)]; self._rows=[(params[0],)] if params[0] in self.e else []
        elif "bp_deal_proposal" in s and "confirmed" in s:
            self.description=[("deal_id",)]; self._rows=[(params[0],)] if params[0] in self.c else []
        else:
            self.description=None; self._rows=[]
    def fetchall(self): return self._rows
    def fetchone(self): return self._rows[0] if self._rows else None

def test_established_deal_true_when_in_bp_deal():
    assert das.is_established_deal(_Cur({"DEALV3-1"}, set()), "DEALV3-1") is True

def test_batch_label_is_not_established():
    assert das.is_established_deal(_Cur(set(), set()), "ANALYSISSET_19072620260719339") is False

def test_confirmed_proposal_deal_is_established():
    assert das.is_established_deal(_Cur(set(), {"DEALV3-9"}), "DEALV3-9") is True
