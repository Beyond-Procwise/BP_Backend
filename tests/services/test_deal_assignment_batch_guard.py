from src.services import deal_assignment_service as das


class _Cur:
    """Fake cursor: `deals` maps deal_id -> is_tracked; `confirmed` is the set
    of deal_ids with a confirmed bp_deal_proposal row."""

    def __init__(self, deals, confirmed):
        self.d = dict(deals); self.c = confirmed
        self.description = None; self._rows = []

    def execute(self, sql, params=()):
        s = sql.lower()
        if "from proc.bp_deal " in s and "where deal_id" in s:
            hit = params[0] in self.d
            if hit and "is_tracked" in s:
                hit = bool(self.d[params[0]])
            self.description = [("deal_id",)]
            self._rows = [(params[0],)] if hit else []
        elif "bp_deal_proposal" in s and "confirmed" in s:
            self.description = [("deal_id",)]
            self._rows = [(params[0],)] if params[0] in self.c else []
        else:
            self.description = None; self._rows = []

    def fetchall(self): return self._rows
    def fetchone(self): return self._rows[0] if self._rows else None


def test_established_deal_true_when_tracked_in_bp_deal():
    assert das.is_established_deal(_Cur({"DEALV3-1": True}, set()), "DEALV3-1") is True


def test_batch_label_is_not_established():
    assert das.is_established_deal(_Cur({}, set()), "ANALYSISSET_19072620260719339") is False


def test_confirmed_proposal_deal_is_established():
    assert das.is_established_deal(_Cur({"DEALV3-9": False}, {"DEALV3-9"}), "DEALV3-9") is True


def test_upload_draft_bp_deal_row_is_not_established():
    # The gateway inserts bp_deal (is_tracked=false) at the moment of upload
    # (data-integration.service.ts:253). That draft row must NOT make the
    # batch label an authoritative deal, or every named upload collapses
    # into a single deal (ses-20260730-UJF3).
    cur = _Cur({"TESTDATA_3007262026073025": False}, set())
    assert das.is_established_deal(cur, "TESTDATA_3007262026073025") is False
