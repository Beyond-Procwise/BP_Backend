"""The process_monitor deal carry must respect the established-deal gate.

promotion.promote() mirrors process_monitor.deal_id onto the staged row.
Before this fix it did so unconditionally, so an un-confirmed upload batch
label (e.g. TESTDATA_3007262026073025) leaked into _stg even though
_look_forward correctly refused to stamp it — one of the two paths that
collapsed the ses-20260730-UJF3 batch into a single deal.
"""
from src.services.extraction.promotion import _carry_pm_deal


class _Cur:
    def __init__(self, pm_deal, tracked=frozenset(), confirmed=frozenset()):
        self._pm = pm_deal            # (deal_id, deal_name) for pm row
        self._tracked = tracked       # deal_ids with bp_deal.is_tracked
        self._confirmed = confirmed   # deal_ids with a confirmed proposal
        self._rows = []

    def execute(self, sql, params=()):
        s = sql.lower()
        if "from proc.process_monitor" in s:
            self._rows = [self._pm] if self._pm else []
        elif "from proc.bp_deal " in s:
            hit = params[0] in self._tracked or "is_tracked" not in s
            self._rows = [(params[0],)] if (params[0] in self._tracked or
                                            (hit and params[0] in self._tracked)) else []
        elif "bp_deal_proposal" in s:
            self._rows = [(params[0],)] if params[0] in self._confirmed else []
        else:
            self._rows = []

    def fetchone(self): return self._rows[0] if self._rows else None
    def fetchall(self): return self._rows


def test_unestablished_batch_label_is_not_carried_to_stg():
    raw = {"process_monitor_id": 7}
    _carry_pm_deal(_Cur(("TESTDATA_3007262026073025", "Test Data_300726")), raw)
    assert "deal_id" not in raw


def test_tracked_deal_is_still_carried():
    raw = {"process_monitor_id": 7}
    _carry_pm_deal(
        _Cur(("DEALV2-PO123", "Acme — PO123"), tracked={"DEALV2-PO123"}), raw)
    assert raw["deal_id"] == "DEALV2-PO123"
    assert raw["deal_name"] == "Acme — PO123"


def test_confirmed_proposal_deal_is_still_carried():
    raw = {"process_monitor_id": 7}
    _carry_pm_deal(
        _Cur(("DEALV3-4", "Freight 2025"), confirmed={"DEALV3-4"}), raw)
    assert raw["deal_id"] == "DEALV3-4"


def test_existing_deal_id_never_overwritten():
    raw = {"process_monitor_id": 7, "deal_id": "DEALV2-KEEP"}
    _carry_pm_deal(_Cur(("DEALV3-4", "x"), confirmed={"DEALV3-4"}), raw)
    assert raw["deal_id"] == "DEALV2-KEEP"
