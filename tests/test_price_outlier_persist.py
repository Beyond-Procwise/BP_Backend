from services.price_outlier.detector import (
    ISSUE_TYPE, Finding, persist_findings,
)
from services.price_outlier.rule import OutlierSettings, assess


class RecordingCursor:
    def __init__(self, existing=()):
        self.existing = set(existing)
        self.inserts = []
        self._last = None

    def execute(self, sql, params=None):
        if "SELECT" in sql.upper() and "bp_extraction_discrepancy" in sql:
            # A real cursor's fetchall() yields one flat row per existing key
            # (doc_type, doc_pk_candidate, field_name) -- not that key
            # re-wrapped in an extra tuple layer, which would never compare
            # equal to the (doc_type, doc_pk, field_name) dedupe key below.
            self._last = list(self.existing)
        else:
            self.inserts.append(params)
            self._last = []

    def fetchall(self):
        return self._last


def _finding():
    # 1.169 (not 1.17) so the ratio against 11.69 lands at exactly 10x --
    # the critical boundary -- while still formatting to "1.17" once
    # persist_findings rounds it for expected_value.
    verdict = assess(11.69, [1.169] * 18, OutlierSettings())
    return Finding(
        doc_type="invoice", doc_pk="INV1", line_number=3,
        field_name="line_items[3].unit_price",
        item_description="A4 Ruled Notebook", price=11.69,
        verdict=verdict, note="line 3: ...",
    )


def test_row_uses_expected_value_and_leaves_computed_value_null():
    cur = RecordingCursor()
    persist_findings(cur, [_finding()])
    (params,) = cur.inserts
    assert params[4] == ISSUE_TYPE
    assert float(params[3]) == 11.69          # raw_value: the observed price
    assert float(params[5]) == 1.17           # expected_value: the peer median
    assert params[6] is None                  # computed_value stays empty


def test_severity_and_status_follow_the_verdict():
    cur = RecordingCursor()
    persist_findings(cur, [_finding()])
    (params,) = cur.inserts
    assert params[7] == "critical"
    assert params[8] == "open"
    assert params[9] is False                 # blocks_promotion


def test_an_existing_open_finding_is_not_duplicated():
    cur = RecordingCursor(existing={("invoice", "INV1", "line_items[3].unit_price")})
    assert persist_findings(cur, [_finding()]) == 0
    assert cur.inserts == []
