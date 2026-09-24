"""The Action Centre row values the writer builds (no database)."""
from src.services.triage import writer
from src.services.triage.model import Severity
from tests.triage.helpers import deal, inv, line, pipeline, po


def test_line_arithmetic_expected_value_says_expected_not_none():
    ds = deal(po(), inv(lines=[line(1, qty="1", price="13531.49", amount="13631.49")]))
    (f,) = [f for f in pipeline(ds).findings if f.rule_id == "line_arithmetic"]
    v = writer._finding_values("RUN-1", f)
    assert v[10] == "expected: 13531.49"
    assert v[9] == "INV-1: 13631.49"


def test_missing_values_are_a_dash_not_none():
    ds = deal(po(), inv("INV-9", po_id=None))
    (f,) = [f for f in pipeline(ds).findings if f.rule_id == "no_po"]
    f.lead.severity = Severity.S2        # no_po is capped at S3; lift it to reach the writer
    v = writer._finding_values("RUN-1", f)
    assert v[9] == "INV-9: -" and v[10] == "expected: -"
