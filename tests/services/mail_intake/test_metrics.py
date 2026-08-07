"""Counts derived from the tables, and the two Plan 1 alert conditions.

The quarantine-rate alert has to be robust to a quiet period: three quarantined
items out of three is a 100% rate and means nothing if three is the whole day's
traffic. A rate alert that fires on tiny samples trains people to ignore it.
"""
from src.services.mail_intake import metrics

from .conftest import FakeCursor


def test_collect_returns_the_spec_metric_names():
    cur = FakeCursor([(12,), (30,), [("macro", 2), ("hidden_text", 1)], (4,), (3,), (5,)])
    result = metrics.collect(cur)
    for key in (
        "bundles_received",
        "artifacts_ingested",
        "quarantine_by_reason",
        "duplicate_count",
        "sender_auth_failures",
        "unmatched_domains",
    ):
        assert key in result


def test_quarantine_by_reason_is_a_mapping():
    cur = FakeCursor([(12,), (30,), [("macro", 2), ("hidden_text", 1)], (4,), (3,), (5,)])
    result = metrics.collect(cur)
    assert result["quarantine_by_reason"]["macro"] == 2


def test_quarantine_rate_alert_fires_above_baseline():
    fired = []
    cur = FakeCursor([(100,), (40,)])
    alerts = metrics.check_alerts(
        cur, quarantine_baseline=0.15, alert=lambda code, payload: fired.append(code)
    )
    assert metrics.ALERT_QUARANTINE_RATE in alerts
    assert fired == [metrics.ALERT_QUARANTINE_RATE]


def test_quarantine_rate_alert_stays_quiet_below_baseline():
    cur = FakeCursor([(100,), (5,)])
    assert metrics.check_alerts(cur, alert=lambda *a: None) == []


def test_quarantine_rate_alert_ignores_tiny_samples():
    # 3 of 3 is 100%, and means nothing.
    cur = FakeCursor([(3,), (3,)])
    assert metrics.check_alerts(cur, alert=lambda *a: None) == []


def test_stale_binding_raises_the_adapter_failure_alert():
    cur = FakeCursor([(100,), (1,), [(1, 9000)]])
    alerts = metrics.check_alerts(cur, alert=lambda *a: None)
    assert metrics.ALERT_ADAPTER_FAILURE in alerts
