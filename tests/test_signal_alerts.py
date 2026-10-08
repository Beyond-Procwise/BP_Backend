"""Alerts on /health's output signals.

The signals exist because findings writes were silently rejected for 65 days
and the vector index sat empty for five weeks. A signal nobody watches is the
same silence, so this turns them into alerts: evaluated every five minutes by
the existing bp-extraction-health timer, mailed only when an alert starts or
clears, never on every run.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from src.services import signal_alerts as sa

NOW = datetime(2026, 10, 8, 22, 0, tzinfo=timezone.utc)


def _health(*, finished=NOW - timedelta(hours=2), failed=0, points=165):
    return {
        "status": "ok",
        "last_triage_run": (None if finished is None else
                            {"finished_at": finished.isoformat(), "failed_deals": failed}),
        "vector_store": {"collection": "procwise_document_embeddings", "points": points},
    }


# --- evaluate -----------------------------------------------------------------

def test_a_healthy_system_raises_nothing():
    assert sa.evaluate(_health(), NOW) == {}


def test_no_answer_from_the_api_is_an_alert():
    assert set(sa.evaluate(None, NOW)) == {"api_down"}


def test_triage_older_than_the_limit_is_stale():
    assert "triage_stale" in sa.evaluate(_health(finished=NOW - timedelta(hours=49)), NOW)
    assert "triage_stale" not in sa.evaluate(_health(finished=NOW - timedelta(hours=47)), NOW)


def test_the_stale_limit_is_configurable():
    h = _health(finished=NOW - timedelta(hours=7))
    assert "triage_stale" in sa.evaluate(h, NOW, triage_max_age=timedelta(hours=6))


def test_failed_deals_are_an_alert_naming_the_count():
    alerts = sa.evaluate(_health(failed=3), NOW)
    assert "3" in alerts["triage_failed_deals"]


def test_triage_that_has_never_finished_is_an_alert():
    assert "triage_never_ran" in sa.evaluate(_health(finished=None), NOW)


def test_an_empty_index_is_an_alert():
    assert "index_empty" in sa.evaluate(_health(points=0), NOW)


@pytest.mark.parametrize("field,key", [
    ("last_triage_run", "triage_unreadable"),
    ("vector_store", "index_unreadable"),
])
def test_a_signal_that_cannot_be_read_is_an_alert_not_a_pass(field, key):
    h = _health()
    h[field] = "unavailable" if field == "last_triage_run" else {"points": "unavailable"}
    assert key in sa.evaluate(h, NOW)


def test_a_health_body_missing_the_signals_is_an_alert_not_a_pass():
    """An older build without the fields must not read as healthy."""
    alerts = sa.evaluate({"status": "ok"}, NOW)
    assert {"triage_unreadable", "index_unreadable"} <= set(alerts)


# --- run: state and notification ---------------------------------------------

class _Mail:
    def __init__(self, ok=True):
        self.sent, self.ok = [], ok

    def __call__(self, subject, body):
        self.sent.append((subject, body))
        return self.ok


def _run(tmp_path, health, mail, now=NOW):
    return sa.run(fetch=lambda: health, send=mail, state_path=tmp_path / "state.json", now=now)


def test_a_new_alert_is_mailed_once_then_stays_quiet(tmp_path):
    mail = _Mail()
    _run(tmp_path, _health(points=0), mail)
    _run(tmp_path, _health(points=0), mail)

    assert len(mail.sent) == 1
    subject, body = mail.sent[0]
    assert "index_empty" in subject or "index" in subject.lower()
    assert "raised" in body.lower()


def test_a_cleared_alert_is_mailed(tmp_path):
    mail = _Mail()
    _run(tmp_path, _health(points=0), mail)
    _run(tmp_path, _health(), mail)

    assert len(mail.sent) == 2
    assert "cleared" in mail.sent[1][1].lower()


def test_nothing_is_mailed_while_healthy(tmp_path):
    mail = _Mail()
    _run(tmp_path, _health(), mail)
    assert mail.sent == []


def test_a_failed_send_is_retried_next_run(tmp_path):
    """State only advances once the mail went; otherwise the alert would be
    recorded as notified and never reach anyone."""
    failing, working = _Mail(ok=False), _Mail()
    _run(tmp_path, _health(points=0), failing)
    _run(tmp_path, _health(points=0), working)

    assert len(working.sent) == 1


def test_state_survives_between_runs_on_disk(tmp_path):
    _run(tmp_path, _health(points=0), _Mail())
    saved = json.loads((tmp_path / "state.json").read_text())
    assert "index_empty" in saved["active"]


def test_a_corrupt_state_file_is_treated_as_no_state(tmp_path):
    (tmp_path / "state.json").write_text("{not json")
    mail = _Mail()
    _run(tmp_path, _health(points=0), mail)
    assert len(mail.sent) == 1


def test_run_reports_what_it_did(tmp_path):
    out = _run(tmp_path, _health(points=0), _Mail())
    assert out["active"] == ["index_empty"]
    assert out["raised"] == ["index_empty"] and out["cleared"] == []
    assert out["notified"] is True
