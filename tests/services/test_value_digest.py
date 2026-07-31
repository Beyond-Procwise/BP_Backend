"""The weekly value digest.

The rule that matters most here is the one about NOT sending. A digest that arrives every
week saying nothing is a digest people stop opening, and then the week it does matter they
miss it. So an empty week sends nothing at all.

Composition is pure (summary dict + now -> {subject, body} | None); the scheduler wrapper
is gated on two env vars and never raises.
"""
from datetime import datetime, timedelta, timezone

import pytest

from src.services import value_digest as vd

NOW = datetime(2026, 7, 31, 9, 0, tzinfo=timezone.utc)


def _iso(days_ago):
    return (NOW - timedelta(days=days_ago)).isoformat()


def _finding(**kw):
    row = {"id": "disc:80", "tier": "verified", "source": "discrepancy",
           "amount_gbp": 950.0, "recovered_gbp": None, "title": "Invoice INV-1042 bills over",
           "supplier_name": "Techworld", "age_days": 2, "found_at": _iso(2),
           "resolved_at": None, "status": "open", "superseded_by": None}
    row.update(kw)
    return row


def _summary(findings, **kw):
    row = {"verified_found_gbp": 950.0, "recovered_gbp": 0.0, "potential_gbp": 0.0,
           "finding_count": len(findings), "findings": findings, "by_supplier": [],
           "sources": {"discrepancies": "ok", "opportunities": "ok", "benchmark": "ok"}}
    row.update(kw)
    return row


# ---- when NOT to send ----------------------------------------------------

def test_empty_week_sends_nothing():
    # Findings exist, but none of them are from this week and nothing came back.
    old = _summary([_finding(found_at=_iso(30), age_days=30)])
    assert vd.compose_digest(old, NOW) is None


def test_a_week_with_nothing_at_all_sends_nothing():
    assert vd.compose_digest(_summary([]), NOW) is None
    assert vd.compose_digest(None, NOW) is None


def test_a_recovery_this_week_is_reason_enough_even_with_no_new_findings():
    # Nothing new found, but £950 came back — that is worth an email.
    recovered = _summary([_finding(found_at=_iso(40), age_days=40, status="resolved",
                                   recovered_gbp=950.0, resolved_at=_iso(2))],
                         recovered_gbp=950.0)
    digest = vd.compose_digest(recovered, NOW)
    assert digest is not None
    assert "950" in digest["subject"]


def test_a_superseded_finding_is_not_news():
    # It counts under its stronger source; reporting it as this week's find would double it.
    hidden = _summary([_finding(superseded_by="disc:1")])
    assert vd.compose_digest(hidden, NOW) is None


# ---- what it says --------------------------------------------------------

def test_digest_content():
    d = vd.compose_digest(_summary([_finding()]), NOW)
    assert "value found this week" in d["body"].lower()
    assert "£" in d["body"]
    assert "found " in d["body"]          # every listed finding carries its age
    assert "/spendiq" in d["body"]        # one way in, to the drawer


def test_it_lists_the_top_three_open_findings_by_amount():
    findings = [_finding(id=f"disc:{i}", amount_gbp=amount, title=f"Finding {i}")
                for i, amount in enumerate([100.0, 5000.0, 250.0, 9000.0], start=1)]
    d = vd.compose_digest(_summary(findings, verified_found_gbp=14350.0), NOW)
    body = d["body"]
    assert "Finding 4" in body and "Finding 2" in body and "Finding 3" in body
    assert "Finding 1" not in body                     # only the top three
    assert body.index("Finding 4") < body.index("Finding 2") < body.index("Finding 3")


def test_it_lists_only_findings_somebody_can_still_act_on():
    # A resolved finding is not a to-do. It counts in the totals; it is not in the list.
    findings = [_finding(id="disc:1", amount_gbp=9000.0, title="Already done",
                         status="resolved", resolved_at=_iso(1), recovered_gbp=9000.0),
                _finding(id="disc:2", amount_gbp=100.0, title="Still open")]
    body = vd.compose_digest(_summary(findings, recovered_gbp=9000.0), NOW)["body"]
    assert "Still open" in body and "Already done" not in body


def test_the_subject_carries_both_figures():
    # "Recovered this week" is the weekly slice, not the summary's all-time total — a
    # recovery from six months ago is not news this Friday.
    week = [_finding(),
            _finding(id="disc:81", amount_gbp=200.0, status="resolved",
                     recovered_gbp=200.0, resolved_at=_iso(1), found_at=_iso(1))]
    d = vd.compose_digest(_summary(week, recovered_gbp=200.0), NOW)
    assert d["subject"].startswith("Value found this week")
    assert "1,150.00" in d["subject"] and "200.00" in d["subject"]


def test_an_unconvertible_amount_is_counted_but_never_shown_as_zero():
    # amount_gbp None means we could not convert it. It is still a real finding, and
    # "£0.00 value found across 1 finding" is a contradiction that erodes trust in the
    # number — so the count is reported and no figure is invented.
    d = vd.compose_digest(_summary([_finding(amount_gbp=None)]), NOW)
    assert d is not None
    assert "1 finding found this week" in d["body"]
    assert "£0.00 value found" not in d["body"]
    assert d["subject"].startswith("Value found this week: 1 finding")


def test_a_mixed_week_totals_what_it_can_and_says_what_it_could_not():
    mixed = [_finding(id="disc:1", amount_gbp=950.0),
             _finding(id="disc:2", amount_gbp=None)]
    body = vd.compose_digest(_summary(mixed), NOW)["body"]
    assert "£950.00 value found this week across 2 findings." in body
    assert "A further 1 finding could not be converted" in body


def test_a_source_that_did_not_answer_is_declared():
    d = vd.compose_digest(
        _summary([_finding()], sources={"discrepancies": "ok", "opportunities": "unavailable",
                                        "benchmark": "ok"}), NOW)
    assert "unavailable source" in d["body"]


# ---- the scheduler wrapper ----------------------------------------------

def test_recipients_env_gate(monkeypatch):
    monkeypatch.setenv("VALUE_DIGEST_ENABLED", "1")
    monkeypatch.delenv("VALUE_DIGEST_RECIPIENTS", raising=False)
    sent = []
    monkeypatch.setattr(vd, "_send_email", lambda **kw: sent.append(kw) or True)
    assert vd.run_weekly_digest() == 0
    assert sent == []                      # skipped, nothing attempted


def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv("VALUE_DIGEST_ENABLED", raising=False)
    monkeypatch.setenv("VALUE_DIGEST_RECIPIENTS", "ap@example.com")
    called = []
    monkeypatch.setattr(vd, "_load_summary", lambda: called.append(1) or _summary([_finding()]))
    assert vd.run_weekly_digest() == 0
    assert called == []                    # not even a read while disabled


def test_sends_to_every_configured_recipient(monkeypatch):
    monkeypatch.setenv("VALUE_DIGEST_ENABLED", "1")
    monkeypatch.setenv("VALUE_DIGEST_RECIPIENTS", "ap@example.com, finance@example.com")
    monkeypatch.setattr(vd, "_load_summary", lambda: _summary([_finding()]))
    sent = {}
    monkeypatch.setattr(vd, "_send_email", lambda **kw: sent.update(kw) or True)
    assert vd.run_weekly_digest() == 1
    assert sent["to"] == ["ap@example.com", "finance@example.com"]
    assert "Value found this week" in sent["subject"]


def test_an_empty_week_is_not_sent_even_when_enabled(monkeypatch):
    monkeypatch.setenv("VALUE_DIGEST_ENABLED", "1")
    monkeypatch.setenv("VALUE_DIGEST_RECIPIENTS", "ap@example.com")
    monkeypatch.setattr(vd, "_load_summary",
                        lambda: _summary([_finding(found_at=_iso(40), age_days=40)]))
    sent = []
    monkeypatch.setattr(vd, "_send_email", lambda **kw: sent.append(kw) or True)
    assert vd.run_weekly_digest() == 0
    assert sent == []


def test_a_send_failure_never_escapes(monkeypatch):
    # The digest runs on the scheduler. A mail problem must not break the chain.
    monkeypatch.setenv("VALUE_DIGEST_ENABLED", "1")
    monkeypatch.setenv("VALUE_DIGEST_RECIPIENTS", "ap@example.com")
    monkeypatch.setattr(vd, "_load_summary", lambda: _summary([_finding()]))

    def _boom(**kw):
        raise RuntimeError("SES down")
    monkeypatch.setattr(vd, "_send_email", _boom)
    assert vd.run_weekly_digest() == 0


def test_a_dead_backend_never_escapes(monkeypatch):
    monkeypatch.setenv("VALUE_DIGEST_ENABLED", "1")
    monkeypatch.setenv("VALUE_DIGEST_RECIPIENTS", "ap@example.com")

    def _boom():
        raise RuntimeError("DB unreachable")
    monkeypatch.setattr(vd, "_load_summary", _boom)
    assert vd.run_weekly_digest() == 0
