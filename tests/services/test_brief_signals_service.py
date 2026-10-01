"""Unit tests for Today's brief signals. Pure functions on dicts — no DB.

Row shapes mirror proc.bp_contract_master, proc.bp_invoice_trgt,
proc.bp_requirement, proc.bp_value_outcome and proc.bp_opportunity.

The rule every one of these tests is really asserting: a signal the corpus
cannot ground is ABSENT, never a zero. The brief's readings drop a missing
signal and keep going, so a short brief is right and an invented one is not.
"""
from datetime import date, datetime, timedelta, timezone

from src.services import brief_signals_service as bss
from src.services.brief_signals_service import (
    expiring_signal,
    identities_for,
    my_requests_signal,
    period_label,
    pick_trend_periods,
    savings_signal,
    spend_signal,
)

TODAY = date(2026, 10, 1)
NOW = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)


# ── period naming and selection ──────────────────────────────────────────────

def test_period_label_names_the_quarter():
    assert period_label(date(2026, 4, 1)) == "Q2 2026"
    assert period_label(date(2026, 1, 31)) == "Q1 2026"
    assert period_label(date(2025, 12, 31)) == "Q4 2025"


def test_pick_trend_periods_uses_the_latest_quarter_that_has_data():
    """The corpus stops at 2026-07-18 while today is in Q4. Comparing the
    current calendar quarter (empty) against the last one would print "down
    100%" — true of the rows, false as a statement. The latest POPULATED
    quarter is the one to report, and it must carry its own name."""
    buckets = [
        {"period": date(2025, 10, 1), "amount": 96_095_862.66, "n": 878},
        {"period": date(2026, 1, 1), "amount": 53_921_585.69, "n": 920},
        {"period": date(2026, 4, 1), "amount": 35_195_187.06, "n": 640},
    ]
    picked = pick_trend_periods(buckets)
    assert picked is not None
    current, prior = picked
    assert current["period"] == date(2026, 4, 1)
    assert prior["period"] == date(2026, 1, 1)


def test_pick_trend_periods_skips_a_thin_tail_quarter():
    """Q3 2026 holds nine invoices against the prior quarter's 640. Reporting
    it as "spend collapsed 99%" would be a statement about when the corpus was
    loaded, not about spend. A quarter under the floor is not a period."""
    buckets = [
        {"period": date(2026, 1, 1), "amount": 53_921_585.69, "n": 920},
        {"period": date(2026, 4, 1), "amount": 35_195_187.06, "n": 640},
        {"period": date(2026, 7, 1), "amount": 122_826.20, "n": 9},
    ]
    current, prior = pick_trend_periods(buckets)
    assert current["period"] == date(2026, 4, 1)
    assert prior["period"] == date(2026, 1, 1)


def test_pick_trend_periods_returns_none_without_two_periods():
    assert pick_trend_periods([]) is None
    assert pick_trend_periods([{"period": date(2026, 4, 1), "amount": 10.0, "n": 500}]) is None


# ── spend ────────────────────────────────────────────────────────────────────

def _bucket(period, amount, n=600, currency="GBP"):
    return {"period": period, "currency": currency, "amount": amount, "n": n}


def test_spend_signal_converts_every_currency_and_names_both_periods():
    rates = {"GBP": 0.8, "USD": 1.0, "_fetched_at": "2026-10-01T00:00:00+00:00"}
    rows = [
        _bucket(date(2026, 1, 1), 1_000_000.0, 500, "GBP"),
        _bucket(date(2026, 1, 1), 1_250_000.0, 400, "USD"),   # -> 1,000,000 GBP
        _bucket(date(2026, 4, 1), 1_200_000.0, 500, "GBP"),
        _bucket(date(2026, 4, 1), 1_000_000.0, 400, "USD"),   # ->   800,000 GBP
    ]
    sig = spend_signal(rows, rates)
    assert sig["prior"] == 2_000_000.0
    assert sig["current"] == 2_000_000.0
    assert sig["period"] == "Q1 2026"          # the UI prints this as the comparison
    assert sig["currentPeriod"] == "Q2 2026"   # ...and this as the row it belongs to


def test_spend_signal_reports_what_it_could_not_convert():
    """An unconvertible leg is excluded from the sum and SAID, never folded in
    at 1:1 — summing £ and ₹ invents a number (the gateway refuses the same)."""
    rates = {"GBP": 0.8, "USD": 1.0}
    rows = [
        _bucket(date(2026, 1, 1), 1_000_000.0, 500, "GBP"),
        _bucket(date(2026, 4, 1), 1_200_000.0, 500, "GBP"),
        _bucket(date(2026, 4, 1), 900_000.0, 300, "INR"),     # no rate
    ]
    sig = spend_signal(rows, rates)
    assert sig["current"] == 1_200_000.0
    assert "INR" in sig["excludedCurrencies"]


def test_spend_signal_absent_without_rates():
    """No rate set means no honest cross-currency total. Absent, not zero."""
    rows = [_bucket(date(2026, 1, 1), 1.0), _bucket(date(2026, 4, 1), 2.0)]
    assert spend_signal(rows, None) is None


def test_spend_signal_absent_on_a_single_period():
    rates = {"GBP": 0.8}
    assert spend_signal([_bucket(date(2026, 4, 1), 1.0)], rates) is None


# ── contracts expiring with nothing decided ──────────────────────────────────

def _contract(**kw):
    row = {
        "contract_id": "C-1",
        "contract_title": "MSA",
        "supplier_name": "Acme Industrial",
        "contract_end_date": date(2026, 10, 12),
        "total_contract_value": 400_000.0,
        "currency": "GBP",
        "auto_renew_flag": "Yes",
    }
    row.update(kw)
    return row


def test_expiring_signal_counts_days_values_and_the_nearest():
    rates = {"GBP": 0.8, "USD": 1.0}
    rows = [
        _contract(),
        _contract(contract_id="C-2", supplier_name="Northvale Logistics",
                  contract_title="freight", contract_end_date=date(2026, 10, 26),
                  total_contract_value=840_000.0),
    ]
    sig = expiring_signal(rows, TODAY, 60, rates)
    assert sig["count"] == 2
    assert sig["withinDays"] == 60
    assert sig["valueGbp"] == 1_240_000.0
    assert sig["nearest"] == {"name": "Acme Industrial", "days": 11}
    assert sig["items"][0]["name"] == "Acme Industrial · MSA"
    assert sig["items"][0]["days"] == 11
    assert sig["items"][0]["due"] == "12 Oct"


def test_expiring_signal_orders_by_how_soon():
    rates = {"GBP": 0.8}
    rows = [
        _contract(contract_id="C-2", contract_end_date=date(2026, 11, 20)),
        _contract(contract_id="C-1", contract_end_date=date(2026, 10, 5)),
    ]
    sig = expiring_signal(rows, TODAY, 60, rates)
    assert [i["days"] for i in sig["items"]] == [4, 50]


def test_expiring_signal_keeps_the_count_when_a_value_is_unconvertible():
    """A contract whose value cannot be stated in GBP still EXPIRES. Dropping
    it from the count to keep the arithmetic tidy would hide the exposure."""
    rates = {"GBP": 0.8}
    rows = [_contract(), _contract(contract_id="C-2", currency="INR",
                                   total_contract_value=9_000_000.0)]
    sig = expiring_signal(rows, TODAY, 60, rates)
    assert sig["count"] == 2
    assert sig["valueGbp"] == 400_000.0
    assert sig["valuePartial"] is True


def test_expiring_signal_drops_a_value_it_cannot_total_at_all():
    rates = {"GBP": 0.8}
    rows = [_contract(currency="INR", total_contract_value=9_000_000.0)]
    sig = expiring_signal(rows, TODAY, 60, rates)
    assert sig["count"] == 1
    assert sig["valueGbp"] is None


def test_expiring_signal_samples_the_items_but_counts_them_all():
    """43 contracts expire inside 60 days on the live corpus. A card listing all
    43 is the report the brief exists to replace, so the items are a sample and
    the count is not."""
    rates = {"GBP": 0.8}
    rows = [_contract(contract_id=f"C-{i}",
                      contract_end_date=date(2026, 10, 2) + timedelta(days=i))
            for i in range(43)]
    sig = expiring_signal(rows, TODAY, 60, rates)
    assert sig["count"] == 43
    assert len(sig["items"]) == bss.MAX_ITEMS
    assert sig["items"][0]["days"] == 1          # the soonest, not an arbitrary six


def test_expiring_signal_absent_when_nothing_expires():
    assert expiring_signal([], TODAY, 60, {"GBP": 0.8}) is None


def test_expiring_signal_names_the_supplier_when_there_is_no_title():
    sig = expiring_signal([_contract(contract_title=None)], TODAY, 60, {"GBP": 0.8})
    assert sig["items"][0]["name"] == "Acme Industrial"


# ── the caller's own requests ────────────────────────────────────────────────

def _req(**kw):
    row = {
        "requirement_id": "R-1",
        "title": "Laptop refresh, Q4",
        "status": "gathering",
        "created_by": "nick",
        "created_at": NOW - timedelta(days=9),
        "updated_at": NOW - timedelta(days=1),
    }
    row.update(kw)
    return row


def test_my_requests_signal_counts_what_is_still_moving():
    rows = [
        _req(),
        _req(requirement_id="R-2", title="Meeting room AV", status="complete",
             created_at=NOW - timedelta(days=4)),
    ]
    sig = my_requests_signal(rows, NOW)
    assert sig["open"] == 2
    assert sig["oldestDays"] == 9
    assert sig["latest"] == {"name": "Meeting room AV", "stage": "ready for sourcing"}
    assert {"name": "Laptop refresh, Q4", "stage": "being specified"} in sig["items"]


def test_my_requests_signal_reports_what_was_handed_on_this_week():
    rows = [
        _req(),
        _req(requirement_id="R-3", title="Office fit-out", status="handed_off",
             updated_at=NOW - timedelta(days=2)),
        _req(requirement_id="R-4", title="Print consolidation", status="handed_off",
             updated_at=NOW - timedelta(days=6)),
    ]
    sig = my_requests_signal(rows, NOW)
    assert sig["open"] == 1
    assert sig["approvedRecently"] == 2
    assert sig["approvedNames"] == ["Office fit-out", "Print consolidation"]


def test_my_requests_signal_ignores_an_old_hand_off():
    rows = [_req(), _req(requirement_id="R-5", status="handed_off",
                         updated_at=NOW - timedelta(days=40))]
    sig = my_requests_signal(rows, NOW)
    assert sig.get("approvedRecently") in (None, 0)


def test_my_requests_signal_absent_when_the_caller_raised_nothing():
    assert my_requests_signal([], NOW) is None


def test_identities_match_a_username_an_email_or_its_local_part():
    """created_by holds 'nick', 'p.keerthana' and 'p.keerthana@dhsit.co.uk' in
    the same column. One principal has to match all three spellings of itself,
    or a person's own requests silently belong to nobody."""
    ids = identities_for(username="p.keerthana", email="P.Keerthana@dhsit.co.uk",
                         subject="b265e4a4-40c1")
    assert "p.keerthana" in ids
    assert "p.keerthana@dhsit.co.uk" in ids
    assert "b265e4a4-40c1" in ids


def test_identities_is_empty_without_a_principal():
    """Auth off must not hand one person another person's requests."""
    assert identities_for(username=None, email=None, subject=None) == []


# ── savings: what actually moved ─────────────────────────────────────────────

def test_savings_signal_reports_realised_and_identified_in_the_window():
    sig = savings_signal(
        realised={"current": 226.78, "prior": 0.0},
        identified={"current": 3_233_077.21, "prior": 44_154.80, "count": 300},
        open_total=3_538_813.49,
        window_days=30,
    )
    assert sig["realised"] == 226.78
    assert sig["realisedPrior"] == 0.0
    assert sig["identified"] == 3_233_077.21
    assert sig["identifiedCount"] == 300
    assert sig["identifiedOpen"] == 3_538_813.49
    assert sig["windowDays"] == 30


def test_savings_signal_survives_a_period_with_nothing_realised():
    """Realised savings in this corpus is a measured zero. The brief still has
    something true to say — what was identified — so the signal is PRESENT
    with realised at zero, rather than withheld. A zero that was counted is
    not the same as a figure that is missing."""
    sig = savings_signal(
        realised={"current": 0.0, "prior": 0.0},
        identified={"current": 3_233_077.21, "prior": 44_154.80, "count": 300},
        open_total=3_538_813.49,
        window_days=30,
    )
    assert sig is not None
    assert sig["realised"] == 0.0
    assert sig["identified"] == 3_233_077.21


def test_savings_signal_absent_when_nothing_moved_at_all():
    """Nothing realised and nothing identified is not news; it is an empty
    corpus. The reading drops rather than printing four zeros."""
    assert savings_signal(
        realised={"current": 0.0, "prior": 0.0},
        identified={"current": 0.0, "prior": 0.0, "count": 0},
        open_total=0.0,
        window_days=30,
    ) is None


# ── the assembled payload ────────────────────────────────────────────────────

def test_missed_opportunity_is_never_returned():
    """There is no lapse state: every opportunity in the corpus sits at
    'identified' and none has been retired. Until one can be MEASURED, the
    brief says nothing about missed value — the mock said £86,000."""
    assert "missedOpportunity" not in bss.SIGNAL_KEYS


def test_sources_never_name_an_internal_table_or_route():
    """OutputSafetyMiddleware replaces any field naming an internal route or
    table with "[withheld]", which would turn the diagnostics into noise."""
    for reason in bss.SOURCE_STATES:
        assert "proc." not in reason
        assert "/" not in reason
