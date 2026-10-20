"""The supplier QBR (quarterly business review) fact pack.

No database: the builder's queries and the rate batch are replaced by canned answers, keyed by the
statement they receive. What is pinned is what the pack SAYS: the same measures in the same order
for every supplier and period, the prior period taken as the equal window just before, a figure
that cannot be measured recorded as not assessed (never zero), and what the platform cannot yet
measure for a supplier stated plainly.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal

import pytest

from src.services.analytics.models import Confidence
from src.services.rga import factpack
from src.services.rga.builders import supplier_qbr as qbr
from src.services.rga.factpack import build_fact_pack
from src.services.rga.models import FindingCode

SCOPE = {"supplier_id": "S1", "period_start": "2026-07-01", "period_end": "2026-09-30",
         "period_label": "Q3 2026", "currency": "GBP"}
RATES = {"GBP": 0.75, "USD": 1.0, "EUR": 0.9}
BATCH = datetime(2026, 10, 20, tzinfo=timezone.utc)


class Fake:
    """Canned answers, and a record of what was asked."""

    def __init__(self, *, spend=None, prior=None, deals=3, findings=(5, 2), lines=(), rates=RATES,
                 receipts=0):
        self.spend, self.prior, self.deals, self.findings = spend, prior, deals, findings
        self.receipts = receipts
        self.lines, self.rates, self.asked = lines, rates, []

    def fetch(self, sql, params):
        self.asked.append((sql, params))
        if "bp_goods_receipt_trgt" in sql:
            return [(self.receipts,)]
        if "COUNT(*)::int" in sql and "bp_detection_finding" not in sql:
            return [(self.deals,)]
        if "bp_detection_finding" in sql:
            return [self.findings]
        if "bp_invoice_line_items_trgt" in sql:
            return list(self.lines)
        if "invoice_total" in sql:
            period = params[1:3]
            return list(self.prior if period == self.prior_window else self.spend or [])
        raise AssertionError(f"unexpected query: {sql[:80]}")

    prior_window = (date(2026, 3, 31), date(2026, 6, 30))

    def rates_fn(self):
        return self.rates, BATCH, not self.rates


@pytest.fixture
def run(monkeypatch):
    def _run(fake, scope=SCOPE):
        monkeypatch.setattr(qbr, "_fetch", fake.fetch)
        monkeypatch.setattr(qbr, "_rates", fake.rates_fn)
        return build_fact_pack("supplier_qbr", scope=scope, as_of="2026-10-20", emit_audit=False)
    return _run


def by(pack, derivation_prefix):
    return next(f for f in pack.facts if f.derivation.startswith(derivation_prefix))


def good():
    return Fake(spend=[(Decimal("1000"), "GBP"), (Decimal("900"), "USD")],
                prior=[(Decimal("1200"), "GBP")],
                lines=[(10.0, 11.0, 100.0, "GBP", "GBP"), (5.0, 5.0, 40.0, "GBP", "GBP")])


def test_the_type_is_registered_with_its_sections_and_title():
    assert "supplier_qbr" in factpack.registered_types()
    assert factpack.section_order_for("supplier_qbr") == ["summary", "spend", "price", "findings", "gaps"]
    assert factpack.title_for("supplier_qbr") == "Supplier business review"


def test_the_same_measures_come_in_the_same_order_every_time(run):
    labels = lambda p: [f.derivation.split(" @ ")[0] for f in p.facts]  # noqa: E731
    a = labels(run(good()))
    b = labels(run(Fake(spend=[], prior=[], lines=[], findings=(0, 0), deals=0)))
    assert a == b
    assert a[:3] == ["supplier_qbr.spend", "supplier_qbr.prior_spend", "supplier_qbr.spend_change"]


def test_spend_is_converted_one_currency_at_a_time(run):
    pack = run(good())
    spend = by(pack, "supplier_qbr.spend")
    assert spend.value == Decimal("1000") + Decimal("900") / Decimal("1.0") * Decimal("0.75")
    assert spend.currency == "GBP" and spend.confidence == Confidence.CORROBORATED


def test_the_prior_period_is_the_equal_window_just_before(run):
    fake = good()
    run(fake)
    windows = [params[1:3] for sql, params in fake.asked if "invoice_total" in sql]
    assert (date(2026, 7, 1), date(2026, 9, 30)) in windows
    assert (date(2026, 3, 31), date(2026, 6, 30)) in windows   # the same 92 days, ending the day before
    assert all(p[0] == "S1" for sql, p in fake.asked if "invoice_total" in sql)


def test_the_change_is_against_the_prior_period(run):
    pack = run(good())
    prior, change = by(pack, "supplier_qbr.prior_spend"), by(pack, "supplier_qbr.spend_change")
    assert prior.value == Decimal("1200")
    assert change.value == (Decimal("1675") - Decimal("1200")) / Decimal("1200") * 100


def test_no_spend_in_the_period_is_not_assessed_not_zero(run):
    pack = run(Fake(spend=[], prior=[(Decimal("1200"), "GBP")]))
    spend, change = by(pack, "supplier_qbr.spend"), by(pack, "supplier_qbr.spend_change")
    assert spend.value is None and spend.confidence == Confidence.UNASSESSED
    assert change.value is None


def test_no_prior_spend_means_no_change_figure_never_a_hundred_percent(run):
    pack = run(Fake(spend=[(Decimal("1000"), "GBP")], prior=[]))
    assert by(pack, "supplier_qbr.spend").value == Decimal("1000")
    assert by(pack, "supplier_qbr.prior_spend").value is None
    assert by(pack, "supplier_qbr.spend_change").value is None


def test_a_currency_with_no_rate_drops_every_money_figure_to_not_assessed(run):
    pack = run(Fake(spend=[(Decimal("1000"), "GBP"), (Decimal("500"), "NZD")],
                    prior=[(Decimal("1200"), "GBP")]))
    assert by(pack, "supplier_qbr.spend").confidence == Confidence.UNASSESSED
    assert any(f.code == FindingCode.PARTIAL_CURRENCY_CONVERSION for f in pack.findings)


def test_rates_unavailable_gives_no_money_figure(run):
    pack = run(Fake(spend=[(Decimal("1000"), "GBP")], prior=[(Decimal("1"), "GBP")], rates={}))
    assert by(pack, "supplier_qbr.spend").value is None


def test_counts_need_no_rates(run):
    pack = run(Fake(spend=[], prior=[], deals=7, findings=(5, 2), rates={}))
    assert by(pack, "supplier_qbr.deals").value == Decimal(7)
    assert by(pack, "supplier_qbr.open_findings").value == Decimal(5)
    assert by(pack, "supplier_qbr.open_overbilling_findings").value == Decimal(2)


def test_price_variance_is_the_ppv_formula_over_the_matched_lines(run):
    pack = run(good())
    total = by(pack, "supplier_qbr.price_variance")
    assert total.value == Decimal(str(100.0 - 0.0))          # +100 on one line, 0 on the other
    assert total.currency == "GBP"
    assert by(pack, "supplier_qbr.price_variance_pct").value == pytest.approx(Decimal("100") / Decimal("1200") * 100)
    assert by(pack, "supplier_qbr.lines_billed_off_po_price").value == Decimal(1)


def test_lines_compared_across_currencies_are_left_out_and_said_so(run):
    fake = good()
    fake.lines = list(fake.lines) + [(10.0, 99.0, 100.0, "USD", "GBP")]
    pack = run(fake)
    assert by(pack, "supplier_qbr.price_variance").value == Decimal("100.0")   # the odd line did not count
    assert any("different currencies" in f.detail for f in pack.findings)


def test_no_matched_lines_means_price_variance_is_not_assessed(run):
    pack = run(Fake(spend=[(Decimal("1"), "GBP")], prior=[], lines=[]))
    for d in ("price_variance", "price_variance_pct", "lines_billed_off_po_price"):
        assert by(pack, f"supplier_qbr.{d}").value is None


def test_what_the_platform_cannot_measure_is_stated_not_filled(run):
    pack = run(good())
    for d in ("on_time_in_full", "contract_coverage"):
        fact = by(pack, f"supplier_qbr.{d}")
        assert fact.value is None and fact.confidence == Confidence.UNASSESSED
    reasons = " ".join(f.detail for f in pack.findings if f.code == FindingCode.MEASURE_UNAVAILABLE)
    assert "delivery" in reasons.lower() and "contract" in reasons.lower()


def test_if_receipts_now_exist_the_pack_does_not_claim_there_are_none(run):
    fake = good()
    fake.receipts = 12
    pack = run(fake)
    fact = by(pack, "supplier_qbr.on_time_in_full")
    assert fact.value is None
    detail = " ".join(f.detail for f in pack.findings if f.fact_id == fact.fact_id)
    assert "no goods receipts" not in detail.lower()
    assert "not yet" in detail.lower()


def test_a_supplier_is_required(run):
    with pytest.raises(ValueError, match="supplier_id"):
        run(good(), scope={k: v for k, v in SCOPE.items() if k != "supplier_id"})


def test_a_period_that_ends_before_it_starts_is_refused(run):
    with pytest.raises(ValueError, match="period"):
        run(good(), scope={**SCOPE, "period_start": "2026-10-01", "period_end": "2026-09-01"})


def test_the_composer_is_told_not_to_name_the_supplier_or_invent_numbers():
    note = factpack.composer_note_for("supplier_qbr")
    text = note if isinstance(note, str) else " ".join(map(str, note.values() if hasattr(note, "values") else note))
    assert "never write the supplier's name" in text.lower() or "never write a supplier's name" in text.lower()
    assert "not measured" in text.lower() or "say so plainly" in text.lower()


def test_same_inputs_give_the_same_pack(run):
    a, b = run(good()), run(good())
    assert a.pack_id == b.pack_id
    assert [(f.label, f.value) for f in a.facts] == [(f.label, f.value) for f in b.facts]
