"""The supplier criticality review, measured from what the platform can actually trace.

The brief's source for this report -- a criticality scoring system (CISE) -- does not exist,
and the third-party risk register holds seeded rows that match no supplier we buy from. So
the review states concentration, the largest suppliers and the open findings against them,
all measured, and says plainly that criticality itself is not assessed (ruled 2026-09-24).

Queries are replaced with canned rows: what is under test is the arithmetic and the honesty
rules, not Postgres (the live half is test_supplier_review_live.py).
"""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from src.services.analytics.models import Confidence
from src.services.rga.builders import supplier_criticality_review as scr
from src.services.rga.factpack import FactBuilder, registered_types
from src.services.rga.models import FindingCode

SCOPE = {"period_start": "2026-04-01", "period_end": "2026-06-30",
         "period_label": "2026 Q2", "currency": "GBP"}
RATES = {"GBP": Decimal("0.8"), "EUR": Decimal("0.9")}      # USD-quoted

# supplier_id, supplier_name, invoice_total, currency
AMOUNTS = [
    ("SUP-A", "Ashcroft Associates 10", Decimal("400"), "GBP"),
    ("SUP-A", "Ashcroft Associates 10", Decimal("200"), "GBP"),
    ("SUP-B", "Birch Ltd", Decimal("300"), "GBP"),
    ("SUP-C", "Cedar plc", Decimal("90"), "EUR"),          # = 80 GBP
    ("SUP-D", "Dune & Co", Decimal("10"), "GBP"),
    ("SUP-E", "Elm GmbH", Decimal("5"), "GBP"),
    ("SUP-F", "Fir SA", Decimal("5"), "GBP"),
]
FINDINGS = [("SUP-A", 3), ("SUP-C", 1), ("SUP-F", 2)]


def _build(monkeypatch, *, amounts=AMOUNTS, findings=FINDINGS, rates=RATES, suppliers=6):
    def fetch(sql, params):
        assert params == ("2026-04-01", "2026-06-30")
        if sql is scr._SUPPLIERS:
            return [(suppliers,)]
        if sql is scr._AMOUNTS:
            return list(amounts)
        if sql is scr._FINDINGS:
            return list(findings)
        raise AssertionError(sql)
    monkeypatch.setattr(scr, "_fetch", fetch)
    monkeypatch.setattr(scr, "_rates", lambda: (rates, datetime(2026, 9, 24, tzinfo=timezone.utc),
                                                not rates))
    fb = FactBuilder(pack_id="FP-test", scope=dict(SCOPE), as_of="2026-09-24")
    scr.build(fb)
    return fb


def _by_label(fb):
    return {f.label: f for f in fb.facts}


def test_it_is_a_registered_report_type():
    assert "supplier_criticality_review" in registered_types()


def test_the_book_is_measured_in_one_currency(monkeypatch):
    f = _by_label(_build(monkeypatch))
    assert f["Suppliers transacted with"].value == 6
    spend = f["Invoiced spend (GBP)"]
    assert spend.value == Decimal("1000") and spend.confidence is Confidence.CORROBORATED


def test_concentration_is_stated_from_the_converted_totals(monkeypatch):
    f = _by_label(_build(monkeypatch))
    assert f["Largest supplier's share of spend"].value == Decimal("60")          # 600 / 1000
    assert f["Top five suppliers' share of spend"].value == Decimal("99.5")       # 995 / 1000
    assert f["Suppliers making up four-fifths of spend"].value == 2                       # 600 + 300


def test_the_top_five_are_named_ranked_and_carry_their_findings(monkeypatch):
    fb = _build(monkeypatch)
    f = _by_label(fb)
    assert f["Spend with Ashcroft Associates 10 (GBP)"].value == Decimal("600")
    assert f["Share of spend — Birch Ltd"].value == Decimal("30")
    assert f["Spend with Cedar plc (GBP)"].value == Decimal("80")                 # EUR converted
    assert f["Open findings — Ashcroft Associates 10"].value == 3
    assert f["Open findings — Birch Ltd"].value == 0                             # measured zero
    # Ranked, and only five: Elm and Fir tie on 5; the tie breaks on supplier id, so Elm is in.
    named = [x.label for x in fb.facts if x.label.startswith("Spend with ")]
    assert named == ["Spend with Ashcroft Associates 10 (GBP)", "Spend with Birch Ltd (GBP)",
                     "Spend with Cedar plc (GBP)", "Spend with Dune & Co (GBP)",
                     "Spend with Elm GmbH (GBP)"]
    assert f["Open findings against the period's deals"].value == 6


def test_what_cannot_be_measured_is_said_to_be_unassessed(monkeypatch):
    f = _by_label(_build(monkeypatch))
    for label in ("Supplier criticality rating", "Single-source dependence",
                  "Spend covered by a contract"):
        assert f[label].value is None and f[label].confidence is Confidence.UNASSESSED, label


def test_the_fixed_measures_keep_their_ids_whatever_the_period(monkeypatch):
    """A hand-written or edited report refers to F0001..F0009 -- they must not move."""
    one = _build(monkeypatch)
    two = _build(monkeypatch, amounts=AMOUNTS[:3], suppliers=2)
    assert [(f.fact_id, f.label) for f in one.facts[:9]] == \
           [(f.fact_id, f.label) for f in two.facts[:9]]


def test_a_partial_conversion_is_unassessed_and_said(monkeypatch):
    fb = _build(monkeypatch, amounts=AMOUNTS + [("SUP-G", "Gum Inc", Decimal("50"), "INR")])
    f = _by_label(fb)
    assert f["Invoiced spend (GBP)"].confidence is Confidence.UNASSESSED
    assert f["Largest supplier's share of spend"].confidence is Confidence.UNASSESSED
    assert any(x.code is FindingCode.PARTIAL_CURRENCY_CONVERSION for x in fb.findings)


def test_no_rates_means_no_money_and_no_shares(monkeypatch):
    f = _by_label(_build(monkeypatch, rates={}))
    assert f["Invoiced spend (GBP)"].value is None
    assert f["Largest supplier's share of spend"].value is None
    assert not any(label.startswith("Spend with ") for label in f)
    assert f["Open findings against the period's deals"].value == 6              # still countable


def test_a_period_with_no_spend_states_nothing_it_cannot(monkeypatch):
    f = _by_label(_build(monkeypatch, amounts=[], findings=[], suppliers=0))
    assert f["Suppliers transacted with"].value == 0
    assert f["Invoiced spend (GBP)"].value is None
    assert f["Largest supplier's share of spend"].value is None


def test_its_sections_come_in_the_review_s_own_order():
    from src.services.rga.style import resolve_style_brief
    brief = resolve_style_brief("supplier_criticality_review")
    assert brief.get("report.style.section_order") == scr.SECTION_ORDER
    assert brief.provenance["report.style.section_order"] == "report_type:supplier_criticality_review"
    # The exec summary is unchanged: it registered no order of its own.
    assert resolve_style_brief("exec_procurement_summary").provenance[
        "report.style.section_order"] == "platform_default"


def test_a_tie_breaks_on_the_supplier_id_whatever_order_the_rows_arrive_in(monkeypatch):
    rows = [r for r in AMOUNTS if r[0] == "SUP-F"] + [r for r in AMOUNTS if r[0] != "SUP-F"]
    named = [x.label for x in _build(monkeypatch, amounts=rows).facts
             if x.label.startswith("Spend with ")]
    assert named[-1] == "Spend with Elm GmbH (GBP)"


def test_reaching_exactly_four_fifths_counts_as_reaching_it(monkeypatch):
    rows = [("SUP-A", "A", Decimal("500"), "GBP"), ("SUP-B", "B", Decimal("300"), "GBP"),
            ("SUP-C", "C", Decimal("200"), "GBP")]
    f = _by_label(_build(monkeypatch, amounts=rows, findings=[], suppliers=3))
    assert f["Suppliers making up four-fifths of spend"].value == 2


def test_the_composer_is_told_to_name_suppliers_only_through_their_figures(pack, brief):
    """Live, 2026-09-24: the model wrote 'Lighthouse Associates 13' into a sentence and the
    report was refused for a typed number. The review tells it how to name a supplier."""
    from src.services.rga.compose import build_prompt
    prompt = build_prompt(pack, brief, "supplier_criticality_review")
    assert "never write a supplier's name in a sentence" in prompt
    assert "never write a supplier's name" not in build_prompt(pack, brief,
                                                              "exec_procurement_summary")


def test_no_fixed_label_carries_a_digit_a_composer_could_copy(monkeypatch):
    """Live: 'Suppliers making up 80% of spend' put an 80 in the model's prose."""
    fb = _build(monkeypatch)
    assert not any(ch.isdigit() for f in fb.facts[:9] for ch in f.label)


def test_the_composer_sees_suppliers_by_rank_never_by_name(monkeypatch):
    """Live, 2026-09-24: told not to, the model still copied 'Lighthouse Associates 13' into
    prose in 3 runs of 3. A name it never sees is a name it cannot copy; the figure cards still
    print the real name, from the fact's own label."""
    from src.services.rga.compose import build_prompt
    from src.services.rga.factpack import FactPack
    from src.services.rga.style import resolve_style_brief
    fb = _build(monkeypatch)
    pack = FactPack(pack_id="FP-test", report_type_id=scr.REPORT_TYPE_ID, scope=dict(SCOPE),
                    as_of="2026-09-24", generated_at=datetime(2026, 9, 24, tzinfo=timezone.utc),
                    generated_by="tests", facts=fb.facts, findings=fb.findings)
    prompt = build_prompt(pack, resolve_style_brief(scr.REPORT_TYPE_ID), scr.REPORT_TYPE_ID)
    for name in ("Ashcroft Associates 10", "Birch Ltd", "Cedar plc", "Dune & Co", "Elm GmbH"):
        assert name not in prompt, name
    assert "Spend with the largest supplier (GBP)" in prompt
    assert "Share of spend — the second largest supplier" in prompt
    assert "Open findings — the fifth largest supplier" in prompt
    assert pack.fact("F0010").label == "Spend with Ashcroft Associates 10 (GBP)"   # on the page


def test_each_report_type_has_its_own_title():
    from src.services.rga.factpack import title_for
    assert title_for("supplier_criticality_review") == "Supplier criticality review"
    assert title_for("exec_procurement_summary") == "Executive procurement summary"


def test_the_pipeline_draws_with_the_report_type_s_title(monkeypatch):
    import inspect
    from src.services.rga import pipeline
    assert inspect.signature(pipeline.generate_report).parameters["title"].default is None
