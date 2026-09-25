"""The board paper: one deal, in the Pipeline Approve stage's own structure (ruled 2026-09-25).

Every figure is read off the deal record; what the record does not hold -- budget, strategy,
the forum, the decisions the Board is asked to take, who has signed -- is stated as not
recorded, exactly as the Pipeline's paper does. Queries are canned; test_board_paper_live.py
runs the real ones.
"""
from __future__ import annotations

from decimal import Decimal

import pytest

from src.services.analytics.models import Confidence
from src.services.rga.builders import board_paper as bp
from src.services.rga.factpack import FactBuilder, registered_types, title_for

SCOPE = {"deal_id": "DEAL-1", "period_label": "Cloud platform renewal", "category": "SaaS / IT"}

# deal_name, supplier_id, supplier_name, quote_count, po_count, invoice_count,
# quote_total, po_total, invoice_total, currency, three_way_match, cycle_days_quote_to_po
DEAL = ("Cloud platform renewal", "SUP-A", "Ashcroft Associates 10", 3, 1, 1,
        Decimal("300000"), Decimal("300000"), Decimal("300000"), "GBP", False, 12)
# supplier_id, supplier_name, total_amount, currency  (each supplier's latest bid)
BIDS = [("SUP-A", "Ashcroft Associates 10", Decimal("300000"), "GBP"),
        ("SUP-B", "Birch Ltd", Decimal("240000"), "GBP"),
        ("SUP-C", "Cedar plc", Decimal("270000"), "GBP")]
# category, severity, count
CHECKS = [("quantity", "critical", 2), ("quantity", "warning", 1), ("price", "critical", 1)]
OPPS = [(2, Decimal("15000"))]


def _build(monkeypatch, *, deal=DEAL, bids=BIDS, checks=CHECKS, opps=OPPS, scope=SCOPE):
    def fetch(sql, params):
        assert params == (scope["deal_id"],)
        return {bp._DEAL: [deal] if deal else [], bp._BIDS: list(bids),
                bp._CHECKS: list(checks), bp._OPPS: list(opps)}[sql]
    monkeypatch.setattr(bp, "_fetch", fetch)
    fb = FactBuilder(pack_id="FP-bp", scope=dict(scope), as_of="2026-09-25")
    bp.build(fb)
    return fb


def _by(fb):
    return {f.label: f for f in fb.facts}


def test_it_is_registered_with_its_own_title_and_sections():
    from src.services.rga.style import resolve_style_brief
    assert "board_paper" in registered_types()
    assert title_for("board_paper") == "Board paper"
    assert resolve_style_brief("board_paper").get("report.style.section_order") == [
        "overview", "recommendation", "background", "risks", "benefits", "approvals"]


def test_the_four_headline_tiles(monkeypatch):
    f = _by(_build(monkeypatch))
    assert f["Total value (GBP)"].value == Decimal("300000")
    assert f["Identified benefit (GBP)"].value == Decimal("15000")
    for label in ("Total budget", "Strategy"):
        assert f[label].value is None and f[label].confidence is Confidence.UNASSESSED


def test_what_the_record_does_not_hold_is_said_to_be_not_recorded(monkeypatch):
    f = _by(_build(monkeypatch))
    for label in ("Board forum", "Decisions the Board is asked to take",
                  "Approval decisions recorded"):
        assert f[label].value is None and f[label].confidence is Confidence.UNASSESSED, label


def test_the_evidence_behind_the_deal(monkeypatch):
    f = _by(_build(monkeypatch))
    assert (f["Documents on file"].value, f["Quotes on file"].value,
            f["Purchase orders on file"].value, f["Invoices on file"].value) == (5, 3, 1, 1)
    assert f["Three-way match rate for this deal"].value == 0
    assert f["Quote-to-PO cycle"].value == 12
    assert (f["Open checks"].value, f["Critical checks"].value, f["Warning checks"].value) == (4, 3, 1)
    assert f["Opportunities raised on this deal"].value == 2


def test_the_bids_are_compared_in_one_currency(monkeypatch):
    fb = _build(monkeypatch)
    f = _by(fb)
    assert f["Suppliers who bid"].value == 3
    assert f["Spread from the lowest to the highest bid"].value == Decimal("25")   # 60k / 240k
    assert f["Recoverable by taking the lowest bid (GBP)"].value == Decimal("60000")
    bids = [x.label for x in fb.facts if x.label.startswith("Bid from ")]
    assert bids == ["Bid from Birch Ltd (GBP)", "Bid from Cedar plc (GBP)",
                    "Bid from Ashcroft Associates 10 (GBP)"]                    # lowest first


def test_one_bidder_means_no_benchmark(monkeypatch):
    f = _by(_build(monkeypatch, bids=BIDS[:1]))
    assert f["Suppliers who bid"].value == 1
    for label in ("Spread from the lowest to the highest bid",
                  "Recoverable by taking the lowest bid (GBP)"):
        assert f[label].value is None and f[label].confidence is Confidence.UNASSESSED


def test_bids_in_different_currencies_are_not_compared(monkeypatch):
    f = _by(_build(monkeypatch, bids=BIDS[:2] + [("SUP-C", "Cedar plc", Decimal("1"), "EUR")]))
    assert f["Spread from the lowest to the highest bid"].value is None


def test_the_chosen_supplier_as_the_lowest_bidder_recovers_nothing(monkeypatch):
    bids = [("SUP-A", "Ashcroft Associates 10", Decimal("200000"), "GBP"), BIDS[1]]
    f = _by(_build(monkeypatch, bids=bids))
    assert f["Recoverable by taking the lowest bid (GBP)"].value == 0            # measured zero


def test_the_approval_route_is_stated_step_by_step(monkeypatch):
    fb = _build(monkeypatch)
    f = _by(fb)
    assert f["Approvers required for this deal"].value == 4       # 300k > 250k: CFO is in
    steps = [x.label for x in fb.facts if x.label.startswith("Approval step")]
    assert steps[-1] == ("Approval step — CFO sign-off: required. Required > £250k — this deal "
                         "is GBP 300,000.")
    assert [x.value for x in fb.facts if x.label.startswith("Approval step")] == [1, 2, 3, 4]


def test_no_category_means_no_route_and_says_why(monkeypatch):
    fb = _build(monkeypatch, scope={"deal_id": "DEAL-1", "period_label": "x"})
    f = _by(fb)
    assert f["Approvers required for this deal"].value is None
    assert not any(x.label.startswith("Approval step") for x in fb.facts)


def test_the_checks_are_grouped_by_what_was_found(monkeypatch):
    fb = _build(monkeypatch)
    groups = {x.label: x.value for x in fb.facts if x.label.startswith("Open checks — ")}
    assert groups == {"Open checks — quantity": 3, "Open checks — price": 1}


def test_the_fixed_measures_keep_their_ids(monkeypatch):
    one = _build(monkeypatch)
    two = _build(monkeypatch, bids=[], checks=[], opps=[(0, None)],
                 scope={"deal_id": "DEAL-1", "period_label": "x"})
    n = len(bp.FIXED_LABELS)
    assert [(f.fact_id, f.label) for f in one.facts[:n]] == [(f.fact_id, f.label) for f in two.facts[:n]]


def test_an_unknown_deal_states_nothing_it_cannot(monkeypatch):
    fb = _build(monkeypatch, deal=None, bids=[], checks=[], opps=[(0, None)])
    f = _by(fb)
    assert f["Total value"].value is None
    assert f["Documents on file"].value is None


def test_the_composer_sees_bids_by_rank_never_by_supplier_name(monkeypatch):
    fb = _build(monkeypatch)
    shown = [bp.composer_label(x) for x in fb.facts]
    assert "Bid from the lowest bidder (GBP)" in shown
    assert "Bid from the third lowest bidder (GBP)" in shown
    assert not any("Ashcroft" in s or "Birch" in s for s in shown)


def test_each_bid_carries_its_exact_difference_from_the_lowest(monkeypatch):
    """Live 2026-09-25: three bids all DISPLAY as £1.2M (the shared formatter is compact), and
    the model wrote 'all bids were identical'. Each bid now carries its measured difference."""
    fb = _build(monkeypatch)
    diffs = {x.label: x.value for x in fb.facts if x.label.startswith("Above the lowest bid")}
    assert diffs == {"Above the lowest bid — Birch Ltd (GBP)": 0,
                     "Above the lowest bid — Cedar plc (GBP)": Decimal("30000"),
                     "Above the lowest bid — Ashcroft Associates 10 (GBP)": Decimal("60000")}
    shown = [bp.composer_label(x) for x in fb.facts]
    assert "Above the lowest bid — the second lowest bidder (GBP)" in shown
    assert not any("Birch" in s for s in shown)


def test_the_composer_is_held_to_what_the_record_says(monkeypatch):
    """Live 2026-09-25: the model wrote that 'the purchase order was issued after all required
    approvals were completed' -- no approval is recorded -- and recommended approval with a
    critical check open. Words the post-check cannot see, so the composer is told."""
    note = bp.deal_note(_build(monkeypatch).facts)
    assert "never say anyone has approved, signed or cleared this deal" in note
    assert "Do not recommend that the Board approves or rejects the deal" in note
    assert "at most one findings list" in bp.COMPOSER_NOTE
    assert "Displayed amounts are exact" in bp.COMPOSER_NOTE


def test_amounts_on_a_board_paper_are_exact(monkeypatch):
    """Live: bids of £1,309,000, £1,270,000 and £1,248,000 all DISPLAYED as £1.2M-£1.3M, and the
    model called them identical three times. A board decides on the exact figure."""
    f = _by(_build(monkeypatch))
    assert f["Total value (GBP)"].display == "£300,000"
    assert f["Bid from Cedar plc (GBP)"].display == "£270,000"
    assert f["Above the lowest bid — Birch Ltd (GBP)"].display == "£0"


def test_an_exact_amount_keeps_its_pence():
    from src.services.rga.models import FactEntry, FormatHint, Origin
    e = FactEntry(fact_id="F0001", label="x", value=Decimal("1234.5"), currency="GBP",
                  format_hint=FormatHint.MONEY_EXACT, confidence=Confidence.CORROBORATED,
                  origin=Origin.OBSERVED, provenance_id="p", derivation="d")
    assert e.display == "£1,234.50"


def test_the_composer_is_given_this_deal_s_recommendations_and_only_those(monkeypatch):
    """The Pipeline paper derives its recommendations from the record (dealRecommendations);
    so does this one, and the composer is told to write those and no others."""
    fb = _build(monkeypatch)
    note = bp.deal_note(fb.facts)
    assert "Resolve the {{F0016}} critical checks open against this deal before anything else." in note
    assert "Reconcile the quote, purchase order and invoice: they do not three-way match." in note
    assert "Taking the lowest bid would recover {{F0012}}." in note
    assert "{{F0010}} suppliers bid, so there is a competing benchmark." in note
    one = bp.deal_note(_build(monkeypatch, bids=BIDS[:1], checks=[],
                              deal=DEAL[:10] + (True, 12)).facts)
    assert "Only one supplier bid, so there is no competing benchmark." in one
    clean = bp.deal_note(_build(monkeypatch, checks=[], deal=DEAL[:10] + (True, 12),
                                bids=[("SUP-A", "A", Decimal("1"), "GBP"),
                                      ("SUP-B", "B", Decimal("2"), "GBP")]).facts)
    rec = clean.split("RECOMMEND")[1].split("STATE")[0]
    assert "critical" not in rec and "Reconcile" not in rec
    assert "Nothing measured on this deal is asking for a decision" in rec
    assert "The chosen supplier's bid is already the lowest bid." in clean
    import re as _re
    assert not _re.search(r"(?<!F)\d", _re.sub(r"\{\{F\d{4}\}\}", "", note)), "a typed digit a composer would copy"


@pytest.mark.parametrize("text", [
    "The deal has progressed through the required approval steps.",
    "The deal has passed through four required approval steps, though no decisions are recorded.",
    "The deal has been approved by the CFO.",
    "All approvals were completed before the order was issued.",
])
def test_a_sentence_claiming_an_approval_is_a_fault(text):
    """Live 2026-09-25, even when told not to: no approval is recorded for any deal."""
    from src.services.rga.models import NarrativeBlock, ReportAST, Section
    ast = ReportAST(sections=[Section(id="a", title="Approvals",
                                      blocks=[NarrativeBlock(text=text)])])
    assert bp.prose_faults(ast)


@pytest.mark.parametrize("text", [
    "No approval decision is recorded against the deal.",
    "The deal has not been formally approved.",
    "The approval route is defined but not completed.",
    "It cannot be considered approved until all steps are completed.",
])
def test_saying_no_approval_is_recorded_is_not_a_fault(text):
    from src.services.rga.models import NarrativeBlock, ReportAST, Section
    ast = ReportAST(sections=[Section(id="a", title="Approvals",
                                      blocks=[NarrativeBlock(text=text)])])
    assert bp.prose_faults(ast) == []


def test_a_draft_that_breaks_a_prose_rule_is_sent_back_once(monkeypatch):
    from src.services.rga import compose
    from src.services.rga.models import NarrativeBlock, ReportAST, Section
    bad = ReportAST(sections=[Section(id="a", title="A", blocks=[
        NarrativeBlock(text="The deal has passed through its approval steps.")])])
    good = ReportAST(sections=[Section(id="a", title="A", blocks=[
        NarrativeBlock(text="No approval decision is recorded.")])])
    drafts = iter([bad, good])
    asks = []
    monkeypatch.setattr(compose, "_attempt", lambda raw, pack: (next(drafts), None))

    def generate(prompt, **k):
        asks.append(prompt)
        return "{}"
    from tests.services.rga.conftest import make_pack
    pack = make_pack([]).model_copy(update={"report_type_id": "board_paper"})
    from src.services.rga.style import resolve_style_brief
    ast = compose.compose_report(pack, resolve_style_brief("board_paper"), "board_paper",
                                 generate=generate, emit_audit=False)
    assert ast == good and len(asks) == 2
    assert "no approval decision is recorded" in asks[1].lower()
