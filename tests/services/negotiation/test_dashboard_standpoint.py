"""The Negotiate page shows a position it can defend, or it shows none.

Until 2026-09-05 `negotiation_strategy` emitted:

    supplier_rate = 50; our_aim = 47; walk_away = 50
    if quote and actual and quote > 0:
        supplier_rate = round(actual / quote * 50)
        our_aim = max(0, supplier_rate - 3)
        walk_away = supplier_rate

`ourAim` is labelled "optimalPrice" in the payload the buyer sees. It was
"three index points below wherever the supplier already is" -- an arbitrary
constant, not an optimum. `walkAway` was set equal to the supplier's own rate,
which as a walk-away threshold says "we will never walk away". With no quote or
no actual, the literals 50/47/50 shipped unchanged.

Authorised behaviour change: the block now reports what it measured
(`supplierRate`) and declines to invent the two it cannot derive.
"""
import src.services.negotiate_dashboard as nd


class _Cur:
    def execute(self, *a, **k):
        return None

    def fetchall(self):
        return []

    def fetchone(self):
        return None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _patch(monkeypatch, **deal):
    base = {"deal_id": "D-1", "supplier_id": "SUP-1", "currency": "GBP",
            "quote_total": None, "po_total": None, "invoice_total": None,
            "last_activity_date": None}
    base.update(deal)
    monkeypatch.setattr(nd, "_deal", lambda cur, deal_id: base)
    monkeypatch.setattr(nd, "_supplier_insights", lambda cur, d: ("p", "k", "r"))
    monkeypatch.setattr(nd, "_advice_plays", lambda deal_id: [])
    return nd.negotiation_strategy(_Cur(), "D-1")[0]


class TestNoFabricatedPosition:
    def test_optimal_price_is_not_invented_from_an_offset(self, monkeypatch):
        out = _patch(monkeypatch, quote_total=100.0, po_total=90.0)
        assert out["preferredOutcome"]["optimalPrice"] is None

    def test_walk_away_is_not_the_suppliers_own_rate(self, monkeypatch):
        out = _patch(monkeypatch, quote_total=100.0, po_total=90.0)
        assert out["currentStandpoint"]["walkAway"] is None
        assert out["preferredOutcome"]["walkAway"] is None

    def test_the_old_literals_never_ship(self, monkeypatch):
        """50/47/50 with no deal figures at all."""
        out = _patch(monkeypatch)
        stand = out["currentStandpoint"]
        assert stand["supplierRate"] is None
        assert stand["walkAway"] is None
        assert out["preferredOutcome"]["optimalPrice"] is None

    def test_target_range_is_not_a_string_of_invented_numbers(self, monkeypatch):
        out = _patch(monkeypatch, quote_total=100.0, po_total=90.0)
        assert out["preferredOutcome"]["targetRange"] is None

    def test_unavailable_positions_are_explained(self, monkeypatch):
        out = _patch(monkeypatch, quote_total=100.0, po_total=90.0)
        assert out["preferredOutcome"].get("unavailableReason")


class TestMeasuredPartsSurvive:
    def test_supplier_rate_is_still_measured_when_derivable(self, monkeypatch):
        out = _patch(monkeypatch, quote_total=100.0, po_total=90.0)
        assert out["currentStandpoint"]["supplierRate"] == 45

    def test_summary_and_plays_unaffected(self, monkeypatch):
        out = _patch(monkeypatch, quote_total=100.0, po_total=90.0)
        assert "highLevelSummary" in out
        assert out["plays"] == []
        assert "supplierInsights" in out
