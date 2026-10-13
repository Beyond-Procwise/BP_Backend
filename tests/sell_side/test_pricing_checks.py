"""The limits a quote is tested against. Unset is never zero."""
import pytest

from src.services import governed_limits
from src.services.sell_side import pricing_checks as pc


def _stated(monkeypatch, rules):
    def _limit(policy, rule, **kw):
        assert policy == "sales_quote_pricing"
        if rules is None or rule not in rules:
            raise governed_limits.LimitUnavailable(rule)
        return rules[rule]
    monkeypatch.setattr(pc.governed_limits, "limit", _limit)


def test_limits_nobody_has_set_are_none_and_say_so(monkeypatch):
    _stated(monkeypatch, {"margin_floor_pct": None, "discount_corridor_max_pct": None})
    assert pc.for_quote("Any") == {
        "costToServe": None, "discountCorridor": None, "marginFloorPct": None,
        "sources": {"costToServe": "not_modelled", "discountCorridor": "not_set",
                    "marginFloorPct": "not_set"}}


def test_limits_that_cannot_be_read_are_none_and_say_that_instead(monkeypatch):
    _stated(monkeypatch, None)
    checks = pc.for_quote("Any")
    assert checks["marginFloorPct"] is None and checks["discountCorridor"] is None
    assert checks["sources"]["marginFloorPct"] == "unavailable"


def test_a_stated_floor_and_one_corridor_for_every_category(monkeypatch):
    _stated(monkeypatch, {"margin_floor_pct": 30, "discount_corridor_max_pct": 8})
    checks = pc.for_quote("Platform / Enterprise")
    assert checks["marginFloorPct"] == 30.0 and checks["discountCorridor"] == {"maxPct": 8.0}
    assert checks["sources"]["marginFloorPct"] == "policy"


def test_a_corridor_by_category_falls_back_to_its_default_and_then_to_not_set(monkeypatch):
    _stated(monkeypatch, {"margin_floor_pct": None,
                          "discount_corridor_max_pct": {"Platform / Enterprise": 8, "default": 5}})
    assert pc.for_quote("Platform / Enterprise")["discountCorridor"] == {"maxPct": 8.0}
    assert pc.for_quote("Logistics")["discountCorridor"] == {"maxPct": 5.0}
    _stated(monkeypatch, {"margin_floor_pct": None,
                          "discount_corridor_max_pct": {"Platform / Enterprise": 8}})
    checks = pc.for_quote("Logistics")
    assert checks["discountCorridor"] is None
    assert checks["sources"]["discountCorridor"] == "not_set"


@pytest.mark.parametrize("bad", ["30", True, -1, 101, [30]])
def test_a_floor_that_is_not_a_percentage_is_not_used(monkeypatch, bad):
    _stated(monkeypatch, {"margin_floor_pct": bad, "discount_corridor_max_pct": None})
    checks = pc.for_quote(None)
    assert checks["marginFloorPct"] is None and checks["sources"]["marginFloorPct"] == "not_set"


def test_a_floor_of_zero_is_a_stated_zero_not_unset(monkeypatch):
    _stated(monkeypatch, {"margin_floor_pct": 0, "discount_corridor_max_pct": 0})
    checks = pc.for_quote(None)
    assert checks["marginFloorPct"] == 0.0 and checks["discountCorridor"] == {"maxPct": 0.0}
    assert checks["sources"]["marginFloorPct"] == "policy"
