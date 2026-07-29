import pytest
from src.services.negotiation_advice import signals as sg


class _Cur:
    """Matches a SQL substring -> (columns, rows)."""

    def __init__(self, data):
        self._data = data
        self.description = []
        self._rows = []

    def execute(self, sql, params=()):
        for needle, (cols, rows) in self._data.items():
            if needle in sql:
                self.description = [(c,) for c in cols]
                self._rows = list(rows)
                return
        self.description = []
        self._rows = []

    def fetchall(self):
        return self._rows


_DEAL_COLS = ["deal_id", "supplier_id", "supplier_name", "currency",
              "quote_count", "quote_total", "po_total", "invoice_total",
              "price_variance_pct"]


def _cur(alt_count=23, risk=0.2, preferred=True, variance=8.4):
    return _Cur({
        "from proc.bp_deal_overview": (
            _DEAL_COLS,
            [("D-1", "SUP-1", "Orbis Ltd", "GBP", 1, 100000.0, 98000.0,
              105000.0, variance)],
        ),
        "bp_quote_line_items_trgt": (["n"], [(alt_count,)]),
        "from proc.bp_supplier": (["risk_score", "is_preferred_supplier"],
                                  [(risk, preferred)]),
    })


def test_gather_signals_reads_deal_and_alternatives():
    s = sg.gather_signals(_cur(), "D-1")
    assert s["supplier_id"] == "SUP-1"
    assert s["alternative_supplier_count"] == 23
    assert s["deal_value"] == 105000.0        # invoice preferred over po/quote
    assert s["risk_score"] == 0.2
    assert s["is_preferred"] is True
    assert s["price_variance_pct"] == 8.4


def test_absent_values_are_none_not_zero():
    cur = _Cur({"from proc.bp_deal_overview": (_DEAL_COLS,
               [("D-2", None, None, None, 0, None, None, None, None)])})
    s = sg.gather_signals(cur, "D-2")
    assert s["deal_value"] is None
    assert s["risk_score"] is None
    assert s["alternative_supplier_count"] is None


def test_unknown_deal_returns_none():
    assert sg.gather_signals(_Cur({}), "NOPE") is None


def test_market_context_omits_uncomputable_keys():
    market = sg.market_context_dict({"alternative_supplier_count": None,
                                     "risk_score": None})
    assert "supply_risk" not in market


def test_market_context_flags_high_supply_risk_when_few_alternatives():
    market = sg.market_context_dict({"alternative_supplier_count": 1,
                                     "risk_score": 0.8})
    # the existing scorer tests for these exact strings
    assert market["supply_risk"] in {"high", "elevated", "tight"}


def test_supplier_performance_uses_a_key_the_scorer_reads():
    perf = sg.supplier_performance_dict({"on_time_ratio": 0.72})
    assert set(perf) & {"on_time_delivery", "on_time", "delivery_score", "otif"}


def test_risk_threshold_is_on_the_0_to_100_scale():
    # 0.8 is a low risk on a 0-100 scale and must NOT flag
    assert "supply_risk" not in sg.market_context_dict(
        {"alternative_supplier_count": 200, "risk_score": 0.8})
    # 80 is genuinely high and must flag
    assert sg.market_context_dict(
        {"alternative_supplier_count": 200, "risk_score": 80.0})["supply_risk"]


def test_thin_market_uses_the_per_deal_scale():
    # 200 alternatives is a contested market, not a thin one
    assert "supply_risk" not in sg.market_context_dict(
        {"alternative_supplier_count": 200, "risk_score": 10.0})
    assert sg.market_context_dict(
        {"alternative_supplier_count": 40, "risk_score": 10.0})["supply_risk"]
