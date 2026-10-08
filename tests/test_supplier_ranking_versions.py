"""Supplier ranking stands each bid at its latest version (2026-10-08)."""
import os
import sys

import pandas as pd
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from agents.supplier_ranking_agent import SupplierRankingAgent  # noqa: E402


def _load(monkeypatch, rows):
    agent = SupplierRankingAgent.__new__(SupplierRankingAgent)
    quotes = pd.DataFrame(rows, columns=["quote_id", "deal_id", "supplier_id", "quote_date",
                                         "total_amount", "currency"])
    monkeypatch.setattr(agent, "_read_table", lambda *a, **k: quotes)
    return agent._load_deal_quotes("D1").set_index("supplier_id")


def test_a_bid_with_no_supplier_is_still_a_rival(monkeypatch):
    out = _load(monkeypatch, [
        ("CL-1", "D1", "Condor", None, 210000.0, "GBP"),
        ("SDP-9", "D1", None, None, 207656.0, "GBP"),
        ("SDP-9 (V3)", "D1", None, None, 199806.0, "GBP"),
    ])
    assert set(out.index) == {"Condor", "quote SDP-9"}
    assert out.loc["quote SDP-9", "final_quote_amount"] == pytest.approx(199806.0)


def test_an_unpriced_latest_version_is_not_replaced_by_an_older_price(monkeypatch):
    out = _load(monkeypatch, [
        ("A-1", "D1", "SA", None, 100.0, "GBP"),
        ("A-1 (V2)", "D1", "SA", None, 95.0, "GBP"),
        ("A-1 (V3)", "D1", "SA", None, None, "GBP"),
        ("B-1", "D1", "SB", None, 110.0, "GBP"),
    ])
    assert "SA" not in out.index           # no standing price -- not ranked at V2's 95
    assert list(out.index) == ["SB"]


def test_two_lots_from_one_supplier_are_both_counted(monkeypatch):
    out = _load(monkeypatch, [
        ("LOT-A", "D1", "SA", None, 100.0, "GBP"), ("LOT-A (V2)", "D1", "SA", None, 90.0, "GBP"),
        ("LOT-B", "D1", "SA", None, 50.0, "GBP"),
        ("B-1", "D1", "SB", None, 130.0, "GBP"),
    ])
    assert out.loc["SA", "final_quote_amount"] == pytest.approx(140.0)
    assert out.loc["SA", "quote_rounds"] == 2
