"""Endpoint behavior: pre-stored summaries + graceful 'Summary not available'.

These call the router handler functions directly with a faked DB connection
(the handlers import get_conn lazily, so we patch src.services.db.get_conn).
"""
import importlib
from datetime import datetime, timezone

import src.services.db as db

mod = importlib.import_module("src.api.routers.deal_summary")


class _FakeCur:
    def __init__(self, rows, desc):
        self._rows = rows
        self.description = [(c,) for c in desc]

    def execute(self, sql, params=()):
        pass

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return self._rows


class _FakeConn:
    def __init__(self, rows, desc):
        self._cur = _FakeCur(rows, desc)

    def cursor(self):
        return self._cur

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _patch(monkeypatch, rows, desc):
    monkeypatch.setattr(db, "get_conn", lambda: _FakeConn(rows, desc))


_ANALYSIS_DESC = ["deal_id", "deal_name", "supplier", "category", "deal_value",
                  "currency", "volume", "unit_price", "price_change_pct",
                  "volume_change_pct", "efficiency_score", "items", "item_count"]


def test_analysis_summary_not_available(monkeypatch):
    _patch(monkeypatch, [], _ANALYSIS_DESC)
    res = mod.get_analysis_summary("NOPE")
    assert res["row"] is None
    assert res["message"] == "Summary not available"


def test_analysis_summary_all_empty_message(monkeypatch):
    _patch(monkeypatch, [], _ANALYSIS_DESC)
    res = mod.get_analysis_summary_all()
    assert res["count"] == 0
    assert res["rows"] == []
    assert res["message"] == "Summary not available"


def test_analysis_summary_returns_row(monkeypatch):
    row = ("DEAL-1", "Deal 1", "Acme", "Electronics", 1000.0, "GBP", 200.0,
           5.0, 7.5, -1.7, 40.0, [{"name": "Widget A"}], 1)
    _patch(monkeypatch, [row], _ANALYSIS_DESC)
    res = mod.get_analysis_summary("DEAL-1")
    assert "message" not in res
    assert res["row"]["id"] == "DEAL-1"
    assert res["row"]["items"] == "Widget A"


def test_deal_summary_not_available(monkeypatch):
    _patch(monkeypatch, [], ["summary", "model", "generated_at"])
    res = mod.get_deal_summary("NOPE")
    assert res["summary"] is None
    assert res["message"] == "Summary not available"


def test_deal_summary_null_summary_not_available(monkeypatch):
    # analysis row exists but its narrative is NULL (narrative generation failed)
    ts = datetime(2026, 6, 19, tzinfo=timezone.utc)
    _patch(monkeypatch, [(None, None, ts)], ["summary", "model", "generated_at"])
    res = mod.get_deal_summary("DEAL-1")
    assert res["summary"] is None
    assert res["message"] == "Summary not available"


def test_deal_summary_returns_stored(monkeypatch):
    ts = datetime(2026, 6, 19, tzinfo=timezone.utc)
    row = ("This deal involves Acme.", "BeyondProcwise/AgentNick:unified", ts)
    _patch(monkeypatch, [row], ["summary", "model", "generated_at"])
    res = mod.get_deal_summary("DEAL-1")
    assert res["summary"] == "This deal involves Acme."
    assert res["model"] == "BeyondProcwise/AgentNick:unified"
    assert "message" not in res
