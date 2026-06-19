import importlib

mod = importlib.import_module("src.services.deal_analysis_service")

SAMPLE = {
    "deal_id": "DEAL-1", "deal_name": "DEAL-1", "supplier": "Acme",
    "category": "Electronics", "deal_value": 1000.0, "currency": "GBP",
    "volume": 200.0, "unit_price": 5.0, "price_change_pct": 4.17,
    "volume_change_pct": 0.0, "efficiency_score": 40.0,
    "items": [{"name": "Widget A", "qty": 100, "unit_price": 8.0}],
    "item_count": 1, "data_snapshot": {"invoice_total": 1000.0},
}


class FakeCur:
    def __init__(self):
        self.executed = []
        self._rows = []

    def execute(self, sql, params=()):
        self.executed.append((sql, params))
        if "select distinct deal_id" in sql:
            self._rows = [("DEAL-1",), ("DEAL-2",)]

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return None


class FakeConn:
    def __init__(self):
        self._cur = FakeCur()

    def cursor(self):
        return self._cur

    def commit(self):
        pass


def test_upsert_demotes_then_inserts():
    conn = FakeConn()
    aid = mod.upsert_analysis_row(conn, SAMPLE, "sum-123", "BeyondProcwise/AgentNick:unified")
    sqls = " | ".join(s for s, _ in conn._cur.executed)
    assert "UPDATE proc.bp_analysis_summary SET is_current = false" in sqls
    assert "INSERT INTO proc.bp_analysis_summary" in sqls
    assert isinstance(aid, str) and len(aid) > 0


def test_sync_processes_linked_deals(monkeypatch):
    calls = []
    monkeypatch.setattr(mod, "_linked_deal_ids_needing_summary",
                        lambda cur: ["DEAL-1", "DEAL-2"])
    monkeypatch.setattr(mod, "generate_for_deal",
                        lambda deal_id, conn: calls.append(deal_id) or {"deal_id": deal_id})
    res = mod.sync_deal_summaries(conn=FakeConn(), max_workers=2)
    assert res["processed"] == 2
    assert set(calls) == {"DEAL-1", "DEAL-2"}


def test_sync_one_failure_isolated(monkeypatch):
    def boom(deal_id, conn):
        if deal_id == "DEAL-2":
            raise RuntimeError("llm down")
        return {"deal_id": deal_id}
    monkeypatch.setattr(mod, "_linked_deal_ids_needing_summary",
                        lambda cur: ["DEAL-1", "DEAL-2"])
    monkeypatch.setattr(mod, "generate_for_deal", boom)
    res = mod.sync_deal_summaries(conn=FakeConn(), max_workers=2)
    assert res["processed"] == 1
    assert res["failed"] == 1
