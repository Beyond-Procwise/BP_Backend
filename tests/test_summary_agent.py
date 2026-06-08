import json
import src.services.summary_agent as sa


class _FakeCursor:
    """Matches a SQL substring -> (columns, rows). Supports fetchone/fetchall."""

    def __init__(self, table_data, recorder=None):
        self._table_data = table_data
        self._recorder = recorder
        self.description = []
        self._rows = []

    def execute(self, sql, params=()):
        if self._recorder is not None:
            self._recorder.append((sql, params))
        for needle, (cols, rows) in self._table_data.items():
            if needle in sql:
                self.description = [(c,) for c in cols]
                self._rows = list(rows)
                return
        self.description = []
        self._rows = []

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return self._rows

    def close(self):
        pass


class _FakeConn:
    def __init__(self, table_data, recorder=None):
        self._cur = _FakeCursor(table_data, recorder)
        self.committed = False

    def cursor(self):
        return self._cur

    def commit(self):
        self.committed = True


def test_resolve_persona_hits_bp_prompt():
    conn = _FakeConn({
        "FROM proc.bp_prompt": (
            ["prompts_desc"],
            [({"prompt_template": "You are a compliance auditor."},)],
        ),
    })
    framing, source = sa.resolve_persona("compliance", conn)
    assert framing == "You are a compliance auditor."
    assert source == "bp_prompt"


def test_resolve_persona_falls_back_to_raw():
    conn = _FakeConn({"FROM proc.bp_prompt": (["prompts_desc"], [])})
    framing, source = sa.resolve_persona("some ad-hoc persona", conn)
    assert framing == "some ad-hoc persona"
    assert source == "raw"


def _portfolio_conn():
    return _FakeConn({
        "count(*) FROM proc.bp_invoice_trgt": (["count"], [(2,)]),
        "count(*) FROM proc.bp_purchase_order_trgt": (["count"], [(1,)]),
        "count(*) FROM proc.bp_quote_trgt": (["count"], [(3,)]),
        "SUM(converted_amount_usd),0) FROM proc.bp_invoice_trgt": (["s"], [(1500.0,)]),
        "FROM proc.bp_purchase_order_trgt t": (["s"], [(800.0,)]),
        "GROUP BY supplier_id": (["supplier_id", "usd"], [("SUP-A", 1200.0), ("SUP-B", 300.0)]),
        "GROUP BY currency": (["currency", "n"], [("USD", 2)]),
        "FROM proc.bp_extraction_discrepancy": (["count"], [(4,)]),
        "FROM proc.bp_agent_actions": (["count"], [(7,)]),
    })


def test_gather_portfolio_context_aggregates():
    ctx = sa.gather_portfolio_context(_portfolio_conn())
    assert ctx is not None
    assert ctx["scope"] == "portfolio"
    assert ctx["totals"]["invoices"] == 2
    assert ctx["totals"]["quotes"] == 3
    assert ctx["totals"]["invoice_spend_usd"] == 1500.0
    assert ctx["top_suppliers"][0]["supplier_id"] == "SUP-A"
    assert ctx["sources"]["discrepancies"] == 4


def test_gather_portfolio_context_empty_returns_none():
    conn = _FakeConn({
        "count(*) FROM proc.bp_invoice_trgt": (["count"], [(0,)]),
        "count(*) FROM proc.bp_purchase_order_trgt": (["count"], [(0,)]),
        "count(*) FROM proc.bp_quote_trgt": (["count"], [(0,)]),
    })
    assert sa.gather_portfolio_context(conn) is None


def test_build_persona_prompt_includes_framing_rules_and_facts():
    facts = {"scope": "portfolio", "totals": {"invoices": 2}}
    prompt = sa._build_persona_prompt("You are a compliance auditor.", facts)
    assert "You are a compliance auditor." in prompt
    # grounded no-fabrication rule reused from deal_summary._build_prompt
    assert "Do not fabricate" in prompt
    # the facts are embedded as JSON
    assert '"invoices": 2' in prompt


def test_store_summary_flips_current_and_inserts():
    rec = []
    conn = _FakeConn({}, recorder=rec)
    out = sa._store_summary(
        conn,
        persona="compliance",
        persona_source="bp_prompt",
        scope="deal",
        deal_id="D-9",
        summary="text",
        data_snapshot={"a": 1},
        sources={"invoices": 1},
        model="gpt-oss:120b",
        is_current=True,
    )
    assert out["summary_id"]
    assert out["persona"] == "compliance"
    assert out["deal_id"] == "D-9"
    assert "generated_at" in out
    assert conn.committed is True
    sqls = " ".join(s for s, _ in rec)
    assert "UPDATE proc.bp_summary SET is_current = false" in sqls
    assert "INSERT INTO proc.bp_summary" in sqls


def test_store_summary_as_of_does_not_flip_current():
    rec = []
    conn = _FakeConn({}, recorder=rec)
    sa._store_summary(
        conn, persona="compliance", persona_source="raw", scope="deal",
        deal_id="D-9", summary="t", data_snapshot={}, sources=None,
        model="m", is_current=False,
    )
    sqls = " ".join(s for s, _ in rec)
    assert "UPDATE proc.bp_summary SET is_current = false" not in sqls
    assert "INSERT INTO proc.bp_summary" in sqls
