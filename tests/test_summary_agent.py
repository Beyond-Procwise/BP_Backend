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


# --- portfolio facts must survive into the prompt -------------------------
# Regression: the persona prompt was built through deal_summary._summary_facts,
# which reads only `documents` and `discrepancies`. Portfolio context has
# neither, so every real figure was dropped and the model was handed
# {"document_count": 0, "documents": [], "discrepancies": []} under an
# instruction to use ONLY those facts. All 26 stored portfolio summaries read
# "Not available" while their own data_snapshot held the real numbers.

_PORTFOLIO_CTX = {
    "scope": "portfolio",
    "totals": {"invoices": 412, "purchase_orders": 388, "quotes": 501,
               "invoice_spend_usd": 323467.17, "po_value_usd": 298001.02},
    "top_suppliers": [{"supplier_id": "SUP-1", "invoice_usd": 88120.5}],
    "currency_mix": {"GBP": 300, "EUR": 80},
    "sources": {"invoices": 412, "discrepancies": 24, "actions": 1801},
}


def test_portfolio_facts_reach_the_prompt():
    prompt = sa._build_persona_prompt("You are a procurement data analyst.",
                                      _PORTFOLIO_CTX)
    for expected in ["323467.17", "412", "SUP-1", "top_suppliers",
                     "currency_mix", "88120.5"]:
        assert expected in prompt, f"{expected!r} missing from portfolio prompt"


def test_portfolio_prompt_is_not_deal_shaped():
    prompt = sa._build_persona_prompt("You are a procurement data analyst.",
                                      _PORTFOLIO_CTX)
    low = prompt.lower()
    # It must not order a summary "of the deal" for a portfolio, nor claim the
    # portfolio has no documents.
    assert "summary of the deal" not in low
    assert '"document_count": 0' not in prompt
    assert '"documents": []' not in prompt


def test_persona_prompt_allows_next_steps_and_key_points():
    # The old base format mandated only Key Outcomes + Conclusion, so a persona
    # asking for levers, concessions or next steps could not express them.
    prompt = sa._build_persona_prompt(
        "You are a procurement negotiation strategist. Emphasize leverage "
        "points and concession opportunities.",
        _PORTFOLIO_CTX,
    )
    assert "Next Steps" in prompt
    assert "Key Points" in prompt
    # the persona's own emphasis survives
    assert "leverage points and concession opportunities" in prompt.lower()
    # and the format no longer forbids anything outside a fixed three sections
    assert "EXACTLY this format" not in prompt


def test_persona_prompt_forbids_inventing_next_steps():
    prompt = sa._build_persona_prompt("You are an auditor.", _PORTFOLIO_CTX)
    low = prompt.lower()
    assert "do not fabricate" in low
    # an empty section must be dropped rather than filled with plausible filler
    assert "omit" in low


def test_deal_scope_persona_prompt_still_carries_deal_facts():
    deal_ctx = {
        "deal_id": "D-9", "deal_name": "Acme Deal",
        "documents": {"invoices": [{"invoice_number": "INV-1",
                                    "invoice_amount": 100, "currency": "GBP"}],
                      "purchase_orders": [], "quotes": []},
        "discrepancies": [], "actions": [], "sources": {},
    }
    prompt = sa._build_persona_prompt("You are an auditor.", deal_ctx)
    assert "Acme Deal" in prompt
    # _doc_facts carries type/supplier/currency/total, not the document number
    assert '"currency": "GBP"' in prompt
    assert '"total_amount": 100' in prompt


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


def test_generate_summary_portfolio(monkeypatch):
    monkeypatch.setattr(sa, "ollama_generate", lambda *a, **k: "PORTFOLIO SUMMARY")
    monkeypatch.setattr(sa, "resolve_persona", lambda persona, conn: ("FRAME", "bp_prompt"))
    monkeypatch.setattr(sa, "gather_portfolio_context", lambda conn: {"scope": "portfolio", "sources": {"invoices": 2}})
    rec = []
    conn = _FakeConn({}, recorder=rec)
    out = sa.generate_summary("analysis", deal_id=None, conn=conn)
    assert out["summary"] == "PORTFOLIO SUMMARY"
    assert out["scope"] == "portfolio"
    assert out["deal_id"] is None
    assert "INSERT INTO proc.bp_summary" in " ".join(s for s, _ in rec)


def test_generate_summary_deal(monkeypatch):
    monkeypatch.setattr(sa, "ollama_generate", lambda *a, **k: "DEAL SUMMARY")
    monkeypatch.setattr(sa, "resolve_persona", lambda persona, conn: ("FRAME", "raw"))
    monkeypatch.setattr(sa, "gather_deal_context", lambda deal_id, conn=None: {"deal_id": deal_id, "sources": {"invoices": 1}})
    conn = _FakeConn({})
    out = sa.generate_summary("compliance", deal_id="D-9", conn=conn)
    assert out["scope"] == "deal"
    assert out["deal_id"] == "D-9"
    assert out["persona_source"] == "raw"


def test_generate_summary_no_data_returns_none(monkeypatch):
    monkeypatch.setattr(sa, "resolve_persona", lambda persona, conn: ("FRAME", "bp_prompt"))
    monkeypatch.setattr(sa, "gather_portfolio_context", lambda conn: None)
    assert sa.generate_summary("analysis", conn=_FakeConn({})) is None


def test_generate_summary_empty_llm_raises(monkeypatch):
    monkeypatch.setattr(sa, "ollama_generate", lambda *a, **k: "")
    monkeypatch.setattr(sa, "resolve_persona", lambda persona, conn: ("FRAME", "bp_prompt"))
    monkeypatch.setattr(sa, "gather_deal_context", lambda deal_id, conn=None: {"deal_id": deal_id, "sources": {}})
    import pytest
    with pytest.raises(sa.SummarizationError):
        sa.generate_summary("analysis", deal_id="D-9", conn=_FakeConn({}))


def test_generate_summary_as_of_uses_snapshot(monkeypatch):
    monkeypatch.setattr(sa, "ollama_generate", lambda *a, **k: "HISTORICAL")
    monkeypatch.setattr(sa, "resolve_persona", lambda persona, conn: ("FRAME", "bp_prompt"))
    conn = _FakeConn({
        "SELECT data_snapshot FROM proc.bp_summary": (["data_snapshot"], [({"scope": "deal", "old": True},)]),
    })
    out = sa.generate_summary("analysis", deal_id="D-9", as_of="2026-05-01T00:00:00Z", conn=conn)
    assert out["summary"] == "HISTORICAL"


def test_generate_summary_as_of_missing_snapshot_raises(monkeypatch):
    monkeypatch.setattr(sa, "resolve_persona", lambda persona, conn: ("FRAME", "bp_prompt"))
    conn = _FakeConn({"SELECT data_snapshot FROM proc.bp_summary": (["data_snapshot"], [])})
    import pytest
    with pytest.raises(sa.SnapshotNotFound):
        sa.generate_summary("analysis", deal_id="D-9", as_of="2020-01-01T00:00:00Z", conn=conn)


def test_precompute_iterates_personas_and_scopes(monkeypatch):
    calls = []
    def fake_generate(persona, deal_id=None, as_of=None, conn=None):
        calls.append((persona, deal_id))
        return {"summary_id": "x"}
    monkeypatch.setattr(sa, "generate_summary", fake_generate)
    conn = _FakeConn({
        "prompt_type = 'summary_persona'": (["prompt_name"], [("analysis",), ("compliance",)]),
        "SELECT DISTINCT deal_id": (["deal_id"], [("D-1",), ("D-2",)]),
    })
    out = sa.precompute_summaries(conn=conn)
    assert out["generated"] == 6
    assert ("analysis", None) in calls
    assert ("compliance", "D-2") in calls


def test_precompute_continues_past_failures(monkeypatch):
    def fake_generate(persona, deal_id=None, as_of=None, conn=None):
        if deal_id == "D-1":
            raise RuntimeError("boom")
        return {"summary_id": "x"}
    monkeypatch.setattr(sa, "generate_summary", fake_generate)
    conn = _FakeConn({
        "prompt_type = 'summary_persona'": (["prompt_name"], [("analysis",)]),
        "SELECT DISTINCT deal_id": (["deal_id"], [("D-1",)]),
    })
    out = sa.precompute_summaries(conn=conn)
    assert out["failed"] == 1
    assert out["generated"] == 1


def test_get_cached_summary_returns_current_row():
    conn = _FakeConn({
        "FROM proc.bp_summary": (
            ["summary_id", "persona", "persona_source", "scope", "deal_id", "summary", "sources", "generated_at"],
            [("sid-1", "compliance", "bp_prompt", "deal", "D-9", "cached text", {"invoices": 1}, "2026-06-08T00:00:00Z")],
        ),
    })
    out = sa.get_cached_summary("compliance", "D-9", conn=conn)
    assert out["summary_id"] == "sid-1"
    assert out["summary"] == "cached text"


def test_get_cached_summary_none_when_absent():
    conn = _FakeConn({"FROM proc.bp_summary": (["summary_id"], [])})
    assert sa.get_cached_summary("compliance", "D-9", conn=conn) is None


# --- precompute must never be able to occupy the model indefinitely -------------------
# A daily job that plans 3 personas x 5038 deals = 15114 sequential LLM generations does
# not finish: observed live, it ran 6 hours without completing while every interactive
# request queued behind it on the single Ollama model. These lock the three bounds.

def test_precompute_caps_the_number_of_deals(monkeypatch):
    calls = []
    monkeypatch.setattr(sa, "generate_summary",
                        lambda persona, deal_id=None, as_of=None, conn=None: calls.append((persona, deal_id)) or {"summary_id": "x"})
    conn = _FakeConn({
        "prompt_type = 'summary_persona'": (["prompt_name"], [("analysis",)]),
        "FROM proc.bp_invoice_trgt": (["deal_id"], [(f"D-{i}",) for i in range(100)]),
        "SELECT DISTINCT deal_id": (["deal_id"], [(f"D-{i}",) for i in range(100)]),
    })
    out = sa.precompute_summaries(conn=conn, max_deals=5)
    # portfolio + 5 deals, not portfolio + 100
    assert out["scopes"] == 6, out
    assert len(calls) == 6
    assert out["skipped_deals"] == 95


def test_precompute_stops_at_its_time_budget(monkeypatch):
    clock = {"t": 0.0}
    monkeypatch.setattr(sa.time, "monotonic", lambda: clock["t"])
    def slow(persona, deal_id=None, as_of=None, conn=None):
        clock["t"] += 10.0
        return {"summary_id": "x"}
    monkeypatch.setattr(sa, "generate_summary", slow)
    conn = _FakeConn({
        "prompt_type = 'summary_persona'": (["prompt_name"], [("analysis",)]),
        "FROM proc.bp_invoice_trgt": (["deal_id"], [(f"D-{i}",) for i in range(50)]),
        "SELECT DISTINCT deal_id": (["deal_id"], [(f"D-{i}",) for i in range(50)]),
    })
    out = sa.precompute_summaries(conn=conn, budget_seconds=25)
    assert out["generated"] < 51
    assert out["stopped_reason"] == "budget"


def test_precompute_aborts_when_the_model_keeps_failing(monkeypatch):
    def always_fails(persona, deal_id=None, as_of=None, conn=None):
        raise RuntimeError("Ollama read timeout")
    monkeypatch.setattr(sa, "generate_summary", always_fails)
    conn = _FakeConn({
        "prompt_type = 'summary_persona'": (["prompt_name"], [("analysis",)]),
        "FROM proc.bp_invoice_trgt": (["deal_id"], [(f"D-{i}",) for i in range(500)]),
        "SELECT DISTINCT deal_id": (["deal_id"], [(f"D-{i}",) for i in range(500)]),
    })
    out = sa.precompute_summaries(conn=conn, max_consecutive_failures=3)
    # a model that is timing out will not recover by being asked 500 more times
    assert out["failed"] == 3
    assert out["stopped_reason"] == "failures"


def test_explicit_deal_ids_are_not_silently_truncated(monkeypatch):
    calls = []
    monkeypatch.setattr(sa, "generate_summary",
                        lambda persona, deal_id=None, as_of=None, conn=None: calls.append(deal_id) or {"summary_id": "x"})
    conn = _FakeConn({"prompt_type = 'summary_persona'": (["prompt_name"], [("analysis",)])})
    out = sa.precompute_summaries(conn=conn, deal_ids=["D-1", "D-2", "D-3"], max_deals=1)
    assert out["skipped_deals"] == 0
    assert calls == [None, "D-1", "D-2", "D-3"]
