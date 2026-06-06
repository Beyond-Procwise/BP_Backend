import src.services.deal_summary as ds


class _FakeCursor:
    """Returns canned (description, rows) based on a substring of the SQL."""

    def __init__(self, table_data):
        # table_data: {sql_substring: (columns, rows)}
        self._table_data = table_data
        self.description = []
        self._rows = []

    def execute(self, sql, params=()):
        for needle, (cols, rows) in self._table_data.items():
            if needle in sql:
                self.description = [(c,) for c in cols]
                self._rows = list(rows)
                return
        self.description = []
        self._rows = []

    def fetchall(self):
        return self._rows

    def close(self):
        pass


class _FakeConn:
    def __init__(self, table_data):
        self._cur = _FakeCursor(table_data)

    def cursor(self):
        return self._cur


def _conn_for_deal_with_one_invoice():
    return _FakeConn({
        "bp_invoice_trgt": (
            ["invoice_id", "supplier_id", "invoice_amount", "deal_id", "deal_name", "document_id"],
            [("INV-1", "ACME", 100, "D-9", "Acme Deal", "DOC-1")],
        ),
        "bp_invoice_line_items_trgt": (
            ["invoice_id", "line_number", "item_description", "quantity"],
            [("INV-1", 1, "Widget", 3)],
        ),
        "bp_purchase_order_trgt": (["po_id", "deal_id"], []),
        "bp_quote_trgt": (["quote_id", "deal_id"], []),
        "agent_actions": (
            ["phase", "action_type", "summary"],
            [("extraction", "persist", "persisted raw_id=5")],
        ),
        "bp_extraction_discrepancy": (["field_name", "issue_type"], []),
    })


def test_gather_deal_context_assembles_documents_and_trail():
    ctx = ds.gather_deal_context("D-9", conn=_conn_for_deal_with_one_invoice())
    assert ctx is not None
    assert ctx["deal_id"] == "D-9"
    assert ctx["deal_name"] == "Acme Deal"
    invoices = ctx["documents"]["invoices"]
    assert len(invoices) == 1
    assert invoices[0]["invoice_id"] == "INV-1"
    assert invoices[0]["line_items"][0]["item_description"] == "Widget"
    assert ctx["actions"][0]["action_type"] == "persist"
    assert ctx["sources"]["invoices"] == 1
    assert ctx["sources"]["actions"] == 1


def test_gather_deal_context_unknown_deal_returns_none():
    empty = _FakeConn({
        "bp_invoice_trgt": (["invoice_id", "deal_id"], []),
        "bp_purchase_order_trgt": (["po_id", "deal_id"], []),
        "bp_quote_trgt": (["quote_id", "deal_id"], []),
        "agent_actions": (["phase"], []),
        "bp_extraction_discrepancy": (["field_name"], []),
    })
    assert ds.gather_deal_context("NOPE", conn=empty) is None


def test_build_prompt_is_grounded_and_factual():
    ctx = ds.gather_deal_context("D-9", conn=_conn_for_deal_with_one_invoice())
    prompt = ds._build_prompt(ctx)
    low = prompt.lower()
    assert "do not fabricate" in low or "only the data" in low
    assert "INV-1" in prompt        # facts are present in the prompt
    assert "Acme Deal" in prompt


def test_summarize_deal_returns_text_and_sources(monkeypatch):
    monkeypatch.setattr(
        ds, "gather_deal_context",
        lambda deal_id, conn=None: {
            "deal_id": deal_id, "deal_name": "Acme Deal",
            "documents": {"invoices": [], "purchase_orders": [], "quotes": []},
            "actions": [], "discrepancies": [],
            "sources": {"invoices": 1, "purchase_orders": 0, "quotes": 0,
                        "actions": 2, "discrepancies": 0},
        },
    )
    monkeypatch.setattr(ds, "ollama_cloud_generate", lambda *a, **k: "Acme Deal: one invoice for ACME.")
    out = ds.summarize_deal("D-9")
    assert out["deal_id"] == "D-9"
    assert "Acme Deal" in out["summary"]
    assert out["sources"]["invoices"] == 1


def test_summarize_deal_unknown_returns_none(monkeypatch):
    monkeypatch.setattr(ds, "gather_deal_context", lambda deal_id, conn=None: None)
    assert ds.summarize_deal("NOPE") is None


def test_summarize_deal_raises_on_empty_llm(monkeypatch):
    monkeypatch.setattr(
        ds, "gather_deal_context",
        lambda deal_id, conn=None: {
            "deal_id": deal_id, "deal_name": None,
            "documents": {"invoices": [], "purchase_orders": [], "quotes": []},
            "actions": [], "discrepancies": [], "sources": {"invoices": 1},
        },
    )
    monkeypatch.setattr(ds, "ollama_cloud_generate", lambda *a, **k: "")
    import pytest
    with pytest.raises(ds.SummarizationError):
        ds.summarize_deal("D-9")
