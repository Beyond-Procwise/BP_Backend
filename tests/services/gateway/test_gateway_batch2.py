"""Batch 2: the corpus_facts intents as named, row-scoped gateway functions, and the flagged /ask path that uses them."""
import types

import pytest

from src.services import ask_flags, ask_trace, corpus_facts
from src.services.gateway import Backends, CallerContext, build_default_gateway
from src.services.gateway.corpus import INTENT_FUNCTIONS, facts_via_gateway
from src.services.model_selector import RAGPipeline
from tests.services.gateway.test_gateway_functions import FakeDB

BATCH2 = {
    "get_supplier_overview": {}, "get_invoice_overview": {}, "get_quote_overview": {}, "get_purchase_order_overview": {},
    "list_deals": {}, "get_findings_overview": {}, "list_policies": {},
}
SCOPED = [n for n in BATCH2 if n != "list_policies"]       # policies are the organisation's own rule set, not buyer data


def C(scope):
    return CallerContext("sub", "Buyer", scope, request_id="r", agent="ask_analytics")


def run(db, name, params=None, scope=("B1",)):
    g = build_default_gateway(Backends(connect=db))
    return g, g.call(C(scope), name, params or {})


@pytest.mark.parametrize("name", SCOPED)
def test_a_restricted_caller_is_filtered_by_buyer_in_every_query(name):
    db = FakeDB()
    _, r = run(db, name)
    assert r.ok and db.executed
    for sql, params in db.executed:
        assert "buyer_id" in sql.lower(), (name, sql[:80])
        assert ["B1"] in [p for p in params if isinstance(p, list)], (name, sql[:80])


@pytest.mark.parametrize("name", SCOPED)
def test_an_unrestricted_caller_gets_no_buyer_clause(name):
    db = FakeDB()
    run(db, name, scope=None)
    assert db.executed and not any("buyer_id" in s.lower() for s, _ in db.executed)


@pytest.mark.parametrize("name", SCOPED)
def test_an_empty_scope_sees_nothing_and_runs_no_query(name):
    db = FakeDB()
    _, r = run(db, name, scope=())
    assert r.ok and db.executed == []
    assert all(v == [] for v in r.data.values())


@pytest.mark.parametrize("name", list(BATCH2))
def test_a_database_failure_is_unavailable_never_an_empty_answer(name):
    db = FakeDB(("select", RuntimeError("down")))
    _, r = run(db, name, scope=None)
    assert not r.ok and r.error.code == "UNAVAILABLE"


@pytest.mark.parametrize("name", list(BATCH2))
def test_limit_is_bounded_by_config(name):
    _, r = run(FakeDB(), name, {"limit": 500}, scope=None)
    assert r.error.code == "SCHEMA_INVALID"


@pytest.mark.parametrize("name", list(BATCH2))
def test_every_result_names_its_source(name):
    _, r = run(FakeDB(), name, scope=None)
    assert r.provenance and all(p["store"] == "postgres" for p in r.provenance)


def test_supplier_overview_withholds_the_whole_population_count_from_a_restricted_caller():
    db = FakeDB(("suppliers_we_have_invoices_from", (("suppliers_we_have_invoices_from",), [(3,)])))
    _, r = run(db, "get_supplier_overview")
    assert "suppliers_on_record" not in " ".join(s for s, _ in db.executed)
    assert r.data["totals"] == [{"suppliers_we_have_invoices_from": 3}]
    db2 = FakeDB()
    run(db2, "get_supplier_overview", scope=None)
    assert "suppliers_on_record" in " ".join(s for s, _ in db2.executed)


def test_findings_for_a_restricted_caller_go_through_the_documents_deal_without_duplicating_rows():
    db = FakeDB()
    run(db, "get_findings_overview")
    sql = " ".join(db.executed[0][0].lower().split())
    assert "exists" in sql and "bp_deal_documents" in sql and "bp_deal_overview" in sql and "left join" not in sql


def test_findings_for_an_unrestricted_caller_use_the_plain_open_findings_query():
    db = FakeDB()
    run(db, "get_findings_overview", scope=None)
    assert all("bp_deal_documents" not in s.lower() for s, _ in db.executed)


def test_quote_totals_count_the_latest_version_of_each_quote():
    db = FakeDB()
    run(db, "get_quote_overview", scope=None)
    assert "not exists" in db.executed[0][0].lower()


def test_list_deals_reads_overview_and_documents_each_scoped():
    db = FakeDB()
    _, r = run(db, "list_deals")
    assert set(r.data) == {"deals", "documents_in_deals"} and len(db.executed) == 2


def test_list_policies_returns_policies_in_force():
    db = FakeDB(("bp_policy", (("policy_name", "policy_type", "policy_desc", "version"), [("P", "limit", "d", 1)])))
    _, r = run(db, "list_policies", scope=None)
    assert r.data["policies_in_force"][0]["policy_name"] == "P"
    assert "policy_status" in db.executed[0][0].lower()


# --- spend summary is now scoped (Batch 1 gap) --------------------------------------------------------------------------------------

def test_spend_summary_filters_a_restricted_caller_by_buyer():
    db = FakeDB()
    run(db, "get_spend_summary", {"period_start": "2026-01-01", "period_end": "2026-02-01"})
    sql, params = db.executed[0]
    assert "buyer_id" in sql.lower() and ["B1"] in [p for p in params if isinstance(p, list)]


def test_spend_summary_for_an_empty_scope_runs_no_query():
    db = FakeDB()
    _, r = run(db, "get_spend_summary", {"period_start": "2026-01-01", "period_end": "2026-02-01"}, scope=())
    assert r.ok and db.executed == [] and r.data["suppliers"] == []


def test_spend_summary_for_an_unrestricted_caller_is_unchanged():
    db = FakeDB()
    run(db, "get_spend_summary", {"period_start": "2026-01-01", "period_end": "2026-02-01"}, scope=None)
    assert "buyer_id" not in db.executed[0][0].lower()


# --- corpus facts via the gateway --------------------------------------------------------------------------------------------------

def test_every_corpus_intent_maps_to_a_catalogue_function():
    g = build_default_gateway()
    assert set(INTENT_FUNCTIONS) == {name for name, _ in corpus_facts._INTENTS}
    assert all(fn in g.functions for fn in INTENT_FUNCTIONS.values())


@pytest.mark.parametrize("intent", [n for n, _ in corpus_facts._INTENTS if n != "spend"])
def test_gateway_facts_have_the_same_keys_as_the_legacy_facts(intent):
    class Cur:                                   # legacy path reads the same shape of fake data
        description = [("x",)]
        def execute(self, *a): pass
        def fetchall(self): return [(1,)]
    legacy_keys = set(corpus_facts._fetch(Cur(), intent))
    g = build_default_gateway(Backends(connect=FakeDB()))
    facts = facts_via_gateway(g, C(None), intent)
    assert set(facts) == legacy_keys


def test_spend_intent_returns_per_supplier_and_per_currency_facts():
    rows = [("S1", "Acme", "GBP", 100, 2), ("S1", "Acme", "USD", 5, 1)]
    db = FakeDB(("sum(i.invoice_amount)", (("supplier_id", "supplier_name", "currency", "amount", "invoices"), rows)))
    facts = facts_via_gateway(build_default_gateway(Backends(connect=db)), C(None), "spend")
    assert [r["currency"] for r in facts["invoiced_spend_by_supplier"]] == ["GBP", "USD"]
    assert {r["currency"] for r in facts["totals_by_currency"]} == {"GBP", "USD"}


def test_a_failed_lookup_is_stated_so_the_model_does_not_invent_totals():
    g = build_default_gateway(Backends(connect=FakeDB(("select", RuntimeError("down")))))
    facts = facts_via_gateway(g, C(None), "invoices")
    assert "could not be read" in facts["corpus_lookup_unavailable"][0]["note"]


def test_no_intent_means_no_gateway_call():
    g = build_default_gateway(Backends(connect=FakeDB()))
    assert facts_via_gateway(g, C(None), None) == {} and g.audit.rows == []


# --- the pipeline stage ------------------------------------------------------------------------------------------------------------

def pipeline(db):
    p = RAGPipeline.__new__(RAGPipeline)
    p.agent_nick = types.SimpleNamespace(get_db_connection=db)
    p._gw = build_default_gateway(Backends(connect=db))
    return p


@pytest.fixture
def traced():
    ask_trace.start()
    yield
    ask_trace.end()


def test_with_the_flag_on_corpus_facts_are_read_through_the_gateway_as_the_caller(monkeypatch, traced):
    monkeypatch.setenv(ask_flags.GATEWAY, "1")
    monkeypatch.setattr(corpus_facts, "fetch_graph", lambda q, limit=6: [])
    db = FakeDB(("invoices_total", (("invoices_total", "citing_a_po"), [(7, 5)])))
    facts = pipeline(db)._corpus_facts("how many invoices do we have", CallerContext("s", "Buyer", ("B1",)))
    assert facts["intent"] == "invoices" and facts["totals"] == [{"invoices_total": 7, "citing_a_po": 5}]
    assert all("buyer_id" in s.lower() for s, _ in db.executed)
    assert [c["function"] for c in ask_trace.current().gateway_calls] == ["get_invoice_overview"]


def test_with_the_flag_on_and_no_caller_no_corpus_rows_are_read(monkeypatch, traced):
    monkeypatch.setenv(ask_flags.GATEWAY, "1")
    monkeypatch.setattr(corpus_facts, "fetch_graph", lambda q, limit=6: [])
    db = FakeDB(("invoices_total", (("invoices_total",), [(7,)])))
    assert pipeline(db)._corpus_facts("how many invoices do we have", None) is None and db.executed == []


def test_with_the_flag_off_the_legacy_function_is_used(monkeypatch, traced):
    monkeypatch.setenv(ask_flags.GATEWAY, "0")
    seen = {}
    monkeypatch.setattr(corpus_facts, "fetch_facts", lambda nick, q: seen.setdefault("legacy", {"intent": "x", "a": [1]}))
    assert pipeline(FakeDB())._corpus_facts("anything", None)["intent"] == "x" and "legacy" in seen
