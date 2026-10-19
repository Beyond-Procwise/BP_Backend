"""Batch 1 gateway functions, run through the real Gateway against a scripted fake database."""
import datetime as dt
import os
import types

import pytest
import yaml

from src.services.gateway import Backends, CallerContext, build_default_gateway
from src.services.gateway.broker import CONFIG_DIR


class FakeDB:
    """Rules are (substring of the lower-cased SQL, (columns, rows) | Exception). First match wins; unmatched SQL returns no rows."""

    def __init__(self, *rules):
        self.rules, self.executed = list(rules), []

    def __call__(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def cursor(self):
        return _Cur(self)


class _Cur:
    def __init__(self, db):
        self.db, self.description, self._rows = db, [("x",)], []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=()):
        self.db.executed.append((sql, params))
        low = " ".join(sql.lower().split())
        for needle, result in self.db.rules:
            if needle in low:
                if isinstance(result, Exception):
                    raise result
                cols, rows = result
                self.description, self._rows = [(c,) for c in cols], list(rows)
                return
        self.description, self._rows = [("x",)], []

    def fetchall(self):
        return list(self._rows)

    def fetchone(self):
        return self._rows[0] if self._rows else None


def ctx(scope=("B1",), agent="ask_records"):
    return CallerContext(principal_subject="sub-1", role="Buyer", buyer_scope=scope, request_id="r1", agent=agent)


def call(db, name, params, scope=("B1",), search=None):
    g = build_default_gateway(Backends(connect=db, search=search))
    return g, g.call(ctx(scope), name, params)


PO = ("po_id", "supplier_name")


# --- resolve_entity ---------------------------------------------------------------------------------------------------------------

def test_resolve_entity_finds_a_po_by_id_and_scopes_it_to_the_callers_buyers():
    db = FakeDB(("bp_purchase_order_trgt", (PO, [("PO-2025-0257", "Acme")])))
    _, r = call(db, "resolve_entity", {"text": "status of PO-2025-0257?"})
    assert r.ok and r.data == [{"type": "po", "id": "PO-2025-0257", "label": "Acme"}]
    sql, params = db.executed[0]
    assert "buyer_id" in sql.lower() and ["B1"] in [p for p in params if isinstance(p, list)]
    assert r.provenance[0]["store"] == "postgres" and r.provenance[0]["record_ids"] == ["PO-2025-0257"]


def test_an_unrestricted_caller_gets_no_buyer_clause():
    db = FakeDB(("bp_purchase_order_trgt", (PO, [("PO-2025-0257", "Acme")])))
    call(db, "resolve_entity", {"text": "PO-2025-0257"}, scope=None)
    assert "buyer_id" not in db.executed[0][0].lower()


def test_a_caller_with_an_empty_scope_sees_nothing():
    db = FakeDB(("bp_purchase_order_trgt", (PO, [("PO-2025-0257", "Acme")])))
    _, r = call(db, "resolve_entity", {"text": "PO-2025-0257"}, scope=())
    assert r.ok and r.data == []
    assert not any("bp_purchase_order_trgt" in s.lower() for s, _ in db.executed)


def test_resolve_entity_with_two_suppliers_of_the_same_name_asks_for_a_choice():
    db = FakeDB(("bp_supplier", (("supplier_id", "supplier_name"), [("SUP-1", "Acme Ltd"), ("SUP-2", "Acme Ltd")])))
    _, r = call(db, "resolve_entity", {"text": "Acme Ltd", "types": ["supplier"]})
    assert not r.ok and r.error.code == "AMBIGUOUS" and len(r.error.candidates) == 2


def test_resolve_entity_that_matches_nothing_is_a_successful_empty_result():
    _, r = call(FakeDB(), "resolve_entity", {"text": "PO-2025-0999"})
    assert r.ok and r.data == [] and r.provenance


def test_resolve_entity_rejects_an_unknown_entity_type():
    _, r = call(FakeDB(), "resolve_entity", {"text": "x", "types": ["spaceship"]})
    assert r.error.code == "SCHEMA_INVALID"


# --- get_record / get_po_status ---------------------------------------------------------------------------------------------------

def test_get_record_for_a_po_outside_the_callers_scope_looks_like_not_found():
    db = FakeDB(("bp_purchase_order_trgt", (PO, [])))          # the scoped existence check finds nothing
    _, r = call(db, "get_record", {"type": "po", "id": "PO-2025-0257"})
    assert r.ok and r.data == []
    assert len(db.executed) == 1                                # nothing was read after the scope check failed


def test_get_record_returns_the_record_with_its_lines_and_provenance():
    db = FakeDB(("select po_id, supplier_name from", (PO, [("PO-2025-0257", "Acme")])),
                ("from proc.bp_purchase_order_trgt where upper", (("po_id", "po_status"), [("PO-2025-0257", "open")])),
                ("bp_po_line_items_trgt", (("po_id", "line_number"), [("PO-2025-0257", 1)])))
    _, r = call(db, "get_record", {"type": "po", "id": "PO-2025-0257"})
    assert r.ok and r.data[0]["rows"][0]["po_status"] == "open" and r.data[0]["lines"]
    assert r.provenance[0]["record_ids"] == ["PO-2025-0257"]


def test_get_record_database_failure_is_unavailable_never_an_empty_not_found():
    db = FakeDB(("bp_supplier", RuntimeError("connection refused")))
    _, r = call(db, "get_record", {"type": "supplier", "id": "SUP-ACME"}, scope=None)
    assert not r.ok and r.error.code == "UNAVAILABLE" and r.error.retryable


def test_get_po_status_names_a_source_that_failed_instead_of_dropping_it():
    db = FakeDB(("select po_id, supplier_name from", (PO, [("PO000001", "Acme")])),
                ("select po_id, supplier_name, po_status", (("po_id", "po_status"), [("PO000001", "open")])),
                ("bp_goods_receipt_trgt", RuntimeError("boom")))
    _, r = call(db, "get_po_status", {"po_id": "PO000001"})
    states = {s["source"]: s["state"] for s in r.data}
    assert r.ok and states["Purchase order record"] == "ok" and states["Goods receipts"] == "unavailable"


def test_get_po_status_outside_scope_returns_nothing():
    db = FakeDB(("select po_id, supplier_name from", (PO, [])))
    _, r = call(db, "get_po_status", {"po_id": "PO000001"})
    assert r.ok and r.data == []


# --- get_spend_summary ------------------------------------------------------------------------------------------------------------

SPEND_COLS = ("supplier_id", "supplier_name", "currency", "amount", "invoices")


def test_spend_summary_groups_by_supplier_and_currency_bounds_the_list_and_totals_per_currency():
    rows = [(f"S{i}", f"Supplier {i}", "GBP", 100 - i, 1) for i in range(30)] + [("S99", "Dollar Co", "USD", 5, 2)]
    db = FakeDB(("sum(i.invoice_amount)", (SPEND_COLS, rows)))
    _, r = call(db, "get_spend_summary", {"period_start": "2026-04-01", "period_end": "2026-10-18", "limit": 5}, scope=None)
    assert r.ok and len(r.data["suppliers"]) == 5 and r.truncated and r.total_available == 31
    assert r.data["suppliers"][0]["amount"] == "100"
    assert r.data["totals_by_currency"]["GBP"] == str(sum(100 - i for i in range(30))) and r.data["totals_by_currency"]["USD"] == "5"
    assert r.provenance[0]["table"] == "bp_invoice_trgt"


def test_spend_summary_rejects_a_backwards_period():
    _, r = call(FakeDB(), "get_spend_summary", {"period_start": "2026-10-18", "period_end": "2026-04-01"}, scope=None)
    assert r.error.code == "SCHEMA_INVALID"


# --- get_policy / find_applicable_contracts ---------------------------------------------------------------------------------------

def test_get_policy_returns_the_active_row():
    db = FakeDB(("bp_policy", (("policy_id", "policy_name", "policy_details"), [(7, "sales_quote_pricing", {"tax": 20})])))
    _, r = call(db, "get_policy", {"policy_name": "sales_quote_pricing"}, scope=None)
    assert r.ok and r.data[0]["policy_details"] == {"tax": 20} and r.provenance[0]["record_ids"] == ["7"]
    assert "policy_status = 1" in " ".join(db.executed[0][0].split())


def test_get_policy_store_outage_is_unavailable_not_no_policy():
    db = FakeDB(("bp_policy", RuntimeError("down")))
    _, r = call(db, "get_policy", {"policy_name": "x"}, scope=None)
    assert not r.ok and r.error.code == "UNAVAILABLE"


def test_find_applicable_contracts_filters_by_supplier_and_date_and_dedupes_across_tables():
    cols = ("contract_id", "contract_title", "supplier_id")
    db = FakeDB(("from proc.bp_contracts", (cols, [("C1", "MSA", "SUP-1")])),
                ("from proc.bp_contract_master", (cols, [("C1", "MSA", "SUP-1"), ("C2", "SOW", "SUP-1")])))
    _, r = call(db, "find_applicable_contracts", {"supplier_id": "SUP-1", "date": "2026-10-18"}, scope=None)
    assert [c["contract_id"] for c in r.data] == ["C1", "C2"]
    assert all(p[:1] == ("SUP-1",) and dt.date(2026, 10, 18) in p for _, p in db.executed)


# --- search_documents -------------------------------------------------------------------------------------------------------------

def hit(i, text="Payment terms are 30 days net. " * 60):
    return types.SimpleNamespace(id=f"d{i}", score=0.9 - i / 10,
                                 payload={"document_id": f"DOC-{i}", "document_type": "policy", "section": f"s{i}", "content": text})


def test_search_documents_without_a_backend_is_unavailable():
    _, r = call(FakeDB(), "search_documents", {"query": "three way match"}, scope=None)
    assert not r.ok and r.error.code == "UNAVAILABLE"


def test_search_documents_returns_bounded_passages_that_say_their_numbers_are_document_text():
    seen = {}

    def search(query, top_k, filters):
        seen.update(query=query, top_k=top_k, filters=filters)
        return [hit(i) for i in range(3)]
    _, r = call(FakeDB(), "search_documents", {"query": "payment terms", "top_k": 3, "filters": {"document_type": "policy"}},
                scope=None, search=search)
    assert r.ok and len(r.data) == 3 and seen["filters"] == {"document_type": "policy"} and seen["top_k"] == 3
    p = r.data[0]
    assert p["document_id"] == "DOC-0" and p["clause"] == "s0" and len(p["text"]) <= 800 and p["numbers_are_document_text"] is True
    assert r.provenance[0]["store"] == "vector" and r.provenance[0]["record_ids"] == ["DOC-0", "DOC-1", "DOC-2"]


def test_search_documents_refuses_an_unknown_filter_and_an_oversized_top_k():
    s = lambda q, k, f: []
    assert call(FakeDB(), "search_documents", {"query": "x", "filters": {"sql": "1=1"}}, scope=None, search=s)[1].error.code == "SCHEMA_INVALID"
    assert call(FakeDB(), "search_documents", {"query": "x", "top_k": 500}, scope=None, search=s)[1].error.code == "SCHEMA_INVALID"


# --- approvals: nothing structured exists yet -------------------------------------------------------------------------------------

@pytest.mark.parametrize("name,params", [
    ("get_approval_chain", {"doc_type": "po", "amount": 80000, "currency": "GBP", "business_unit": "IT"}),
    ("evaluate_approval_status", {"doc_type": "po", "doc_id": "PO000001"}),
])
def test_approval_functions_are_unsupported_and_logged_until_the_rules_exist_as_data(name, params):
    g, r = call(FakeDB(), name, params, scope=None)
    assert not r.ok and r.error.code == "UNSUPPORTED"
    assert g.audit.rows[-1]["function"] == name and g.audit.rows[-1]["error_code"] == "UNSUPPORTED"


# --- catalogue and config agree ---------------------------------------------------------------------------------------------------

def test_catalogue_and_functions_yaml_declare_exactly_the_same_functions():
    cfg = yaml.safe_load(open(os.path.join(CONFIG_DIR, "functions.yaml")))
    g = build_default_gateway()
    assert set(cfg["functions"]) == set(g.functions) and cfg["version"] >= 1
    assert {"resolve_entity", "get_record", "get_po_status", "get_spend_summary", "search_documents", "get_policy",
            "find_applicable_contracts", "get_approval_chain", "evaluate_approval_status"} <= set(g.functions)     # Batch 1


def test_no_catalogue_function_takes_a_free_form_query_parameter():
    g = build_default_gateway()
    banned = {"sql", "cypher", "where", "table", "query_text_sql", "statement"}
    for spec in g.functions.values():
        assert not (set(spec.params.model_fields) & banned), spec.name
