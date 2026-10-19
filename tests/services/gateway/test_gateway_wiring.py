"""ASK_GATEWAY: the identity comes from the session, record lookups go through the gateway, and it fails closed."""
import types

import pytest

from src.services import ask_flags, ask_trace
from src.services.gateway import Backends, CallerContext, build_default_gateway
from src.services.gateway.session import caller_from_principal
from src.services.model_selector import RAGPipeline
from tests.services.gateway.test_gateway_functions import FakeDB

PO_ROW = ("po_id", "supplier_name")


# --- identity from the session ----------------------------------------------------------------------------------------------------

def _principal(sub="sub-9"):
    return types.SimpleNamespace(subject=sub, email="a@b.c", username="u", claims={})


def test_admin_is_unrestricted_and_a_buyer_gets_exactly_their_assigned_buyers(monkeypatch):
    from src.services import rbac
    from src.services.report_data import scope
    monkeypatch.setattr(scope, "_read_buyers", lambda subject: ["B1", "B2"])
    monkeypatch.setattr(rbac, "effective_role", lambda p, *a, **k: "Admin")
    assert caller_from_principal(_principal()).buyer_scope is None
    monkeypatch.setattr(rbac, "effective_role", lambda p, *a, **k: "Buyer")
    c = caller_from_principal(_principal(), request_id="r", session_id="s")
    assert c.buyer_scope == ("B1", "B2") and c.role == "Buyer" and c.principal_subject == "sub-9" and c.request_id == "r"


def test_no_principal_or_an_unreadable_assignment_sees_nothing(monkeypatch):
    from src.services import rbac
    from src.services.report_data import scope
    monkeypatch.setattr(rbac, "effective_role", lambda p, *a, **k: "Buyer")
    assert caller_from_principal(None).buyer_scope == ()
    monkeypatch.setattr(scope, "_read_buyers", lambda s: (_ for _ in ()).throw(RuntimeError("db down")))
    assert caller_from_principal(_principal()).buyer_scope == ()


# --- get_record for a part is scoped through its PO -------------------------------------------------------------------------------

def test_get_record_part_is_joined_to_the_po_and_scoped_to_the_buyer():
    db = FakeDB(("bp_po_line_items_trgt", (("po_id", "item_id", "line_number"), [("PO1", "ITM00001", 1)])))
    g = build_default_gateway(Backends(connect=db))
    r = g.call(CallerContext("s", "Buyer", ("B1",), agent="ask_records"), "get_record", {"type": "part", "id": "ITM00001"})
    assert r.ok and r.data[0]["kind"] == "part" and r.data[0]["lines"][0]["po_id"] == "PO1"
    sql, params = db.executed[0]
    assert "bp_purchase_order_trgt" in sql and "buyer_id" in sql and ["B1"] in [p for p in params if isinstance(p, list)]


def test_get_record_part_for_an_empty_scope_runs_no_query():
    db = FakeDB()
    g = build_default_gateway(Backends(connect=db))
    r = g.call(CallerContext("s", "Buyer", (), agent="ask_records"), "get_record", {"type": "part", "id": "ITM00001"})
    assert r.ok and r.data == [] and db.executed == []


# --- the pipeline stage -----------------------------------------------------------------------------------------------------------

def pipeline(db):
    p = RAGPipeline.__new__(RAGPipeline)
    p.agent_nick = types.SimpleNamespace(get_db_connection=db)
    p._gw = build_default_gateway(Backends(connect=db))
    return p


@pytest.fixture(autouse=True)
def _flags(monkeypatch):
    for f in (ask_flags.EXACT_ID, ask_flags.PO_STATUS, ask_flags.DATE_FACTS):
        monkeypatch.setenv(f, "1")
    ask_trace.start()
    yield
    ask_trace.end()


def test_flag_is_off_by_default_even_in_development(monkeypatch):
    monkeypatch.delenv(ask_flags.GATEWAY, raising=False)
    assert not ask_flags.enabled(ask_flags.GATEWAY)


def test_with_the_flag_on_a_po_lookup_goes_through_the_gateway_and_is_scoped(monkeypatch):
    monkeypatch.setenv(ask_flags.GATEWAY, "1")
    db = FakeDB(("select po_id, supplier_name from", (PO_ROW, [("PO-2025-0257", "Acme")])),
                ("select po_id, supplier_id, supplier_name, po_status", (("po_id", "po_status"), [("PO-2025-0257", "open")])),
                ("bp_po_line_items_trgt", (("po_id", "line_number"), [])))
    p = pipeline(db)
    caller = CallerContext("s", "Buyer", ("B1",), request_id="r")
    text = p._exact_id_facts("What is PO-2025-0257?", caller)
    assert "PO-2025-0257" in text and "po_status=open" in text
    assert all("buyer_id" in s.lower() for s, _ in db.executed if s.lower().startswith("select po_id, supplier_name from"))
    assert [c["function"] for c in ask_trace.current().gateway_calls] == ["get_record"]


def test_with_the_flag_on_a_po_outside_scope_reads_as_not_found(monkeypatch):
    monkeypatch.setenv(ask_flags.GATEWAY, "1")
    p = pipeline(FakeDB(("select po_id, supplier_name from", (PO_ROW, []))))
    text = p._exact_id_facts("What is PO-2025-0257?", CallerContext("s", "Buyer", ("B9",)))
    assert text == ""                                   # nothing found: normal retrieval carries on, nothing leaks


def test_with_the_flag_on_and_no_caller_the_record_lookup_fails_closed(monkeypatch):
    monkeypatch.setenv(ask_flags.GATEWAY, "1")
    db = FakeDB(("bp_purchase_order_trgt", (PO_ROW, [("PO-2025-0257", "Acme")])))
    assert pipeline(db)._exact_id_facts("What is PO-2025-0257?", None) == "" and db.executed == []


def test_a_gateway_outage_is_stated_not_read_as_the_record_not_existing(monkeypatch):
    monkeypatch.setenv(ask_flags.GATEWAY, "1")
    p = pipeline(FakeDB(("bp_purchase_order_trgt", RuntimeError("down"))))
    text = p._exact_id_facts("What is PO-2025-0257?", CallerContext("s", "Admin", None))
    assert "could not be completed" in text and "UNAVAILABLE" in text


def test_status_question_uses_get_po_status_and_names_a_failed_source(monkeypatch):
    monkeypatch.setenv(ask_flags.GATEWAY, "1")
    db = FakeDB(("select po_id, supplier_name from", (PO_ROW, [("PO000001", "Acme")])),
                ("select po_id, supplier_name, po_status", (("po_id", "po_status"), [("PO000001", "open")])),
                ("bp_goods_receipt_trgt", RuntimeError("boom")))
    text = pipeline(db)._exact_id_facts("What is the status of PO000001?", CallerContext("s", "Buyer", ("B1",)))
    assert "po_status=open" in text and "Goods receipts" in text and "could not be read" in text
    assert [c["function"] for c in ask_trace.current().gateway_calls] == ["get_po_status"]


def test_with_the_flag_off_the_legacy_path_is_unchanged_and_never_touches_the_gateway(monkeypatch):
    monkeypatch.setenv(ask_flags.GATEWAY, "0")
    db = FakeDB(("select po_id, supplier_id, supplier_name, po_status", (("po_id", "po_status"), [("PO-2025-0257", "open")])),
                ("bp_po_line_items_trgt", (("po_id", "line_number"), [])))
    text = pipeline(db)._exact_id_facts("What is PO-2025-0257?", None)
    assert "po_status=open" in text and ask_trace.current().gateway_calls == []
