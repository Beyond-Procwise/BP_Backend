"""Batch 3: functions whose data or decision does not exist yet are declared UNSUPPORTED, never improvised."""
import datetime as dt

import pytest

from src.services.gateway import Backends, CallerContext, build_default_gateway, load_agent_registry
from tests.services.gateway.test_gateway_functions import FakeDB

CASES = {
    "get_inventory_position": {"sku": "ITM00001"},
    "get_shipment_status": {"po_id": "PO000001"},
    "get_landed_cost": {"sku": "ITM00001", "supplier_id": "SUP-1", "quantity": 10},
    "get_reseller_programme_terms": {"reseller_id": "R1"},
    "get_partner_tier": {"reseller_id": "R1"},
    "get_customer_pricing_agreement": {"customer_id": "C1", "date": "2026-10-19"},
}


def call(name, params, scope=None):
    db = FakeDB()
    g = build_default_gateway(Backends(connect=db))
    return g, db, g.call(CallerContext("s", "Admin", scope, agent="ask_records"), name, params)


@pytest.mark.parametrize("name", CASES)
def test_blocked_functions_are_unsupported_run_no_query_and_are_logged_to_the_backlog(name):
    g, db, r = call(name, CASES[name])
    assert not r.ok and r.error.code == "UNSUPPORTED" and db.executed == []
    assert g.audit.rows[-1]["function"] == name and g.audit.rows[-1]["error_code"] == "UNSUPPORTED"


@pytest.mark.parametrize("name", CASES)
def test_each_unsupported_reply_names_what_it_is_waiting_for(name):
    _, _, r = call(name, CASES[name])
    assert "waiting on" in r.error.message and len(r.error.message) > 40


@pytest.mark.parametrize("name", CASES)
def test_parameters_are_still_validated_so_the_contract_is_real(name):
    assert call(name, {"bogus": 1})[2].error.code == "SCHEMA_INVALID"


def test_no_agent_is_allowed_to_request_an_unsupported_function():
    g = build_default_gateway()
    reg = load_agent_registry(known_functions=set(g.functions))
    allowed = {fn for a in reg.agents.values() for fn in a.reads + a.writes}
    assert not (allowed & set(CASES))


def test_a_blocked_function_never_takes_caller_scope_from_its_parameters():
    assert call("get_partner_tier", {"reseller_id": "R1", "tenant_id": "x"})[2].error.code == "SCHEMA_INVALID"
