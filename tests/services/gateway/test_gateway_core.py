"""Gateway core: the function standard (specs/2026-10-19-gateway-step2-contracts-and-registries.md, section A and C)."""
import dataclasses

import pytest
from pydantic import BaseModel

from src.services.gateway import (AgentRegistry, AuditSink, CallerContext, DataRequestBroker, FunctionSpec, Gateway,
                                  GatewayFault, Raw, load_agent_registry)


def ctx(**kw):
    base = dict(principal_subject="sub-1", role="Buyer", buyer_scope=("B1",), tenant_id="default", request_id="r1",
                session_id="s1", agent="ask_records")
    base.update(kw)
    return CallerContext(**base)


class Echo(BaseModel):
    model_config = {"extra": "forbid"}
    value: str
    limit: int = 5


def echo_fn(c, p, backends):
    return Raw(data=[{"v": p.value}] * 50, provenance=[{"store": "postgres", "table": "t", "record_ids": ["1"]}])


def gw(fns=None, sink=None):
    fns = fns or {"echo": FunctionSpec(name="echo", params=Echo, fn=echo_fn, max_rows=10)}
    return Gateway(fns, audit=sink or AuditSink(), backends=None)


def test_caller_context_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        ctx().role = "Admin"


def test_unknown_function_is_unsupported_and_audited():
    sink = AuditSink()
    r = gw(sink=sink).call(ctx(), "nope", {})
    assert not r.ok and r.error.code == "UNSUPPORTED"
    assert sink.rows[-1]["function"] == "nope" and sink.rows[-1]["error_code"] == "UNSUPPORTED"


def test_unknown_parameter_is_rejected_not_ignored():
    r = gw().call(ctx(), "echo", {"value": "x", "colour": "red"})
    assert not r.ok and r.error.code == "SCHEMA_INVALID"


@pytest.mark.parametrize("key", ["user_id", "role", "tenant_id", "buyer_scope", "reseller_account_ids", "principal_subject"])
def test_an_agent_cannot_supply_caller_context_in_params(key):
    sink = AuditSink()
    r = gw(sink=sink).call(ctx(), "echo", {"value": "x", key: "admin"})
    assert not r.ok and r.error.code == "SCHEMA_INVALID"
    assert sink.rows[-1]["scope_violation"] is True


def test_rows_are_bounded_and_the_cut_is_reported():
    r = gw().call(ctx(), "echo", {"value": "x"})
    assert r.ok and len(r.data) == 10 and r.truncated and r.total_available == 50


def test_result_without_provenance_is_not_returned():
    fns = {"bare": FunctionSpec(name="bare", params=Echo, fn=lambda c, p, b: Raw(data=[{"n": 1}], provenance=[]), max_rows=10)}
    r = gw(fns).call(ctx(), "bare", {"value": "x"})
    assert not r.ok and r.error.code == "NO_PROVENANCE" and r.data == []


def test_not_found_is_a_successful_empty_result_that_records_what_was_searched():
    fns = {"nf": FunctionSpec(name="nf", params=Echo, max_rows=10, fn=lambda c, p, b: Raw(
        data=[], provenance=[{"store": "postgres", "table": "t", "record_ids": []}]))}
    r = gw(fns).call(ctx(), "nf", {"value": "x"})
    assert r.ok and r.data == [] and r.provenance


def test_function_exception_becomes_unavailable_not_a_crash():
    def boom(c, p, b):
        raise RuntimeError("db down")
    r = gw({"boom": FunctionSpec(name="boom", params=Echo, fn=boom, max_rows=10)}).call(ctx(), "boom", {"value": "x"})
    assert not r.ok and r.error.code == "UNAVAILABLE" and r.error.retryable


def test_function_can_raise_a_structured_fault():
    def amb(c, p, b):
        raise GatewayFault("AMBIGUOUS", "two matches", candidates=["A", "B"])
    r = gw({"amb": FunctionSpec(name="amb", params=Echo, fn=amb, max_rows=10)}).call(ctx(), "amb", {"value": "x"})
    assert r.error.code == "AMBIGUOUS" and r.error.candidates == ["A", "B"]


def test_every_call_writes_one_audit_row_with_the_required_fields():
    sink = AuditSink()
    r = gw(sink=sink).call(ctx(), "echo", {"value": "x"})
    row = sink.rows[-1]
    for k in ("call_id", "ts", "caller_subject", "role", "tenant_id", "agent", "request_id", "function", "params",
              "result_record_ids", "row_count", "truncated", "ok", "error_code", "latency_ms"):
        assert k in row, k
    assert row["call_id"] == r.call_id and row["caller_subject"] == "sub-1" and row["agent"] == "ask_records"
    assert len(sink.rows) == 1


def test_result_carries_as_of_provenance_and_call_id():
    r = gw().call(ctx(), "echo", {"value": "x"})
    assert r.as_of and r.provenance[0]["store"] == "postgres" and r.call_id and r.function == "echo"


# --- data-request contract -------------------------------------------------------------------------------------------------------

REGISTRY = {"version": 1, "defaults": {"max_data_requests_per_step": 2, "max_steps": 4, "max_retries": 1},
            "agents": {"ask_records": {"reads": ["echo"], "writes": ["wr"]}, "ask_policy": {"reads": [], "writes": []}}}


def broker(sink=None):
    g = gw({"echo": FunctionSpec(name="echo", params=Echo, fn=echo_fn, max_rows=10),
            "wr": FunctionSpec(name="wr", params=Echo, fn=echo_fn, max_rows=10, kind="write")}, sink)
    return DataRequestBroker(g, AgentRegistry(REGISTRY, known_functions={"echo", "wr"}))


def req(**kw):
    base = {"request_id": "q1", "agent": "ask_records", "function": "echo", "params": {"value": "x"}, "reason": "need it"}
    base.update(kw)
    return base


def test_function_outside_the_agents_allowlist_is_not_allowed():
    r = broker().handle(ctx(agent="ask_policy"), req(agent="ask_policy"))
    assert r.error.code == "NOT_ALLOWED"


def test_a_write_function_listed_under_writes_is_not_callable_as_a_read_request():
    b = broker()
    assert b.handle(ctx(), req(function="wr")).error.code == "NOT_ALLOWED"


def test_request_naming_a_different_agent_than_the_running_one_is_rejected():
    r = broker().handle(ctx(agent="ask_policy"), req(agent="ask_records"))
    assert r.error.code == "NOT_ALLOWED"


def test_data_requests_per_step_are_capped():
    b, c = broker(), ctx()
    assert b.handle(c, req(request_id="1", params={"value": "a"})).ok
    assert b.handle(c, req(request_id="2", params={"value": "b"})).ok
    assert b.handle(c, req(request_id="3", params={"value": "c"})).error.code == "LIMIT_EXCEEDED"


def test_an_identical_request_in_the_step_is_served_from_cache_without_spending_budget():
    sink = AuditSink()
    b, c = broker(sink), ctx()
    first = b.handle(c, req(request_id="1"))
    again = b.handle(c, req(request_id="2"))
    assert again.ok and again.call_id == first.call_id and len(sink.rows) == 1
    assert b.handle(c, req(request_id="3", params={"value": "other"})).ok       # budget still has room


def test_schema_violation_is_returned_to_the_agent_and_counted_as_a_request():
    b, c = broker(), ctx()
    assert b.handle(c, req(params={"value": "x", "role": "Admin"})).error.code == "SCHEMA_INVALID"


def test_a_new_step_resets_the_budget():
    b, c = broker(), ctx()
    for i, v in enumerate("ab"):
        b.handle(c, req(request_id=str(i), params={"value": v}))
    b.new_step()
    assert b.handle(c, req(request_id="9", params={"value": "z"})).ok


# --- registry loading ------------------------------------------------------------------------------------------------------------

def test_registry_load_fails_when_an_allowlisted_function_does_not_exist():
    bad = {"version": 1, "defaults": {}, "agents": {"a": {"reads": ["ghost"], "writes": []}}}
    with pytest.raises(ValueError, match="ghost"):
        AgentRegistry(bad, known_functions={"echo"})


def test_registry_refuses_the_same_function_as_read_and_write():
    bad = {"version": 1, "defaults": {}, "agents": {"a": {"reads": ["echo"], "writes": ["echo"]}}}
    with pytest.raises(ValueError, match="both"):
        AgentRegistry(bad, known_functions={"echo"})


def test_shipped_agent_registry_validates_against_the_shipped_catalogue():
    from src.services.gateway import build_default_gateway
    g = build_default_gateway()
    reg = load_agent_registry(known_functions=set(g.functions))
    assert reg.version >= 1 and "ask_records" in reg.agents
