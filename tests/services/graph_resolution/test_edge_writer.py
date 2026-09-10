import json
import logging
import pytest
from src.services.graph_resolution.edge_writer import (
    DerivedEdge, redact_signals, cypher_for, write_edges, REDACTED_SIGNALS,
)


def _edge(**kw):
    base = dict(
        rel_type="SAME_ENTITY", from_label="Supplier", from_key="supplier_id",
        from_value="SUP-A", to_label="Supplier", to_key="supplier_id",
        to_value="SUP-B", F=94.2, band="auto_link", P_raw=0.97,
        L_evidence=3.4, profile="supplier_identity", profile_version="1.0.0",
        signals=[{"id": "vat", "s": 1.0, "status": "OK"}],
        observations="abc123", resolution="RESOLVED", margin=0.42,
    )
    base.update(kw)
    return DerivedEdge(**base)


def test_bank_signal_value_is_never_written_in_clear():
    sig = [{"id": "bank_account", "s": 1.0, "status": "OK",
            "value": "GB29NWBK60161331926819"}]
    out = redact_signals(sig)
    assert "GB29NWBK60161331926819" not in json.dumps(out)
    assert out[0]["id"] == "bank_account"
    assert out[0]["s"] == 1.0, "the score survives; only the value is redacted"


def test_non_sensitive_signal_values_survive():
    sig = [{"id": "name", "s": 0.8, "status": "OK", "value": "Acme Ltd"}]
    assert redact_signals(sig)[0]["value"] == "Acme Ltd"


def test_bank_account_is_in_the_redaction_list():
    assert "bank_account" in REDACTED_SIGNALS


def test_cypher_merges_rather_than_creates():
    q, _ = cypher_for(_edge())
    assert "MERGE" in q and "CREATE" not in q


def test_cypher_carries_every_required_property():
    _, params = cypher_for(_edge())
    for key in ("F", "band", "P_raw", "L_evidence", "profile",
                "profile_version", "signals", "observations",
                "resolution", "margin"):
        assert key in params["props"], f"{key} missing from edge properties"


def test_signals_are_serialised_as_json_text():
    _, params = cypher_for(_edge())
    assert isinstance(params["props"]["signals"], str)
    assert json.loads(params["props"]["signals"])[0]["id"] == "vat"


def test_capped_profile_cannot_emit_auto_link():
    e = _edge(profile="contract_coverage", band="auto_link", F=97.0)
    with pytest.raises(ValueError, match="capped at review"):
        cypher_for(e)


# --- structural-identifier validation ---------------------------------------

def test_malformed_from_label_is_refused_not_interpolated():
    e = _edge(from_label="Supplier) DETACH DELETE (n")
    with pytest.raises(ValueError, match="invalid Cypher identifier"):
        cypher_for(e)


def test_malformed_rel_type_is_refused_not_interpolated():
    e = _edge(rel_type="SAME_ENTITY]->(x) DETACH DELETE (x")
    with pytest.raises(ValueError, match="invalid Cypher identifier"):
        cypher_for(e)


# --- write_edges: a fake driver, no live Neo4j required ----------------------

class _FakeResult:
    def __init__(self, cnt):
        self._cnt = cnt

    def single(self):
        return None if self._cnt is None else {"cnt": self._cnt}


class _FakeSession:
    def __init__(self, run_fn):
        self._run_fn = run_fn

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def run(self, query, **params):
        return self._run_fn(query, params)


class _FakeDriver:
    """Stub driver: no network, no neo4j package required."""

    def __init__(self, run_fn):
        self._run_fn = run_fn

    def session(self):
        return _FakeSession(self._run_fn)


def test_write_edges_writes_valid_edges_and_returns_count():
    driver = _FakeDriver(lambda q, p: _FakeResult(1))
    edges = [_edge(), _edge(from_value="SUP-C", to_value="SUP-D")]
    assert write_edges(driver, edges) == 2


def test_guard_violation_in_middle_of_batch_does_not_abort_remaining_edges(caplog):
    calls = []

    def run_fn(q, p):
        calls.append(p)
        return _FakeResult(1)

    driver = _FakeDriver(run_fn)
    edges = [
        _edge(from_value="SUP-1", to_value="SUP-2"),
        _edge(profile="contract_coverage", band="auto_link", F=97.0,
              from_value="SUP-3", to_value="SUP-4"),
        _edge(from_value="SUP-5", to_value="SUP-6"),
    ]
    with caplog.at_level(logging.ERROR, logger="src.services.graph_resolution.edge_writer"):
        written = write_edges(driver, edges)

    # only the two valid edges were sent to the driver and counted
    assert written == 2
    assert len(calls) == 2
    assert {c["from_value"] for c in calls} == {"SUP-1", "SUP-5"}
    # the refusal was logged as a refusal, at ERROR
    refusal_records = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(refusal_records) == 1
    assert "refusing" in refusal_records[0].message.lower()


def test_transport_failure_is_logged_distinctly_from_a_guard_refusal(caplog):
    def run_fn(q, p):
        raise ConnectionError("neo4j unreachable")

    driver = _FakeDriver(run_fn)
    edges = [_edge()]
    with caplog.at_level(logging.DEBUG, logger="src.services.graph_resolution.edge_writer"):
        written = write_edges(driver, edges)

    # never raises to the caller
    assert written == 0
    warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
    error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(warning_records) == 1
    assert "transport failure" in warning_records[0].message.lower()
    assert not error_records, "a transport failure must not be logged as a refusal"
