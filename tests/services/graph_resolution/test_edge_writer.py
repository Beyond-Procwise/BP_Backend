import json
import pytest
from src.services.graph_resolution.edge_writer import (
    DerivedEdge, redact_signals, cypher_for, REDACTED_SIGNALS,
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
