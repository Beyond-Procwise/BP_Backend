import os
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.base_agent import AgentContext, AgentStatus
from agents.quote_comparison_agent import QuoteComparisonAgent

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("OLLAMA_USE_GPU", "1")
os.environ.setdefault("OLLAMA_NUM_PARALLEL", "4")
os.environ.setdefault("OMP_NUM_THREADS", "8")


class DummyNick:
    def __init__(self):
        self.settings = SimpleNamespace(script_user="tester")
        self.process_routing_service = SimpleNamespace(
            log_process=lambda **_: None,
            log_run_detail=lambda **_: None,
            log_action=lambda **_: None,
            validate_workflow_id=lambda *args, **kwargs: True,
        )
        self.ollama_options = lambda: {}
        self.pandas_connection = None

    def get_db_connection(self):  # pragma: no cover - defensive
        raise AssertionError("Database access should not be required in this test")


def _build_context(quotes_payload, extra_input=None):
    base_input = {
        "quotes": quotes_payload,
        "weights": 1.0,
        "supplier_ids": ["S1", "S2"],
    }
    if extra_input:
        base_input.update(extra_input)
    return AgentContext(
        workflow_id="wf-1",
        agent_id="quote_comparison",
        user_id="tester",
        input_data=base_input,
    )


def test_quote_comparison_prefers_passed_quotes(monkeypatch):
    nick = DummyNick()
    agent = QuoteComparisonAgent(nick)

    def fail_read(*_args, **_kwargs):  # pragma: no cover - should not be invoked
        raise AssertionError("QuoteComparisonAgent should not read from the database")

    monkeypatch.setattr(agent, "_read_table", fail_read)

    quotes_payload = [
        {
            "name": "weighting",
            "total_spend": 1.0,
            "total_cost": 0,
            "quote_file_s3_path": None,
            "tenure": None,
            "volume": None,
        },
        {
            "name": "Supplier A",
            "supplier_id": "S1",
            "total_spend": 100.789,
            "total_cost": 90.1234,
            "volume": 10.456,
            "tenure": "Net 30",
            "quote_file_s3_path": "s3://bucket/s1.pdf",
        },
        {
            "name": "Supplier B",
            "supplier_id": "S2",
            "total_spend": 200.333,
            "total_cost": 180.8765,
            "volume": 20.111,
            "tenure": "Net 45",
            "quote_file_s3_path": "s3://bucket/s2.pdf",
        },
    ]

    context = _build_context(quotes_payload)
    result = agent.run(context)

    assert result.status == AgentStatus.SUCCESS
    comparison = result.data["comparison"]
    assert len(comparison) == 3
    assert comparison[0]["name"] == "weighting"
    assert "weighting_factors" not in comparison[0]
    assert "total_spend_gbp" not in comparison[0]
    suppliers = {row["supplier_id"] for row in comparison if row["name"] != "weighting"}
    assert suppliers == {"S1", "S2"}
    assert comparison[1]["quote_file_s3_path"] == "s3://bucket/s1.pdf"
    assert comparison[1]["currency"] == "GBP"
    assert "unit_price" not in comparison[0]
    assert "unit_price" not in comparison[1]
    assert "unit_price" not in comparison[2]
    assert comparison[1]["total_cost"] == pytest.approx(90.12)
    assert comparison[1]["total_spend"] == pytest.approx(100.79)
    assert "total_spend_gbp" not in comparison[1]
    assert comparison[1]["volume"] == pytest.approx(10.46)
    assert round(comparison[1]["weighting_score"], 2) == comparison[1]["weighting_score"]
    assert comparison[1]["weighting_score"] > comparison[2]["weighting_score"]
    recommended = result.data.get("recommended_quote")
    assert recommended is not None
    assert recommended["supplier_id"] == "S1"
    assert recommended["ticker"] == "RECOMMENDED"
    assert recommended["weighting_score"] == comparison[1]["weighting_score"]
    assert recommended["total_cost_gbp"] == pytest.approx(90.12)


def test_quote_comparison_filters_by_supplier_tokens(monkeypatch):
    nick = DummyNick()
    agent = QuoteComparisonAgent(nick)

    # Avoid database fallbacks for the test scenario
    monkeypatch.setattr(agent, "_read_table", lambda *_args, **_kwargs: pd.DataFrame())

    quotes_payload = [
        {
            "name": "weighting",
            "total_spend": 1.0,
            "total_cost": 0,
            "quote_file_s3_path": None,
            "tenure": None,
            "volume": None,
        },
        {
            "name": "Supplier A",
            "supplier_id": "S1",
            "total_spend": 100,
            "total_cost": 90,
            "volume": 10,
        },
        {
            "name": "Supplier B",
            "supplier_id": None,
            "total_spend": 200,
            "total_cost": 180,
            "volume": 20,
        },
    ]

    context = _build_context(
        quotes_payload,
        extra_input={"supplier_ids": [], "supplier_names": ["Supplier B"]},
    )

    result = agent.run(context)

    assert result.status == AgentStatus.SUCCESS
    comparison = result.data["comparison"]
    assert len(comparison) == 2
    assert comparison[0]["name"] == "weighting"
    assert "weighting_factors" not in comparison[0]
    assert "total_spend_gbp" not in comparison[0]
    assert comparison[1]["name"] == "Supplier B"
    assert comparison[1]["supplier_id"] is None
    assert "total_spend_gbp" not in comparison[1]
    assert all("unit_price" not in row for row in comparison)
    recommended = result.data.get("recommended_quote")
    assert recommended is not None
    assert recommended["name"] == "Supplier B"


def test_quote_comparison_applies_instruction_weights(monkeypatch):
    nick = DummyNick()
    agent = QuoteComparisonAgent(nick)

    monkeypatch.setattr(agent, "_read_table", lambda *_args, **_kwargs: pd.DataFrame())

    quotes_payload = [
        {
            "name": "weighting",
            "total_spend": 1.0,
            "total_cost": 0,
            "quote_file_s3_path": None,
            "tenure": None,
            "volume": None,
        },
        {
            "name": "Supplier A",
            "supplier_id": "S1",
            "total_spend": 100,
            "total_cost": 90,
            "volume": 10,
        },
        {
            "name": "Supplier B",
            "supplier_id": "S2",
            "total_spend": 200,
            "total_cost": 180,
            "volume": 20,
        },
    ]

    prompts = [
        {
            "promptId": 1,
            "prompts_desc": "{\"metric_weights\": {\"total_cost\": 0.2, \"tenure\": 0.3, \"volume\": 0.5}}",
        }
    ]

    context = _build_context(
        quotes_payload,
        extra_input={"prompts": prompts},
    )

    result = agent.run(context)

    weight_entry = result.data["comparison"][0]
    assert weight_entry["total_cost"] == pytest.approx(0.2)
    assert weight_entry["tenure"] == pytest.approx(0.3)
    assert weight_entry["volume"] == pytest.approx(0.5)
    assert "total_spend_gbp" not in weight_entry


def test_quote_comparison_normalises_metric_weights():
    nick = DummyNick()
    agent = QuoteComparisonAgent(nick)

    entries = [
        {
            "total_cost_gbp": 120.0,
            "tenure": 5,
            "volume": None,
        },
        {
            "total_cost_gbp": 90.0,
            "tenure": 7,
            "volume": None,
        },
    ]

    agent._calculate_weighting_scores(entries, agent.DEFAULT_METRIC_WEIGHTS)

    resolved = agent._resolved_metric_weights
    assert set(resolved.keys()) == {"total_cost", "tenure"}
    assert sum(resolved.values()) == pytest.approx(1.0)
    assert all(entry.get("weighting_score", 0.0) > 0 for entry in entries)


def _versions_tables():
    q = lambda qid, sup, amt: {"quote_id": qid, "supplier_id": sup, "total_amount": amt,
                                "total_amount_incl_tax": amt * 1.2, "currency": "GBP",
                                "quote_date": "2024-04-01", "validity_date": "2024-05-01"}
    quotes = pd.DataFrame([
        q("MCG-1", "S1", 2265700), q("MCG-1 (V2)", "S1", 2153090), q("MCG-1 (V3 (BAFO))", "S1", 2074438),
        q("APX-5", "S2", 2667200), q("APX-5 (V2)", "S2", 2447082), q("APX-5 (V3)", "S2", 2269682),
        q("SDP-9", None, 207656), q("SDP-9 (V3)", None, 199806),
    ])
    lines = pd.DataFrame([{"quote_id": qid, "line_total": amt / 2, "quantity": 1}
                          for qid, amt in zip(quotes["quote_id"], quotes["total_amount"]) for _ in (0, 1)])
    tables = {"proc.bp_quote_trgt": quotes, "proc.bp_quote_line_items_trgt": lines,
              "proc.bp_supplier": pd.DataFrame([{"supplier_id": "S1", "supplier_name": "Meridian"},
                                                {"supplier_id": "S2", "supplier_name": "Apex"}])}
    return tables


def test_each_bid_is_compared_once_at_its_latest_version(monkeypatch):
    # V1+V2+V3 used to be summed per supplier, and the header total was repeated on every line
    # before summing, so a supplier's "total cost" was versions x lines x its price.
    agent = QuoteComparisonAgent(DummyNick())
    tables = _versions_tables()
    monkeypatch.setattr(agent, "_read_table", lambda t, *a, **k: tables[t].copy())
    ctx = AgentContext(workflow_id="wf-1", agent_id="quote_comparison", user_id="tester",
                       input_data={"weights": 1.0})
    out = agent.run(ctx)
    rows = [r for r in out.data["comparison"] if str(r.get("name", "")).lower() != "weighting"]
    assert len(rows) == 3                                   # three bids, not eight versions
    # total_cost is the tax-inclusive header total of the LATEST version; total_spend its lines.
    assert sorted(round(r["total_cost"]) for r in rows) == [round(v * 1.2) for v in (199806, 2074438, 2269682)]
    assert sorted(round(r["total_spend"]) for r in rows) == [199806, 2074438, 2269682]
    assert any(r["supplier_id"] == "quote SDP-9" for r in rows)   # unresolved supplier kept
