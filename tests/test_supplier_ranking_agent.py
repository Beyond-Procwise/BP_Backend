import json
import os
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.supplier_ranking_agent import SupplierRankingAgent, ensure_payment_terms_score, _json_safe
from agents.base_agent import AgentContext, AgentOutput, AgentStatus
from engines.policy_engine import PolicyEngine
from orchestration.orchestrator import Orchestrator


def _supplier_policy_rows():
    weight_details = {
        "rules": {
            "default_weights": {
                "price": 0.4,
                "delivery": 0.3,
                "risk": 0.2,
                "payment_terms": 0.1,
            }
        }
    }
    categorical_details = {
        "rules": {
            "payment_terms": {
                "Net 30": 10,
                "Net 45": 8,
                "Net 60": 6,
                "default": 5,
            }
        }
    }
    normalization_details = {
        "rules": {
            "price": "lower_is_better",
            "delivery": "higher_is_better",
            "risk": "lower_is_better",
            "payment_terms": "higher_is_better",
        }
    }
    return [
        {
            "policy_id": 201,
            "policy_name": "WeightAllocationPolicy",
            "policy_type": "supplier_ranking",
            "policy_desc": "Default supplier ranking weights",
            "policy_details": json.dumps(weight_details),
        },
        {
            "policy_id": 202,
            "policy_name": "CategoricalScoringPolicy",
            "policy_type": "supplier_ranking",
            "policy_desc": "Categorical scoring rules",
            "policy_details": json.dumps(categorical_details),
        },
        {
            "policy_id": 203,
            "policy_name": "NormalizationDirectionPolicy",
            "policy_type": "supplier_ranking",
            "policy_desc": "Normalization direction rules",
            "policy_details": json.dumps(normalization_details),
        },
    ]


class DummyNick:
    def __init__(self):
        self.policy_engine = PolicyEngine(policy_rows=_supplier_policy_rows())
        self.settings = SimpleNamespace(
            extraction_model="gpt-oss", script_user="tester"
        )
        # Minimal query engine stub for agent initialisation. The fetch_*
        # methods below must exist (returning legitimately-empty frames)
        # rather than being absent -- an absent method raises AttributeError,
        # which the agent now correctly treats as a load FAILURE rather than
        # "queried and found nothing", and refuses to rank on top of it.
        self.query_engine = SimpleNamespace(
            fetch_supplier_data=lambda *_: [],
            fetch_purchase_order_data=lambda **_: pd.DataFrame(),
            fetch_invoice_data=lambda **_: pd.DataFrame(),
            fetch_procurement_flow=lambda **_: pd.DataFrame(),
        )


def test_top_n_parsed_from_query(monkeypatch):
    nick = DummyNick()
    agent = SupplierRankingAgent(nick)
    monkeypatch.setattr(agent, "_generate_justification", lambda row, criteria: "ok")

    df = pd.DataFrame({
        "supplier_name": [f"S{i}" for i in range(6)],
        "price": [60, 50, 40, 30, 20, 10],
    })

    context = AgentContext(
        workflow_id="wf1",
        agent_id="supplier_ranking",
        user_id="u1",
        input_data={
            "supplier_data": df,
            "intent": {"parameters": {"criteria": ["price"]}},
            "query": "Rank top 5 suppliers by price",
        },
    )

    output = agent.run(context)
    assert len(output.data["ranking"]) == 5
    assert output.data["rank_count"] == 5
    assert output.data["ranking"][0]["rank_position"] == 1
    assert all(entry["rank_count"] == 5 for entry in output.data["ranking"])


def test_execute_ranking_flow_extracts_criteria_from_query():
    """Orchestrator should derive ranking criteria from the free-text query."""

    class DummyPolicyEngine:
        def __init__(self):
            self.supplier_policies = [
                {
                    "policyName": "WeightAllocationPolicy",
                    "details": {"rules": {"default_weights": {"price": 1.0, "delivery": 1.0}}},
                }
            ]
            self.last_input = None

        def validate_workflow(self, workflow_name, user_id, input_data):
            self.last_input = input_data
            return {"allowed": True, "reason": ""}

    class DummyQueryEngine:
        def fetch_supplier_data(self, intent):  # pragma: no cover - simple stub
            return pd.DataFrame([{"supplier_name": "S1", "price": 10, "delivery": 8}])

    class DummyRankingAgent:
        def execute(self, context):  # pragma: no cover - simple stub
            assert context.input_data["intent"]["parameters"]["criteria"] == ["price"]
            return AgentOutput(status=AgentStatus.SUCCESS, data={})

    nick = SimpleNamespace(
        settings=SimpleNamespace(script_user="tester", max_workers=1),
        agents={"supplier_ranking": DummyRankingAgent()},
        policy_engine=DummyPolicyEngine(),
        query_engine=DummyQueryEngine(),
        routing_engine=SimpleNamespace(routing_model={}),
    )

    orchestrator = Orchestrator(nick)
    result = orchestrator.execute_ranking_flow("Rank suppliers by price")

    assert result["status"] == "completed"
    assert nick.policy_engine.last_input["criteria"] == ["price"]


def test_supplier_ranking_does_not_train_query_engine(monkeypatch):
    class Nick(DummyNick):
        def __init__(self):
            super().__init__()
            self.trained = False

            def train():
                self.trained = True

            self.query_engine = SimpleNamespace(
                fetch_supplier_data=lambda *_: pd.DataFrame(
                    {"supplier_name": ["S1"], "price": [1]}
                ),
                train_procurement_context=train,
            )

    nick = Nick()
    agent = SupplierRankingAgent(nick)
    monkeypatch.setattr(agent, "_generate_justification", lambda row, criteria: "ok")

    context = AgentContext(
        workflow_id="wf1",
        agent_id="supplier_ranking",
        user_id="u1",
        input_data={
            "intent": {"parameters": {"criteria": ["price"]}},
            "query": "Rank suppliers by price",
        },
    )

    agent.run(context)

    assert nick.trained is False


def test_supplier_ranking_injects_missing_candidates(monkeypatch):
    class StubPolicyEngine:
        def __init__(self):
            self.supplier_policies = [
                {
                    "policyName": "WeightAllocationPolicy",
                    "details": {"rules": {"default_weights": {"price": 1.0}}},
                },
                {"policyName": "CategoricalScoringPolicy", "details": {"rules": {}}},
                {
                    "policyName": "NormalizationDirectionPolicy",
                    "details": {"rules": {"price": "lower_is_better"}},
                },
            ]

    class StubQueryEngine:
        def fetch_purchase_order_data(
            self,
            intent=None,
            supplier_ids=None,
            supplier_names=None,
        ):
            return pd.DataFrame()

        def fetch_invoice_data(
            self,
            intent=None,
            supplier_ids=None,
            supplier_names=None,
        ):
            return pd.DataFrame()

        def fetch_procurement_flow(
            self,
            embed: bool = False,
            supplier_ids=None,
            supplier_names=None,
        ):
            return pd.DataFrame()

    nick = SimpleNamespace(
        settings=SimpleNamespace(extraction_model="gpt-oss", script_user="tester"),
        policy_engine=StubPolicyEngine(),
        query_engine=StubQueryEngine(),
    )

    agent = SupplierRankingAgent(nick)
    monkeypatch.setattr(agent, "_generate_justification", lambda row, criteria: "ok")
    monkeypatch.setattr(agent, "_read_table", lambda *args, **kwargs: pd.DataFrame())

    supplier_df = pd.DataFrame(
        {
            "supplier_id": ["S1"],
            "supplier_name": ["Alpha"],
            "avg_unit_price": [10.0],
        }
    )

    context = AgentContext(
        workflow_id="wf1",
        agent_id="supplier_ranking",
        user_id="user",
        input_data={
            "supplier_data": supplier_df,
            "supplier_candidates": ["S1", "S2"],
            "supplier_directory": [
                {"supplier_id": "S1", "supplier_name": "Alpha"},
                {"supplier_id": "S2", "supplier_name": "Beta"},
            ],
            "intent": {"parameters": {"criteria": ["price"], "top_n": 2}},
            "query": "rank suppliers",
        },
    )

    output = agent.run(context)

    assert output.status == AgentStatus.SUCCESS
    ranking_ids = {entry["supplier_id"] for entry in output.data["ranking"]}
    assert ranking_ids == {"S1", "S2"}
    assert any(entry["supplier_name"] == "Beta" for entry in output.data["ranking"])


def test_deal_price_scored_against_cheapest_bid():
    """Scores must reflect how close the race actually was.

    Min-max normalisation scored a supplier 1.1% off the best price as 0.00/100 -- true
    ordering, but a slanderous number. Ratio-to-best makes the score mean something
    absolute: 98.9 == "1.1% off the winning bid".
    """
    agent = SupplierRankingAgent.__new__(SupplierRankingAgent)
    df = pd.DataFrame(
        {
            "supplier_id": ["SteelCore", "AFS", "Northgate"],
            "price": [279210.0, 281570.0, 282260.0],
        }
    )
    scored = agent._score_deal_price(df)
    by_id = dict(zip(scored["supplier_id"], scored["price_score"]))

    assert by_id["SteelCore"] == pytest.approx(100.0)
    assert by_id["AFS"] == pytest.approx(99.16, abs=0.01)
    assert by_id["Northgate"] == pytest.approx(98.92, abs=0.01)


def test_lone_bidder_gets_no_price_score():
    """An uncontested quote is no evidence of a good price, so it scores NULL not 100."""
    agent = SupplierRankingAgent.__new__(SupplierRankingAgent)
    scored = agent._score_deal_price(
        pd.DataFrame({"supplier_id": ["Solo"], "price": [1000.0]})
    )
    assert pd.isna(scored.loc[0, "price_score"])


def test_latest_bidding_round_is_the_standing_offer(monkeypatch):
    """Only the newest round is a live offer; superseded ones are history, not rivals."""
    agent = SupplierRankingAgent.__new__(SupplierRankingAgent)
    quotes = pd.DataFrame(
        {
            "quote_id": ["STC-1 (V1)", "STC-1 (V2)", "STC-1 (V3 (BAFO))"],
            "deal_id": ["D1"] * 3,
            "supplier_id": ["SteelCore"] * 3,
            "quote_date": ["2026-07-02", "2026-07-03", "2026-07-04"],
            "total_amount": [292900.0, 287620.0, 279210.0],
            "currency": ["GBP"] * 3,
        }
    )
    monkeypatch.setattr(agent, "_read_table", lambda *a, **k: quotes)

    loaded = agent._load_deal_quotes("D1")

    assert len(loaded) == 1, "three rounds from one supplier are one offer, not three bids"
    row = loaded.iloc[0]
    assert row["final_quote_amount"] == pytest.approx(279210.0)
    assert row["opening_quote_amount"] == pytest.approx(292900.0)
    assert row["quote_rounds"] == 3
    assert row["concession_pct"] == pytest.approx(4.67, abs=0.01)


def test_merge_supplier_metrics_leaves_avg_unit_price_null_with_no_po_lines():
    """N2 root cause: a supplier with ZERO purchase-order lines must not get a
    fabricated avg_unit_price of 0.0.

    Live bug: three suppliers with 0 invoices/0 POs in the DB were persisted
    with avg_unit_price = 0.00, which then normalised to a *perfect*
    price_score of 100.00 -- 0 looked like the cheapest possible bid, and
    beat every supplier with a real, non-zero price. 0.0 asserts "we priced
    this at zero"; NaN says "we have no idea", which is the honest answer
    for a supplier with no PO-line evidence at all.
    """
    nick = DummyNick()
    agent = SupplierRankingAgent(nick)

    supplier_df = pd.DataFrame(
        {"supplier_id": ["REAL", "GHOST"], "supplier_name": ["Real Co", "Ghost Co"]}
    )
    agent._prime_supplier_aliases(supplier_df, [])

    purchase_orders = pd.DataFrame(
        {
            "po_id": ["PO1"],
            "supplier_id": ["REAL"],
            "supplier_name": ["Real Co"],
            "total_amount": [1000.0],
            "payment_terms": ["Net 30"],
            "order_date": ["2026-01-01"],
            "expected_delivery_date": ["2026-01-10"],
        }
    )
    po_lines = pd.DataFrame(
        {
            "po_id": ["PO1"],
            "unit_price": [50.0],
            "quantity": [20.0],
            "line_total": [1000.0],
            "item_description": ["Widget"],
        }
    )
    tables = {
        "purchase_orders": purchase_orders,
        "po_lines": po_lines,
        "invoices": pd.DataFrame(),
        "invoice_lines": pd.DataFrame(),
        "procurement_flow": pd.DataFrame(),
    }

    result = agent._merge_supplier_metrics(supplier_df, tables)
    by_id = result.set_index("supplier_id")

    assert by_id.loc["REAL", "avg_unit_price"] == pytest.approx(50.0)
    assert pd.isna(by_id.loc["GHOST", "avg_unit_price"]), (
        "a supplier with no PO-line evidence must have avg_unit_price = NaN, "
        f"not a fabricated value (got {by_id.loc['GHOST', 'avg_unit_price']!r})"
    )


def test_normalize_numeric_scores_does_not_score_unmeasured_supplier_as_perfect():
    """N2: a tie among the suppliers that DO have data must not spill a 100
    onto a supplier who has NO data at all for that metric.

    Before the fix, ``out[score_col] = 100.0`` was assigned to the WHOLE
    column whenever the measured values tied -- including rows whose raw
    value was NaN. A supplier with no price simply not being included in
    the tie must not make them "tied for best" by accident.
    """
    agent = SupplierRankingAgent.__new__(SupplierRankingAgent)
    df = pd.DataFrame(
        {
            "supplier_id": ["A", "B", "GHOST"],
            "price": [50.0, 50.0, None],
        }
    )

    scored = agent._normalize_numeric_scores(df, {"price": "lower_is_better"})
    by_id = scored.set_index("supplier_id")["price_score"]

    assert by_id["A"] == pytest.approx(100.0)
    assert by_id["B"] == pytest.approx(100.0)
    assert pd.isna(by_id["GHOST"]), (
        "a supplier with no price at all must not inherit the tied-suppliers' "
        f"perfect score (got {by_id['GHOST']!r})"
    )


def test_ghost_suppliers_with_zero_evidence_do_not_outrank_a_real_bidder(monkeypatch):
    """End-to-end reproduction of the live bug: suppliers with 0 invoices and
    0 purchase orders must not receive a fabricated price of 0.0 (which reads
    as "the cheapest bid possible") and rank ABOVE a supplier with a real,
    evidenced price.
    """
    nick = DummyNick()

    real_pos = pd.DataFrame(
        {
            "po_id": ["PO1"],
            "supplier_id": ["SUP-REAL"],
            "supplier_name": ["Real Co"],
            "total_amount": [1000.0],
            "payment_terms": ["Net 30"],
            "order_date": ["2026-01-01"],
            "expected_delivery_date": ["2026-01-10"],
        }
    )
    real_po_lines = pd.DataFrame(
        {
            "po_id": ["PO1"],
            "unit_price": [50.0],
            "quantity": [20.0],
            "line_total": [1000.0],
            "item_description": ["Widget"],
        }
    )

    nick.query_engine = SimpleNamespace(
        fetch_supplier_data=lambda *_: [],
        fetch_purchase_order_data=lambda **_: real_pos.copy(),
        fetch_invoice_data=lambda **_: pd.DataFrame(),
        fetch_procurement_flow=lambda **_: pd.DataFrame(),
    )

    agent = SupplierRankingAgent(nick)
    monkeypatch.setattr(agent, "_generate_justification", lambda row, criteria: "ok")
    monkeypatch.setattr(
        agent,
        "_read_table",
        lambda table, *a, **k: (
            real_po_lines.copy() if "po_line_items" in table else pd.DataFrame()
        ),
    )

    df = pd.DataFrame(
        {
            "supplier_id": ["SUP-REAL", "SUP-GHOST1", "SUP-GHOST2"],
            "supplier_name": ["Real Co", "Ghost One", "Ghost Two"],
        }
    )

    context = AgentContext(
        workflow_id="wf1",
        agent_id="supplier_ranking",
        user_id="user",
        input_data={
            "supplier_data": df,
            "intent": {"parameters": {"criteria": ["price"], "top_n": 3}},
            "query": "rank suppliers",
        },
    )

    output = agent.run(context)

    assert output.status == AgentStatus.SUCCESS, output.error
    by_id = {e["supplier_id"]: e for e in output.data["ranking"]}

    assert "SUP-REAL" in by_id
    assert by_id["SUP-REAL"]["price_score"] == pytest.approx(100.0), (
        "the only supplier with real price evidence must score on its own merits"
    )
    for ghost_id in ("SUP-GHOST1", "SUP-GHOST2"):
        if ghost_id in by_id:
            ghost_score = by_id[ghost_id].get("price_score")
            assert ghost_score is None or pd.isna(ghost_score) or ghost_score < 100.0, (
                f"{ghost_id} has zero evidence and must not out-score the real bidder "
                f"(got price_score={ghost_score!r})"
            )
            assert (by_id[ghost_id].get("final_score") or 0) <= (
                by_id["SUP-REAL"].get("final_score") or 0
            ), f"{ghost_id} (no evidence) must not outrank SUP-REAL (real evidence)"

    # SUP-REAL, the only supplier with actual evidence, must be rank 1.
    assert by_id["SUP-REAL"]["rank_position"] == 1


def test_ranking_of_suppliers_with_no_evidence_at_all_refuses_to_publish(monkeypatch):
    """N2: ranking a set of suppliers with NO evidence anywhere (no POs, no
    invoices, no quotes) must return the honest 'cannot rank' result, not a
    confident 1/2/3 ranking of fabricated perfect scores.

    Deliberately does NOT monkeypatch ``_load_procurement_tables`` /
    ``_merge_supplier_metrics`` -- this must hold through the REAL merge
    code path, which is exactly where the live fabrication bug lived.
    """
    nick = DummyNick()
    agent = SupplierRankingAgent(nick)
    monkeypatch.setattr(agent, "_read_table", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(agent, "_generate_justification", lambda row, criteria: "ok")

    df = pd.DataFrame(
        {
            "supplier_id": ["SUP-MgmSouvenirShop", "SUP-TechtonicElectronicInc", "SUP-XyzLtd"],
            "supplier_name": ["MGM Souvenir Shop", "TechTonic Electronic, Inc.", "xyz ltd"],
        }
    )
    context = AgentContext(
        workflow_id="wf1",
        agent_id="supplier_ranking",
        user_id="user",
        input_data={
            "supplier_data": df,
            "intent": {"parameters": {"criteria": ["price"], "top_n": 3}},
            "query": "rank suppliers",
        },
    )

    output = agent.run(context)

    assert output.status == AgentStatus.FAILED, (
        "suppliers with zero evidence anywhere must not receive a ranking "
        f"(got status={output.status}, data={output.data})"
    )
    assert "insufficient data" in (output.error or "").lower() or "no evidence" in (
        output.error or ""
    ).lower()


def test_ranking_refuses_to_publish_when_nothing_is_measurable(monkeypatch):
    """No data must fail loudly, not emit a confident table of 0.00s.

    A buyer cannot tell "these suppliers are bad" from "we knew nothing about them",
    so a ranking we have no basis for must not be published at all.
    """
    nick = DummyNick()
    agent = SupplierRankingAgent(nick)
    monkeypatch.setattr(agent, "_load_procurement_tables", lambda *_: {})
    monkeypatch.setattr(agent, "_merge_supplier_metrics", lambda df, _tables: df)
    monkeypatch.setattr(agent, "_build_supplier_profiles", lambda _t, ids: {})
    monkeypatch.setattr(agent, "_read_table", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(agent, "_generate_justification", lambda row, criteria: "ok")

    # Names only -- exactly what proc.bp_supplier holds today.
    df = pd.DataFrame(
        {"supplier_id": ["S1", "S2"], "supplier_name": ["Alpha", "Beta"]}
    )
    context = AgentContext(
        workflow_id="wf1",
        agent_id="supplier_ranking",
        user_id="user",
        input_data={
            "supplier_data": df,
            "intent": {"parameters": {"criteria": ["price", "risk"], "top_n": 2}},
            "query": "rank suppliers",
        },
    )

    output = agent.run(context)

    assert output.status == AgentStatus.FAILED
    assert "insufficient data" in (output.error or "").lower()


def test_ranking_does_not_return_clean_success_when_invoice_load_fails(monkeypatch):
    """A load FAILURE (exception) is not the same as legitimately-empty data.

    Before the fix, ``_load_procurement_tables`` swallowed the exception from
    ``fetch_invoice_data``, substituted an empty DataFrame, and the agent kept
    going -- returning AgentStatus.SUCCESS as if the ranking were computed
    from complete evidence. That is a lie: the ranking was computed with a
    material evidence source missing, not with a source that was checked and
    found empty. The agent must not present that as a clean, unqualified
    success.
    """
    nick = DummyNick()

    def _raise_invoice_load(*_args, **_kwargs):
        raise RuntimeError("relation \"proc.bp_invoice_trgt\" boom")

    nick.query_engine = SimpleNamespace(
        fetch_supplier_data=lambda *_: [],
        fetch_purchase_order_data=lambda **_: pd.DataFrame(),
        fetch_invoice_data=_raise_invoice_load,
        fetch_procurement_flow=lambda **_: pd.DataFrame(),
    )

    agent = SupplierRankingAgent(nick)
    monkeypatch.setattr(agent, "_read_table", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(agent, "_generate_justification", lambda row, criteria: "ok")

    df = pd.DataFrame(
        {
            "supplier_id": ["S1", "S2"],
            "supplier_name": ["Alpha", "Beta"],
            "price": [50.0, 40.0],
        }
    )
    context = AgentContext(
        workflow_id="wf1",
        agent_id="supplier_ranking",
        user_id="user",
        input_data={
            "supplier_data": df,
            "intent": {"parameters": {"criteria": ["price"], "top_n": 2}},
            "query": "rank suppliers",
        },
    )

    output = agent.run(context)

    # Whatever representation is chosen, it must not be a clean, silent
    # SUCCESS that hides the fact that invoice evidence failed to load.
    if output.status == AgentStatus.SUCCESS:
        failures = output.data.get("evidence_load_failures") or []
        assert "invoices" in failures, (
            "SUCCESS was returned without naming the invoice load failure "
            "in the output -- this hides the missing evidence."
        )
    else:
        assert output.status == AgentStatus.FAILED
        assert "invoice" in (output.error or "").lower()


def test_supplier_ranking_normalises_weights_to_available_metrics(monkeypatch):
    nick = DummyNick()
    agent = SupplierRankingAgent(nick)
    monkeypatch.setattr(agent, "_load_procurement_tables", lambda *_: {})
    monkeypatch.setattr(agent, "_merge_supplier_metrics", lambda df, _tables: df)
    monkeypatch.setattr(
        agent,
        "_build_supplier_profiles",
        lambda _tables, ids: {str(s): {} for s in ids},
    )
    monkeypatch.setattr(agent, "_generate_justification", lambda row, criteria: "ok")

    df = pd.DataFrame(
        {
            "supplier_id": ["S1", "S2"],
            "supplier_name": ["Alpha", "Beta"],
            "price": [50.0, 40.0],
            "risk": [5.0, 3.0],
        }
    )

    context = AgentContext(
        workflow_id="wf1",
        agent_id="supplier_ranking",
        user_id="user",
        input_data={
            "supplier_data": df,
            "intent": {"parameters": {"criteria": ["price", "delivery", "risk"], "top_n": 2}},
            "query": "rank suppliers",
        },
    )

    output = agent.run(context)

    weights = output.data["ranking"][0]["weights"]
    assert "price" in weights and "risk" in weights
    assert weights["price"] > 0
    assert weights["risk"] > 0
    assert sum(weights.values()) == pytest.approx(1.0)
    assert output.data["ranking"][0]["final_score"] > 0


def test_supplier_ranking_includes_flow_coverage_from_snapshot(monkeypatch):
    nick = DummyNick()
    agent = SupplierRankingAgent(nick)
    monkeypatch.setattr(agent, "_load_procurement_tables", lambda *_: {})
    monkeypatch.setattr(agent, "_merge_supplier_metrics", lambda df, _tables: df)
    monkeypatch.setattr(
        agent,
        "_build_supplier_profiles",
        lambda _tables, ids: {str(s): {} for s in ids},
    )
    monkeypatch.setattr(agent, "_generate_justification", lambda row, criteria: "ok")

    df = pd.DataFrame(
        {
            "supplier_id": ["S1"],
            "supplier_name": ["Alpha"],
            "price": [50.0],
        }
    )

    snapshot = {
        "supplier_flows": [
            {
                "supplier_id": "S1",
                "supplier_name": "Alpha",
                "coverage_ratio": 0.6,
                "purchase_orders": {"count": 2},
            }
        ]
    }

    context = AgentContext(
        workflow_id="wf2",
        agent_id="supplier_ranking",
        user_id="user",
        input_data={
            "supplier_data": df,
            "data_flow_snapshot": snapshot,
            "intent": {"parameters": {"criteria": ["price"], "top_n": 1}},
            "query": "rank suppliers",
        },
    )

    output = agent.run(context)

    ranking_entry = output.data["ranking"][0]
    assert ranking_entry["flow_coverage"] == pytest.approx(0.6)
    assert ranking_entry["final_score"] > 0


def test_supplier_ranking_limits_to_opportunity_directory(monkeypatch):
    class StubPolicyEngine:
        def __init__(self):
            self.supplier_policies = [
                {
                    "policyName": "WeightAllocationPolicy",
                    "details": {"rules": {"default_weights": {"price": 1.0}}},
                },
                {"policyName": "CategoricalScoringPolicy", "details": {"rules": {}}},
                {
                    "policyName": "NormalizationDirectionPolicy",
                    "details": {"rules": {"price": "lower_is_better"}},
                },
            ]

    class StubQueryEngine:
        def fetch_purchase_order_data(
            self,
            intent=None,
            supplier_ids=None,
            supplier_names=None,
        ):
            return pd.DataFrame()

        def fetch_invoice_data(
            self,
            intent=None,
            supplier_ids=None,
            supplier_names=None,
        ):
            return pd.DataFrame()

        def fetch_procurement_flow(
            self,
            embed: bool = False,
            supplier_ids=None,
            supplier_names=None,
        ):
            return pd.DataFrame()

    nick = SimpleNamespace(
        settings=SimpleNamespace(extraction_model="gpt-oss", script_user="tester"),
        policy_engine=StubPolicyEngine(),
        query_engine=StubQueryEngine(),
    )

    agent = SupplierRankingAgent(nick)
    monkeypatch.setattr(agent, "_generate_justification", lambda row, criteria: "ok")
    monkeypatch.setattr(agent, "_read_table", lambda *args, **kwargs: pd.DataFrame())

    supplier_df = pd.DataFrame(
        {
            "supplier_id": ["S1", "S2", "S3"],
            "supplier_name": ["Alpha", "Beta", "Gamma"],
            "avg_unit_price": [10.0, 8.0, 12.0],
        }
    )

    context = AgentContext(
        workflow_id="wf1",
        agent_id="supplier_ranking",
        user_id="user",
        input_data={
            "supplier_data": supplier_df,
            "supplier_directory": [
                {"supplier_id": "S1", "supplier_name": "Alpha"},
                {"supplier_id": "S2", "supplier_name": "Beta"},
            ],
            "intent": {"parameters": {"criteria": ["price"], "top_n": 3}},
            "query": "rank suppliers",
        },
    )

    output = agent.run(context)

    assert output.status == AgentStatus.SUCCESS
    ranking_ids = {entry["supplier_id"] for entry in output.data["ranking"]}
    assert ranking_ids == {"S1", "S2"}
    assert all(entry["supplier_id"] in {"S1", "S2"} for entry in output.data["ranking"])


def test_ranking_uses_context_policy_weights(monkeypatch):
    nick = DummyNick()
    agent = SupplierRankingAgent(nick)
    monkeypatch.setattr(agent, "_generate_justification", lambda row, criteria: "ok")

    df = pd.DataFrame(
        {
            "supplier_id": ["S1", "S2"],
            "supplier_name": ["Alpha", "Beta"],
            "delivery": [9, 6],
            "price": [12.0, 8.0],
            "risk": [4.0, 2.0],
        }
    )

    policies = [
        {
            "policyName": "WeightAllocationPolicy",
            "details": {
                "rules": {"default_weights": {"delivery": 1.0, "price": 0.0, "risk": 0.0}}
            },
        },
        {
            "policyName": "NormalizationDirectionPolicy",
            "details": {"rules": {"delivery": "higher_is_better"}},
        },
    ]

    context = AgentContext(
        workflow_id="wf1",
        agent_id="supplier_ranking",
        user_id="tester",
        input_data={
            "supplier_data": df,
            "policies": policies,
            "intent": {"parameters": {"criteria": ["delivery"], "top_n": 2}},
            "query": "rank suppliers",
        },
    )

    output = agent.run(context)

    assert output.status == AgentStatus.SUCCESS
    assert output.data["ranking"][0]["supplier_id"] == "S1"
    assert output.data["ranking"][0]["weights"]["delivery"] == 1.0


def test_supplier_ranking_instruction_overrides(monkeypatch):
    class StubPolicyEngine:
        def __init__(self):
            self.supplier_policies = []

    class StubQueryEngine:
        def fetch_supplier_data(self, *_args, **_kwargs):
            return pd.DataFrame()

    nick = SimpleNamespace(
        settings=SimpleNamespace(extraction_model="gpt-oss", script_user="tester"),
        policy_engine=StubPolicyEngine(),
        query_engine=StubQueryEngine(),
    )

    agent = SupplierRankingAgent(nick)
    monkeypatch.setattr(agent, "_load_procurement_tables", lambda *_: {})
    monkeypatch.setattr(agent, "_merge_supplier_metrics", lambda df, _tables: df)
    monkeypatch.setattr(agent, "_build_supplier_profiles", lambda _tables, ids: {sid: {} for sid in ids})
    monkeypatch.setattr(agent, "_generate_justification", lambda row, criteria: "ok")

    df = pd.DataFrame(
        {
            "supplier_id": ["S1", "S2", "S3"],
            "supplier_name": ["Alpha", "Beta", "Gamma"],
            "price": [50, 40, 60],
            "delivery": [5, 10, 7],
        }
    )

    prompts = [
        {
            "promptId": 1,
            "prompts_desc": "{\"criteria\": [\"price\"], \"top_n\": 2}",
        }
    ]

    policies = [
        {
            "policyId": 10,
            "policyName": "WeightAllocationPolicy",
            "details": {"rules": {"default_weights": {"price": 1.0}}},
        },
        {
            "policyId": 11,
            "policyName": "NormalizationDirectionPolicy",
            "details": {"rules": {"price": "lower_is_better"}},
        },
    ]

    context = AgentContext(
        workflow_id="wf-instruction",
        agent_id="supplier_ranking",
        user_id="tester",
        input_data={
            "supplier_data": df,
            "prompts": prompts,
            "policies": policies,
            "intent": {},
            "query": "rank suppliers",
        },
    )

    output = agent.run(context)

    ranking = output.data["ranking"]
    assert len(ranking) == 2
    assert ranking[0]["supplier_id"] == "S2"
    weights = ranking[0]["weights"]
    assert weights["price"] == pytest.approx(1.0)
    assert context.input_data["intent"].get("parameters", {}).get("criteria") == ["price"]


def test_ensure_payment_terms_score_from_terms_text():
    df = pd.DataFrame({"payment_terms": ["Net 30"]})
    result = ensure_payment_terms_score(df.copy())
    assert pytest.approx(66.67, abs=0.01) == result.loc[0, "payment_terms_score"]

def test_ensure_payment_terms_score_leaves_unknown_null(caplog):
    """Unreadable terms must score NULL, never a neutral 50.

    This previously imputed 50.0, which claimed we had measured a supplier we had not.
    It also defeated weight renormalisation: an all-unknown metric is meant to drop out
    of the weighted sum, and a column of 50s never does.
    """
    df = pd.DataFrame({"supplier": ["S1"], "payment_terms": ["Deferred"]})
    with caplog.at_level("INFO"):
        result = ensure_payment_terms_score(df.copy())
    assert result.loc[0, "payment_terms_score"] is None or pd.isna(
        result.loc[0, "payment_terms_score"]
    )
    assert any("payment_terms_score" in record.getMessage() for record in caplog.records)


# ---------------------------------------------------------------------------
# JSON-safe coercion helpers
# ---------------------------------------------------------------------------

import math
import numpy as np


def _is_json_native(v):
    """Return True if *v* is a type that json.dumps handles without a custom encoder."""
    return v is None or isinstance(v, (bool, int, float, str, list, dict))


def test_json_safe_coerces_numpy_nan_to_none():
    assert _json_safe(np.nan) is None


def test_json_safe_coerces_float_nan_to_none():
    assert _json_safe(float("nan")) is None


def test_json_safe_coerces_pandas_na_to_none():
    assert _json_safe(pd.NA) is None


def test_json_safe_coerces_pandas_nat_to_none():
    assert _json_safe(pd.NaT) is None


def test_json_safe_coerces_numpy_int64():
    result = _json_safe(np.int64(42))
    assert result == 42
    assert type(result) is int


def test_json_safe_coerces_numpy_float64():
    result = _json_safe(np.float64(3.14))
    assert math.isclose(result, 3.14)
    assert type(result) is float


def test_json_safe_coerces_numpy_float64_nan_to_none():
    assert _json_safe(np.float64("nan")) is None


def test_json_safe_passes_through_none():
    assert _json_safe(None) is None


def test_json_safe_passes_through_native_float():
    assert _json_safe(1.5) == 1.5
    assert type(_json_safe(1.5)) is float


def test_json_safe_passes_through_native_int():
    assert _json_safe(7) == 7
    assert type(_json_safe(7)) is int


def test_json_safe_passes_through_str():
    assert _json_safe("hello") == "hello"


def test_json_safe_coerces_numpy_bool():
    assert _json_safe(np.bool_(True)) is True
    assert type(_json_safe(np.bool_(True))) is bool


def test_prepare_ranking_entry_no_pandas_objects(monkeypatch):
    """_prepare_ranking_entry must produce only JSON-native types even when the
    DataFrame row contains np.nan, pandas NA, and numpy scalar values."""

    class StubPolicyEngine:
        def __init__(self):
            self.supplier_policies = [
                {
                    "policyName": "WeightAllocationPolicy",
                    "details": {"rules": {"default_weights": {"price": 1.0}}},
                },
            ]

    class StubQueryEngine:
        def fetch_supplier_data(self, *_, **__):
            return pd.DataFrame()

    nick = SimpleNamespace(
        settings=SimpleNamespace(extraction_model="gpt-oss", script_user="tester"),
        policy_engine=StubPolicyEngine(),
        query_engine=StubQueryEngine(),
    )

    agent = SupplierRankingAgent(nick)

    # Build a row that mimics a DataFrame row with problematic types
    row = pd.Series({
        "supplier_id": "SUP-TECHWORLD",
        "supplier_name": pd.NA,          # pandas NA – was serialising as {"__module__":"pandas"}
        "final_score": np.float64(50.0),
        "price_score": np.nan,
        "delivery_score": pd.NA,
        "risk_score": pd.NA,             # same root cause as the live bug
        "payment_terms_score": np.float64(66.67),
        "payment_terms": pd.NA,
        "avg_unit_price": pd.NA,         # same root cause as the live bug
        "total_spend": np.float64(0.0),
        "po_count": pd.NA,
        "invoice_count": pd.NA,
        "avg_lead_time_days": np.nan,
        "justification": "ok",
        "contact_name_1": pd.NA,
        "contact_email_1": pd.NA,
        "flow_coverage": np.float64(0.0),
    })

    entry = agent._prepare_ranking_entry(row, profile=None, weights={"price": 1.0})

    # Every value in the entry must be JSON-native
    bad_fields = {
        k: (v, type(v).__name__)
        for k, v in entry.items()
        if not _is_json_native(v)
    }
    assert bad_fields == {}, f"Non-JSON-native values found: {bad_fields}"

    # Specific assertions from the live bug report
    assert entry["supplier_name"] is None       # pd.NA → None
    assert entry["risk_score"] is None          # pd.NA → None
    assert entry["avg_unit_price"] is None      # pd.NA → None
    assert entry["price_score"] is None         # np.nan → None
    assert entry["delivery_score"] is None      # pd.NA → None
    assert entry["po_count"] is None            # pd.NA → None
    assert isinstance(entry["final_score"], float)
    assert math.isclose(entry["final_score"], 50.0)

    # Must be JSON-serialisable without errors
    json.dumps(entry)  # would raise TypeError if any pandas/numpy objects remained
