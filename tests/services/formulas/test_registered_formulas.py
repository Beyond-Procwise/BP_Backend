"""The registered formulas themselves.

Two jobs. First, re-run every golden vector under CI so a failure is a list
rather than an import error on whichever module happens to load first. Second,
prove that going through ``evaluate`` produces the SAME number as calling the
original implementation directly --- which is the whole claim of the migration.
"""
from __future__ import annotations

import math
from datetime import date

import pytest

from src.services.formulas import (
    REGISTRY, UNASSESSED, evaluate, evaluate_many, model_inventory, verify_all,
)
from src.services.formulas import definitions  # noqa: F401  (registers everything)


def test_every_golden_vector_reproduces():
    failures = verify_all()
    assert failures == [], "\n".join(failures)


def test_the_registry_is_not_empty():
    assert len(REGISTRY) >= 40


def test_every_formula_declares_an_owner_a_purpose_and_a_vector():
    for spec in REGISTRY.values():
        assert spec.owner, spec.name
        assert spec.purpose, spec.name
        assert spec.golden, spec.name
        assert spec.contract.inputs, spec.name


def test_every_version_hash_is_distinct():
    hashes = {}
    for spec in REGISTRY.values():
        assert spec.version_hash not in hashes, (
            f"{spec.name} and {hashes.get(spec.version_hash)} share a version hash"
        )
        hashes[spec.version_hash] = spec.name


def test_no_formula_claims_a_gpss_version_there_is_no_dictionary_for():
    """GPSS does not exist in this project (00_seam_map.md B3).

    The field is accepted because the specification names it. Nothing may
    populate it until there is a dictionary to resolve it against, or the
    column becomes exactly the shadow vocabulary that decision prevents.
    """
    claimed = [s.name for s in REGISTRY.values() if s.gpss_version]
    assert claimed == [], (
        "these formulas claim a GPSS version with no dictionary behind it: "
        f"{claimed}"
    )


# --------------------------------------------------------------- parity


class TestParityWithTheOriginals:
    """evaluate(name, ctx) == the original call, number for number."""

    def test_relationship_confidence(self):
        from src.services import linking_engine as le

        inv = {"invoice_id": "INV-9001", "po_id": "PO-4001", "supplier_id": "SUP-100",
               "converted_amount_usd": 12500.00, "currency": "GBP",
               "invoice_date": "2025-04-10", "country": "GB", "region": "London"}
        po = {"po_id": "PO-4001", "supplier_id": "SUP-100",
              "converted_amount_usd": 12500.00, "currency": "GBP",
              "order_date": "2025-03-01", "ship_to_country": "GB",
              "delivery_region": "London"}
        lines = [{"item_description": "Dell Latitude 5540 laptop",
                  "quantity": 10, "unit_price": 780.0}]

        direct = le.score_link(inv, po, "invoice_po", lines, lines)
        viareg = evaluate("linking.relationship_confidence", {
            "source_row": inv, "target_row": po, "profile_name": "invoice_po",
            "source_lines": lines, "target_lines": lines, "set_amount_usd": None,
        })
        assert viareg.value == direct

    def test_payment_terms_score(self):
        from src.agents.supplier_ranking_agent import _normalize_days_to_score

        for days in (0, 15, 30, 45, 60, 90, 200):
            direct = _normalize_days_to_score(days)
            viareg = evaluate("supplier.payment_terms_score",
                              {"payment_terms_days": days,
                               "min_days": None, "max_days": None})
            assert viareg.value == direct, days

    def test_deal_price_score_matches_the_dataframe_path(self):
        import numpy as np
        import pandas as pd
        from src.agents.supplier_ranking_agent import SupplierRankingAgent as SRA

        prices = [100000.0, 101100.0, 103050.0]
        direct = list(SRA._score_deal_price(None, pd.DataFrame({"price": prices}))
                      ["price_score"])
        batch = evaluate_many("supplier.deal_price_score",
                              [{"price": p} for p in prices])
        assert batch.values == pytest.approx(direct)

    def test_composite_score_matches_the_dataframe_path(self):
        import numpy as np
        import pandas as pd
        from src.agents.supplier_ranking_agent import composite_scores

        frame = pd.DataFrame({
            "price_score": [100.0, 98.9, np.nan],
            "delivery_score": [80.0, 90.0, np.nan],
        })
        weights = {"price": 0.6, "delivery": 0.4}
        direct, _ = composite_scores(frame, weights)

        batch = evaluate_many(
            "supplier.composite_score",
            [{"scores": {"price": 100.0, "delivery": 80.0}},
             {"scores": {"price": 98.9, "delivery": 90.0}},
             {"scores": {"price": float("nan"), "delivery": float("nan")}}],
            shared={"weights": weights},
        )
        for got, want in zip(batch.values, direct):
            if math.isnan(want):
                assert math.isnan(got)
            else:
                assert got == pytest.approx(want)

    def test_counter_plan(self):
        from src.agents.negotiation_agent import compute_decision

        payload = {"current_offer": 100.0, "target_price": 80.0, "round": 2}
        direct = compute_decision(payload, "", 105.0)
        viareg = evaluate("negotiation.counter_plan", {
            "current_offer": 100.0, "target_price": 80.0, "round": 2,
            "max_rounds": None, "walkaway_price": None, "currency": None,
            "ask_early_pay_disc": None, "ask_lead_time_keep": None,
            "supplier_message_text": None, "offer_prev": 105.0,
        })
        assert viareg.value["counter_price"] == direct["counter_price"]
        assert viareg.value["decision"] == direct["decision"]

    def test_price_outlier_verdict(self):
        from services.price_outlier.rule import OutlierSettings, assess

        peers = [100.0, 102.0, 98.0, 101.0, 99.0, 100.5]
        direct = assess(1200.0, peers, OutlierSettings())
        viareg = evaluate("price_outlier.verdict",
                          {"price": 1200.0, "peers": peers, "settings": None})
        assert viareg.value == direct

    def test_benchmark_adjusted_price(self):
        import json
        from pathlib import Path

        from services.benchmark.engine import compute_benchmark
        from services.benchmark.models import (
            BenchmarkPoint, BenchmarkSettings, QuoteLine,
        )

        fixture = Path(__file__).resolve().parents[3] / "tests/fixtures/benchmark/golden.json"
        g = json.loads(fixture.read_text())
        quote, points = g["quotes"][0], g["points"]

        direct = compute_benchmark(
            QuoteLine(**quote), [BenchmarkPoint(**p) for p in points],
            g["location_index_table"], g["index_table"], BenchmarkSettings(),
        )
        viareg = evaluate("benchmark.adjusted_price", {
            "quote": quote, "points": points,
            "location_index_table": g["location_index_table"],
            "index_table": g["index_table"], "settings": None,
        })
        assert viareg.value.final_benchmark == direct.final_benchmark
        assert viareg.value.unit_variance_gbp == direct.unit_variance_gbp
        assert viareg.value.total_cost_gap == direct.total_cost_gap


# --------------------------------------------------------------- contracts bite


class TestContractsBiteOnRealFormulas:
    def test_an_unknown_profile_name_does_not_crash_the_caller(self):
        r = evaluate("linking.relationship_confidence", {
            "source_row": {}, "target_row": {}, "profile_name": "no_such_profile",
            "source_lines": None, "target_lines": None, "set_amount_usd": None,
        })
        assert r.value is UNASSESSED
        assert r.findings[-1].code == "evaluation_error"

    def test_a_risk_score_on_the_wrong_scale_is_refused(self):
        """risk_score is stored 0-100 in one place and 0-1 in another.

        The contract declares 0-1, so the 0-100 value is refused rather than
        silently multiplying the finding weightage by 76.
        """
        ok = evaluate("opportunity.finding_weight_factor",
                      {"base_impact": 1000.0, "risk_score": 0.75, "coverage": 0.25})
        assert ok.value == pytest.approx(1000.0 * 1.75 * 1.25)

        bad = evaluate("opportunity.finding_weight_factor",
                       {"base_impact": 1000.0, "risk_score": 75.0, "coverage": 0.25})
        assert bad.value is UNASSESSED
        assert bad.findings[0].code == "out_of_range"

    def test_a_negative_price_is_refused_by_the_outlier_contract(self):
        r = evaluate("price_outlier.verdict",
                     {"price": -5.0, "peers": [1.0] * 6, "settings": None})
        assert r.value is UNASSESSED


# --------------------------------------------------------------- inventory


class TestModelInventory:
    def test_it_reports_every_registered_formula(self):
        rows = model_inventory(include_dependents=False)
        assert {r["name"] for r in rows} == set(REGISTRY)

    def test_every_row_carries_the_spec_fields(self):
        for row in model_inventory(include_dependents=False):
            for field in ("name", "version", "owner", "purpose", "contract",
                          "gpss_version", "last_validated", "effective_from"):
                assert field in row, (row["name"], field)
            assert row["last_validated"], row["name"]
            assert row["contract"]["inputs"], row["name"]
            assert row["contract"]["output"]["unit"], row["name"]

    def test_markdown_renders(self):
        from src.services.formulas import render_markdown

        text = render_markdown(model_inventory(include_dependents=False))
        assert "# Model Inventory" in text
        assert "linking.relationship_confidence" in text
