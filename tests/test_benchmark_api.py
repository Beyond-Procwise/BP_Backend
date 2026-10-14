"""Benchmark endpoints: pure preview route + live-loader normalisation."""
import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routers import benchmark as benchmark_router
from services.benchmark_live import DISCLOSURES, _norm_currency, _norm_item, _norm_uom

GOLDEN = Path(__file__).parent / "fixtures" / "benchmark" / "golden.json"


def _client() -> TestClient:
    from api.auth import require_user

    app = FastAPI()
    app.include_router(benchmark_router.router)
    # /benchmark/preview resolves the caller (P8 phase 2); these tests are about
    # the engine, not authentication, so the principal is pinned: nobody.
    app.dependency_overrides[require_user] = lambda: None
    return TestClient(app)


def test_preview_reproduces_golden_d001_over_http():
    golden = json.loads(GOLDEN.read_text())
    quote = next(q for q in golden["quotes"] if q["deal_id"] == "D001")
    resp = _client().post("/benchmark/preview", json={
        "quote": quote,
        "points": golden["points"],
        "location_index_table": golden["location_index_table"],
        "index_table": golden["index_table"],
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["final_benchmark"] == 888.98
    assert body["unit_variance_gbp"] == 61.02
    assert body["total_cost_gap"] == 6102
    assert body["confidence"] == "HIGH"
    assert body["gated"] is False


def test_preview_fails_closed_with_no_points():
    golden = json.loads(GOLDEN.read_text())
    quote = next(q for q in golden["quotes"] if q["deal_id"] == "D001")
    resp = _client().post("/benchmark/preview", json={"quote": quote, "points": []})
    assert resp.status_code == 200
    body = resp.json()
    assert body["gated"] is True
    assert body["confidence"] == "No Data"
    assert body["final_benchmark"] is None


def test_normalisation_folds_case_whitespace_and_defaults():
    assert _norm_item("  HR  Transformation   Programme ") == "hr transformation programme"
    assert _norm_uom(None) == "each"
    assert _norm_uom(" Each ") == "each"
    assert _norm_currency(None) == "GBP"
    assert _norm_currency(" gbp ") == "GBP"


def test_disclosures_pass_the_output_safety_gate():
    # The HTTP boundary rewrites mechanism-level text (SAFE_REPLY); an audit
    # payload must never be silently rewritten, so pin gate-cleanliness here.
    from services.output_safety import inspect

    for disclosure in DISCLOSURES:
        assert inspect(disclosure) == [], disclosure
