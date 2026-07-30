from fastapi.testclient import TestClient


def test_value_summary_endpoint(monkeypatch):
    import src.services.value_summary_service as vss
    monkeypatch.setattr(vss, "build_value_summary", lambda conn=None: {
        "verified_found_gbp": 950.0, "recovered_gbp": 0.0, "potential_gbp": 0.0,
        "finding_count": 1, "since": None, "by_supplier": [], "findings": [],
        "sources": {"discrepancies": "ok", "opportunities": "ok", "benchmark": "ok"},
    })
    from src.api.main import app
    client = TestClient(app)
    r = client.get("/spendiq/value-summary")
    assert r.status_code == 200
    body = r.json()
    assert body["verified_found_gbp"] == 950.0
    assert "generated_at" in body
