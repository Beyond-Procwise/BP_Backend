"""Vendor-onboarding API tests.

Exercises the full onboarding loop:
    upload → preview → correct → save → re-extract uses template.
"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.services.extraction_v2.template_store import InMemoryTemplateStore


FIXTURE_DIR = (
    Path(__file__).resolve().parents[1]
    / "structural_extractor" / "fixtures" / "docs"
)
AQUARIUS = FIXTURE_DIR / "AQUARIUS INV-25-050 for PO508084 .pdf"


def _build_app() -> tuple[FastAPI, InMemoryTemplateStore]:
    """Build a FastAPI app with the onboarding router + a fresh store."""
    from src.api.routers.vendors import build_router

    store = InMemoryTemplateStore()
    app = FastAPI()
    app.include_router(build_router(store=store))
    return app, store


@pytest.mark.skipif(not AQUARIUS.exists(), reason="fixture missing")
class TestOnboardingFlow:
    def test_upload_returns_session_with_residuals(self):
        app, _ = _build_app()
        client = TestClient(app)

        with AQUARIUS.open("rb") as fh:
            r = client.post(
                "/vendors/onboard/upload",
                files={"file": (AQUARIUS.name, fh, "application/pdf")},
                data={"doc_type": "Invoice"},
            )
        assert r.status_code == 200, r.text
        body = r.json()
        assert "session_id" in body
        assert "fingerprint" in body
        assert "committed" in body
        assert "residuals" in body
        # supplier_name should land in residuals (one of the gaps we know)
        residual_fields = {r["field"] for r in body["residuals"]}
        assert "supplier_name" in residual_fields

    def test_correct_then_save_writes_template(self):
        app, store = _build_app()
        client = TestClient(app)

        with AQUARIUS.open("rb") as fh:
            r = client.post(
                "/vendors/onboard/upload",
                files={"file": (AQUARIUS.name, fh, "application/pdf")},
                data={"doc_type": "Invoice"},
            )
        body = r.json()
        sid = body["session_id"]
        fingerprint = body["fingerprint"]

        # User corrects supplier_name
        r = client.post(
            f"/vendors/onboard/{sid}/correct",
            json={"field": "supplier_name", "value": "Aquarius Marketing Ltd"},
        )
        assert r.status_code == 200, r.text

        # User saves the template
        r = client.post(
            f"/vendors/onboard/{sid}/save",
            json={"vendor_name": "Aquarius"},
        )
        assert r.status_code == 200, r.text

        # Template should now be in the store with the hint
        t = store.get(fingerprint)
        assert t is not None
        assert t.vendor_name == "Aquarius"
        assert "supplier_name" in t.field_hints
        assert t.field_hints["supplier_name"].value == "Aquarius Marketing Ltd"

    def test_replay_after_save_uses_template(self):
        app, store = _build_app()
        client = TestClient(app)

        # Step 1 — upload, correct, save
        with AQUARIUS.open("rb") as fh:
            up = client.post(
                "/vendors/onboard/upload",
                files={"file": (AQUARIUS.name, fh, "application/pdf")},
                data={"doc_type": "Invoice"},
            )
        sid = up.json()["session_id"]
        client.post(
            f"/vendors/onboard/{sid}/correct",
            json={"field": "supplier_name", "value": "Aquarius Marketing Ltd"},
        )
        client.post(
            f"/vendors/onboard/{sid}/save",
            json={"vendor_name": "Aquarius"},
        )

        # Step 2 — upload the SAME doc again, expect supplier committed
        with AQUARIUS.open("rb") as fh:
            r = client.post(
                "/vendors/onboard/upload",
                files={"file": (AQUARIUS.name, fh, "application/pdf")},
                data={"doc_type": "Invoice"},
            )
        body = r.json()
        committed_fields = {c["field"]: c for c in body["committed"]}
        assert "supplier_name" in committed_fields
        assert committed_fields["supplier_name"]["value"] == "Aquarius Marketing Ltd"
        assert body["template_used"] is True

    def test_form_html_is_served(self):
        app, _ = _build_app()
        client = TestClient(app)
        r = client.get("/vendors/onboard")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]
        # Has a file input — minimal sanity check
        assert "<input" in r.text and "file" in r.text
