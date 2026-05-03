"""Vendor onboarding API.

Lets a non-technical user teach the V2 extraction pipeline a new vendor
layout by uploading a sample document, reviewing/correcting the
extracted fields in a browser, and saving — no YAML, no code edits.

Routes:
    GET  /vendors/onboard              minimal HTML form
    POST /vendors/onboard/upload       upload doc → returns session preview
    POST /vendors/onboard/{sid}/correct  per-field correction
    POST /vendors/onboard/{sid}/save     persist as a vendor template

The router is built via :func:`build_router` so tests can pass an
isolated in-memory store. Production calls :func:`build_router` with a
Postgres-backed store wired at app startup.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import asdict
from typing import Any, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from src.services.extraction_v2.pipeline import ExtractionPipelineV2
from src.services.extraction_v2.template_store import (
    InMemoryTemplateStore, TemplateStore,
)
from src.services.structural_extractor.parsing import parse


logger = logging.getLogger(__name__)


__all__ = ["build_router"]


# Per-session state lives in a process-local dict. Sessions are short-
# lived (one onboarding flow); production wires a Redis-backed map.
class _Session:
    __slots__ = ("session_id", "filename", "doc_bytes", "doc_type",
                 "fingerprint", "corrections")

    def __init__(self, session_id: str, filename: str, doc_bytes: bytes,
                 doc_type: str, fingerprint: str):
        self.session_id = session_id
        self.filename = filename
        self.doc_bytes = doc_bytes
        self.doc_type = doc_type
        self.fingerprint = fingerprint
        self.corrections: dict[str, Any] = {}


_SESSIONS: dict[str, _Session] = {}


# --- Request models -----------------------------------------------------------

class CorrectionRequest(BaseModel):
    field: str = Field(..., min_length=1, max_length=64)
    value: Any
    label: Optional[str] = Field(None, max_length=128)


class SaveRequest(BaseModel):
    vendor_name: Optional[str] = Field(None, max_length=200)


# --- Helpers ------------------------------------------------------------------

def _serialize_committed(c) -> dict:
    return {
        "field": c.field,
        "value": str(c.value) if c.value is not None else None,
        "confidence": round(c.confidence, 2),
        "why": c.why,
        "locator_count": c.locator_count,
    }


def _serialize_residual(r) -> dict:
    return {
        "field": r.field,
        "why": r.why,
        "candidates": [
            {
                "value": str(cand.value),
                "confidence": round(cand.confidence, 2),
                "why": cand.why,
            }
            for cand in r.candidates
        ],
    }


def _run_pipeline(doc_bytes: bytes, filename: str, doc_type: str,
                  store: TemplateStore) -> dict:
    parsed = parse(doc_bytes, filename)
    pipeline = ExtractionPipelineV2(template_store=store)
    result = pipeline.extract(parsed, doc_type)
    return {
        "fingerprint": result.fingerprint,
        "template_used": result.template_used,
        "committed": [_serialize_committed(c) for c in result.committed.values()],
        "residuals": [_serialize_residual(r) for r in result.residuals],
    }


# --- HTML form ----------------------------------------------------------------

_FORM_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>Onboard a vendor</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 960px; margin: 2rem auto; }
  h1 { font-size: 1.4rem; }
  .row { display: flex; gap: 1rem; margin: .4rem 0; align-items: center; }
  .row label { width: 14rem; font-weight: 600; }
  .row input[type=text] { flex: 1; padding: .35rem .5rem; }
  .ok { color: #0a7c2f; }
  .miss { color: #a94400; }
  pre { background: #f4f4f4; padding: .8rem; max-height: 24rem; overflow:auto; }
</style>
</head>
<body>
<h1>Vendor onboarding</h1>
<p>Upload a sample document, correct any missing fields, then save the template.
The system will remember this layout — future documents from the same vendor
will be extracted automatically.</p>

<form id="upload-form" enctype="multipart/form-data">
  <div class="row">
    <label>Document type</label>
    <select name="doc_type">
      <option>Invoice</option><option>Purchase_Order</option><option>Quote</option>
    </select>
  </div>
  <div class="row">
    <label>File (PDF/DOCX/XLSX/CSV)</label>
    <input type="file" name="file" required>
  </div>
  <button type="submit">Upload &amp; preview</button>
</form>

<div id="preview"></div>

<script>
const form = document.getElementById('upload-form');
const preview = document.getElementById('preview');
let SID = null;

form.onsubmit = async (e) => {
  e.preventDefault();
  preview.innerHTML = 'Uploading…';
  const fd = new FormData(form);
  const r = await fetch('/vendors/onboard/upload', {method:'POST', body: fd});
  const data = await r.json();
  if (!r.ok) { preview.innerHTML = 'Error: ' + (data.detail || r.status); return; }
  SID = data.session_id;
  render(data);
};

function render(data) {
  let html = `<h2>Preview <small>(fingerprint ${data.fingerprint.slice(0,8)}…)</small></h2>`;
  html += '<div>';
  for (const c of data.committed) {
    html += `<div class="row"><label class="ok">[OK] ${c.field}</label>` +
            `<input type="text" name="${c.field}" value="${c.value || ''}" data-conf="${c.confidence}">` +
            `<span class="ok">conf ${c.confidence}</span></div>`;
  }
  for (const r of data.residuals) {
    const cand = (r.candidates && r.candidates[0]) ? r.candidates[0].value : '';
    html += `<div class="row"><label class="miss">[review] ${r.field}</label>` +
            `<input type="text" name="${r.field}" value="${cand}" placeholder="${r.why}">` +
            `<span class="miss">${r.why.split(':')[0]}</span></div>`;
  }
  html += '</div>';
  html += '<div class="row"><label>Vendor name</label><input type="text" id="vendor_name"></div>';
  html += '<button id="save-btn">Save template</button>';
  preview.innerHTML = html;

  document.getElementById('save-btn').onclick = async () => {
    // Send each filled-in residual as a correction
    const inputs = preview.querySelectorAll('input[type=text][name]');
    for (const el of inputs) {
      if (!el.value) continue;
      await fetch(`/vendors/onboard/${SID}/correct`, {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({field: el.name, value: el.value})
      });
    }
    const vname = document.getElementById('vendor_name').value || null;
    const r = await fetch(`/vendors/onboard/${SID}/save`, {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({vendor_name: vname})
    });
    if (r.ok) preview.innerHTML += '<p class="ok">Template saved.</p>';
    else { const e = await r.json(); preview.innerHTML += '<p class="miss">Error: '+(e.detail||r.status)+'</p>'; }
  };
}
</script>
</body></html>
"""


# --- Router factory -----------------------------------------------------------

def build_router(store: Optional[TemplateStore] = None) -> APIRouter:
    """Build the vendor onboarding router. Pass `store` for tests."""
    template_store: TemplateStore = store or InMemoryTemplateStore()
    router = APIRouter(prefix="/vendors", tags=["Vendor Onboarding"])

    @router.get("/onboard", response_class=HTMLResponse)
    def onboard_form():
        return HTMLResponse(_FORM_HTML)

    @router.post("/onboard/upload")
    async def upload(file: UploadFile = File(...),
                     doc_type: str = Form(...)):
        body = await file.read()
        if not body:
            raise HTTPException(400, "empty upload")
        try:
            preview = _run_pipeline(body, file.filename or "upload",
                                    doc_type, template_store)
        except Exception as e:
            logger.exception("onboarding extraction failed")
            raise HTTPException(500, f"extraction failed: {e}")

        sid = uuid.uuid4().hex
        _SESSIONS[sid] = _Session(
            session_id=sid,
            filename=file.filename or "upload",
            doc_bytes=body,
            doc_type=doc_type,
            fingerprint=preview["fingerprint"],
        )
        return JSONResponse({"session_id": sid, **preview})

    @router.post("/onboard/{sid}/correct")
    def correct(sid: str, req: CorrectionRequest):
        sess = _SESSIONS.get(sid)
        if sess is None:
            raise HTTPException(404, "session not found")
        sess.corrections[req.field] = (req.value, req.label)
        return {"ok": True, "field": req.field}

    @router.post("/onboard/{sid}/save")
    def save(sid: str, req: SaveRequest):
        sess = _SESSIONS.get(sid)
        if sess is None:
            raise HTTPException(404, "session not found")
        for fname, (value, label) in sess.corrections.items():
            template_store.record_correction(
                fingerprint=sess.fingerprint, field=fname, value=value,
                confidence=0.95, label=label,
                doc_type=sess.doc_type, vendor_name=req.vendor_name,
            )
        return {"ok": True, "fingerprint": sess.fingerprint,
                "fields_saved": list(sess.corrections.keys())}

    return router
