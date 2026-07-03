"""Read-only end-to-end extraction audit harness.

Runs the PRODUCTION default engine (hybrid_v4) on:
  1. A synthetic invoice with KNOWN ground-truth values (accuracy check)
  2. A sample of real invoices from documents/invoice (robustness check)

Calls dispatch._run_hybrid_v4(path, doc_type) which returns an ExtractionResult
WITHOUT persisting. The invoice path opens no DB connection -> zero writes.

Reports per document: parse text length, #candidates (committed),
#residuals, judge_calls, doc_pk, and full committed field list w/ confidence.
For the synthetic doc, scores each committed field against ground truth.
"""
from __future__ import annotations
import json, os, sys, time, tempfile, traceback
from io import BytesIO
from pathlib import Path

PROJECT_ROOT = "/home/muthu/PycharmProjects/BP_Backend"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

OUT = Path(PROJECT_ROOT) / "artifacts" / "e2e_audit" / "extract_results.json"
OUT.parent.mkdir(parents=True, exist_ok=True)

GROUND_TRUTH = {
    "invoice_id": "INV-2025-0042",
    "invoice_date": "2025-03-15",
    "po_id": "PO-7891",
    "supplier_name": "Acme Industrial Supplies Ltd",
    "currency": "GBP",
    "invoice_amount": "1250.00",
    "tax_amount": "250.00",
    "invoice_total_incl_tax": "1500.00",
}


def make_synthetic_invoice() -> str:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    fd, path = tempfile.mkstemp(suffix="_synthetic_invoice.pdf")
    os.close(fd)
    c = canvas.Canvas(path, pagesize=A4)
    w, h = A4
    c.setFont("Helvetica-Bold", 16); c.drawString(40, h-50, "Acme Industrial Supplies Ltd")
    c.setFont("Helvetica", 9)
    c.drawString(40, h-65, "Unit 7, Meadow Park Industrial Estate")
    c.drawString(40, h-77, "Birmingham, B12 9QR, United Kingdom")
    c.setFont("Helvetica-Bold", 22); c.drawString(380, h-50, "INVOICE")
    c.setFont("Helvetica", 10)
    c.drawString(380, h-80, "Invoice No: INV-2025-0042")
    c.drawString(380, h-95, "Invoice Date: 15/03/2025")
    c.drawString(380, h-110, "Due Date: 14/04/2025")
    c.drawString(380, h-125, "PO Reference: PO-7891")
    c.drawString(40, h-120, "Bill To: Pinnacle Manufacturing Group")
    c.drawString(40, h-135, "23 Victoria Road, Manchester, M1 4HJ")
    # line items table
    y = h-200
    c.setFont("Helvetica-Bold", 10)
    c.drawString(40, y, "Description"); c.drawString(300, y, "Qty")
    c.drawString(360, y, "Unit Price"); c.drawString(460, y, "Amount")
    c.setFont("Helvetica", 10)
    rows = [("Steel brackets (box of 50)", "10", "75.00", "750.00"),
            ("Industrial fasteners M8", "20", "25.00", "500.00")]
    for desc, qty, up, amt in rows:
        y -= 18
        c.drawString(40, y, desc); c.drawString(300, y, qty)
        c.drawString(360, y, "GBP " + up); c.drawString(460, y, "GBP " + amt)
    y -= 40
    c.drawString(360, y, "Subtotal:"); c.drawString(460, y, "GBP 1250.00")
    y -= 15
    c.drawString(360, y, "VAT (20%):"); c.drawString(460, y, "GBP 250.00")
    y -= 15
    c.setFont("Helvetica-Bold", 10)
    c.drawString(360, y, "Total:"); c.drawString(460, y, "GBP 1500.00")
    c.save()
    return path


def run_one(path: str, doc_type: str, label: str) -> dict:
    from src.services.extraction_v3.dispatch import _run_hybrid_v4
    from src.services.extraction_v3.extraction_v4.engine import PDFParser
    t0 = time.time()
    rec = {"label": label, "path": os.path.basename(path), "doc_type": doc_type}
    try:
        try:
            txt = PDFParser.extract_text(path)
            rec["parse_text_len"] = len(txt or "")
        except Exception as e:
            rec["parse_text_len"] = None
            rec["parse_err"] = str(e)[:120]
        result = _run_hybrid_v4(path, doc_type)
        rec["doc_pk"] = result.doc_pk
        rec["judge_calls"] = result.judge_calls
        rec["pipeline_version"] = result.pipeline_version
        rec["n_committed"] = len(result.committed)
        rec["n_residuals"] = len(result.residuals)
        rec["committed"] = [
            {"field": cf.field_path, "value": cf.value,
             "conf": round(getattr(cf, "final_confidence", 0.0) or 0.0, 3),
             "model": getattr(cf, "model", None)}
            for cf in result.committed
        ]
        rec["residuals"] = [
            {"field": r.field_path, "reason": str(getattr(r, "reason", ""))}
            for r in result.residuals
        ]
        rec["elapsed_s"] = round(time.time() - t0, 1)
    except Exception as e:
        rec["error"] = str(e)[:300]
        rec["trace"] = traceback.format_exc()[-800:]
        rec["elapsed_s"] = round(time.time() - t0, 1)
    return rec


def score_synthetic(rec: dict) -> dict:
    got = {c["field"]: c["value"] for c in rec.get("committed", [])}
    scored = {}
    correct = 0
    for f, exp in GROUND_TRUTH.items():
        actual = got.get(f)
        ok = False
        if actual is not None:
            a = str(actual).strip().lower().replace(",", "").replace("£", "").replace("gbp", "").strip()
            e = str(exp).strip().lower().replace(",", "")
            ok = (e in a) or (a in e) or (a == e)
        scored[f] = {"expected": exp, "actual": actual, "match": ok}
        if ok:
            correct += 1
    return {"fields_checked": len(GROUND_TRUTH), "correct": correct,
            "accuracy_pct": round(100.0 * correct / len(GROUND_TRUTH), 1), "detail": scored}


def main():
    results = []
    # 1. synthetic
    print("[harness] generating synthetic invoice...", flush=True)
    syn = make_synthetic_invoice()
    print("[harness] running synthetic through hybrid_v4...", flush=True)
    r = run_one(syn, "invoice", "SYNTHETIC")
    r["ground_truth_score"] = score_synthetic(r)
    results.append(r)
    _flush(results)
    os.unlink(syn)
    # 2. real invoices (PDFs only, first 5 deterministic by name)
    inv_dir = Path(PROJECT_ROOT) / "documents" / "invoice"
    pdfs = sorted([p for p in inv_dir.glob("*.pdf")])[:5]
    for p in pdfs:
        print(f"[harness] running real invoice {p.name}...", flush=True)
        results.append(run_one(str(p), "invoice", "REAL"))
        _flush(results)
    print("[harness] DONE", flush=True)
    _flush(results, final=True)


def _flush(results, final=False):
    OUT.write_text(json.dumps({"final": final, "results": results}, indent=2, default=str))


if __name__ == "__main__":
    main()
