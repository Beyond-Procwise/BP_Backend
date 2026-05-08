"""Live extraction test — creates realistic procurement PDFs and runs
them through the full DataExtractionAgent pipeline (including LLM).

Requires:
    - Ollama running at localhost:11434 with BeyondProcwise/AgentNick
    - reportlab (pip install reportlab) for PDF generation

Usage:
    python tests/live_extraction_test.py
"""

import json
import os
import sys
import tempfile
import time
from datetime import date
from io import BytesIO
from types import SimpleNamespace

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(project_root, "src"))
sys.path.insert(0, project_root)
os.chdir(project_root)

from agents.data_extraction_agent import DataExtractionAgent

# ---------------------------------------------------------------------------
# PDF generation helpers
# ---------------------------------------------------------------------------

def _create_invoice_pdf() -> bytes:
    """Create a realistic UK invoice PDF with known field values."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.pdfgen import canvas

    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w, h = A4

    # Company header
    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, h - 50, "Acme Industrial Supplies Ltd")
    c.setFont("Helvetica", 9)
    c.drawString(40, h - 65, "Unit 7, Meadow Park Industrial Estate")
    c.drawString(40, h - 77, "Birmingham, B12 9QR, United Kingdom")

    # Invoice title
    c.setFont("Helvetica-Bold", 22)
    c.drawString(380, h - 50, "INVOICE")

    # Invoice details
    c.setFont("Helvetica-Bold", 10)
    c.drawString(380, h - 80, "Invoice No:")
    c.setFont("Helvetica", 10)
    c.drawString(460, h - 80, "INV-2025-0042")

    c.setFont("Helvetica-Bold", 10)
    c.drawString(380, h - 95, "Invoice Date:")
    c.setFont("Helvetica", 10)
    c.drawString(460, h - 95, "15/03/2025")

    c.setFont("Helvetica-Bold", 10)
    c.drawString(380, h - 110, "Due Date:")
    c.setFont("Helvetica", 10)
    c.drawString(460, h - 110, "14/04/2025")

    c.setFont("Helvetica-Bold", 10)
    c.drawString(380, h - 125, "PO Reference:")
    c.setFont("Helvetica", 10)
    c.drawString(460, h - 125, "PO-7891")

    # Bill To
    c.setFont("Helvetica-Bold", 10)
    c.drawString(40, h - 120, "Bill To:")
    c.setFont("Helvetica", 10)
    c.drawString(40, h - 135, "Pinnacle Manufacturing Group")
    c.drawString(40, h - 147, "23 Victoria Road")
    c.drawString(40, h - 159, "Manchester, M1 4HJ")

    # Payment terms
    c.setFont("Helvetica-Bold", 10)
    c.drawString(380, h - 145, "Payment Terms:")
    c.setFont("Helvetica", 10)
    c.drawString(480, h - 145, "Net 30")

    c.setFont("Helvetica-Bold", 10)
    c.drawString(380, h - 160, "Currency:")
    c.setFont("Helvetica", 10)
    c.drawString(460, h - 160, "GBP")

    # Line items table header
    y = h - 200
    c.setFont("Helvetica-Bold", 9)
    c.drawString(40, y, "Item")
    c.drawString(220, y, "Description")
    c.drawString(340, y, "Qty")
    c.drawString(380, y, "Unit Price")
    c.drawString(450, y, "Amount")
    c.line(40, y - 5, 530, y - 5)

    # Line items
    items = [
        ("WDG-4420", "Industrial Widget Assembly Kit", "10", "45.00", "450.00"),
        ("BLT-M12", "M12 Hex Bolt Pack (100pc)", "5", "28.50", "142.50"),
        ("SLV-009", "Stainless Steel Valve DN50", "2", "312.00", "624.00"),
    ]

    c.setFont("Helvetica", 9)
    for item_id, desc, qty, price, amount in items:
        y -= 18
        c.drawString(40, y, item_id)
        c.drawString(220, y, desc)
        c.drawString(345, y, qty)
        c.drawString(390, y, price)
        c.drawString(455, y, amount)

    # Totals
    y -= 30
    c.line(350, y + 10, 530, y + 10)
    c.setFont("Helvetica-Bold", 10)
    c.drawString(350, y, "Subtotal:")
    c.setFont("Helvetica", 10)
    c.drawRightString(525, y, "1,216.50")

    y -= 18
    c.setFont("Helvetica-Bold", 10)
    c.drawString(350, y, "VAT (20%):")
    c.setFont("Helvetica", 10)
    c.drawRightString(525, y, "243.30")

    y -= 22
    c.line(350, y + 12, 530, y + 12)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(350, y, "Total Due:")
    c.drawRightString(525, y, "£1,459.80")

    # Footer
    c.setFont("Helvetica", 8)
    c.drawString(40, 40, "Bank: Barclays | Sort Code: 20-45-67 | Account: 12345678 | IBAN: GB29NWBK60161331926819")

    c.save()
    return buf.getvalue()


def _create_po_pdf() -> bytes:
    """Create a realistic purchase order PDF."""
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w, h = A4

    c.setFont("Helvetica-Bold", 20)
    c.drawString(40, h - 50, "PURCHASE ORDER")

    c.setFont("Helvetica-Bold", 10)
    c.drawString(380, h - 50, "PO Number:")
    c.setFont("Helvetica", 10)
    c.drawString(460, h - 50, "PO-2025-1138")

    c.setFont("Helvetica-Bold", 10)
    c.drawString(380, h - 68, "Order Date:")
    c.setFont("Helvetica", 10)
    c.drawString(460, h - 68, "22/01/2025")

    c.setFont("Helvetica-Bold", 10)
    c.drawString(380, h - 86, "Delivery Date:")
    c.setFont("Helvetica", 10)
    c.drawString(460, h - 86, "28/02/2025")

    c.setFont("Helvetica-Bold", 10)
    c.drawString(380, h - 104, "Currency:")
    c.setFont("Helvetica", 10)
    c.drawString(460, h - 104, "GBP")

    # Vendor
    c.setFont("Helvetica-Bold", 10)
    c.drawString(40, h - 80, "Vendor:")
    c.setFont("Helvetica", 10)
    c.drawString(40, h - 95, "Henderson & Clarke Engineering Ltd")
    c.drawString(40, h - 107, "14 Caledonian Crescent")
    c.drawString(40, h - 119, "Edinburgh, EH1 2NG")

    # Ship to
    c.setFont("Helvetica-Bold", 10)
    c.drawString(40, h - 145, "Ship To:")
    c.setFont("Helvetica", 10)
    c.drawString(40, h - 160, "Global Dynamics Warehouse")
    c.drawString(40, h - 172, "Unit 3, Harbour Business Park")
    c.drawString(40, h - 184, "Southampton, SO14 3FE")

    c.setFont("Helvetica-Bold", 10)
    c.drawString(380, h - 125, "Payment Terms:")
    c.setFont("Helvetica", 10)
    c.drawString(480, h - 125, "Net 45")

    # Table
    y = h - 220
    c.setFont("Helvetica-Bold", 9)
    c.drawString(40, y, "Item ID")
    c.drawString(130, y, "Description")
    c.drawString(340, y, "Qty")
    c.drawString(380, y, "Unit Price")
    c.drawString(460, y, "Line Total")
    c.line(40, y - 5, 530, y - 5)

    items = [
        ("CMP-001", "Hydraulic Cylinder Assembly", "4", "875.00", "3,500.00"),
        ("FLT-045", "Air Filtration Module", "12", "64.50", "774.00"),
    ]
    c.setFont("Helvetica", 9)
    for item_id, desc, qty, price, total in items:
        y -= 18
        c.drawString(40, y, item_id)
        c.drawString(130, y, desc)
        c.drawString(345, y, qty)
        c.drawString(390, y, price)
        c.drawString(465, y, total)

    y -= 30
    c.line(380, y + 10, 530, y + 10)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(380, y, "Order Total:")
    c.drawRightString(525, y, "£4,274.00")

    c.save()
    return buf.getvalue()


def _create_quote_pdf() -> bytes:
    """Create a realistic quotation PDF."""
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w, h = A4

    c.setFont("Helvetica-Bold", 10)
    c.drawString(40, h - 40, "Northgate Technical Services Ltd")
    c.setFont("Helvetica", 9)
    c.drawString(40, h - 54, "88 Queen Street, Glasgow, G1 3DN")

    c.setFont("Helvetica-Bold", 20)
    c.drawString(380, h - 50, "QUOTATION")

    c.setFont("Helvetica-Bold", 10)
    c.drawString(380, h - 75, "Quote No:")
    c.setFont("Helvetica", 10)
    c.drawString(460, h - 75, "QUT-2025-0553")

    c.setFont("Helvetica-Bold", 10)
    c.drawString(380, h - 93, "Quote Date:")
    c.setFont("Helvetica", 10)
    c.drawString(460, h - 93, "10/02/2025")

    c.setFont("Helvetica-Bold", 10)
    c.drawString(380, h - 111, "Valid Until:")
    c.setFont("Helvetica", 10)
    c.drawString(460, h - 111, "10/03/2025")

    c.setFont("Helvetica-Bold", 10)
    c.drawString(380, h - 129, "Currency:")
    c.setFont("Helvetica", 10)
    c.drawString(460, h - 129, "GBP")

    # Customer
    c.setFont("Helvetica-Bold", 10)
    c.drawString(40, h - 85, "To:")
    c.setFont("Helvetica", 10)
    c.drawString(40, h - 100, "Whitfield & Sons Construction")
    c.drawString(40, h - 112, "Bristol, BS1 4DJ")

    # Table
    y = h - 165
    c.setFont("Helvetica-Bold", 9)
    c.drawString(40, y, "Description")
    c.drawString(300, y, "Qty")
    c.drawString(340, y, "Unit Price")
    c.drawString(420, y, "Line Total")
    c.line(40, y - 5, 530, y - 5)

    items = [
        ("Structural Steel Beam H200x200", "20", "189.00", "3,780.00"),
        ("Welding Consumables Kit", "5", "96.00", "480.00"),
        ("Anti-corrosion Coating (25L)", "8", "135.00", "1,080.00"),
    ]
    c.setFont("Helvetica", 9)
    for desc, qty, price, total in items:
        y -= 18
        c.drawString(40, y, desc)
        c.drawString(305, y, qty)
        c.drawString(350, y, price)
        c.drawString(425, y, total)

    y -= 30
    c.line(350, y + 10, 530, y + 10)
    c.setFont("Helvetica-Bold", 10)
    c.drawString(350, y, "Subtotal:")
    c.drawRightString(525, y, "5,340.00")

    y -= 18
    c.drawString(350, y, "VAT (20%):")
    c.setFont("Helvetica", 10)
    c.drawRightString(525, y, "1,068.00")

    y -= 20
    c.line(350, y + 10, 530, y + 10)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(350, y, "Total Amount:")
    c.drawRightString(525, y, "£6,408.00")

    c.save()
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Expected values for validation
# ---------------------------------------------------------------------------

EXPECTED = {
    "Invoice": {
        "invoice_id": "INV-2025-0042",
        "vendor_name": "Acme Industrial Supplies Ltd",
        "buyer_id": "Pinnacle Manufacturing Group",
        "invoice_date": date(2025, 3, 15),
        "due_date": date(2025, 4, 14),
        "po_id": "PO-7891",
        "currency": "GBP",
        "invoice_amount": 1216.50,
        "tax_percent": 20.0,
        "tax_amount": 243.30,
        "invoice_total_incl_tax": 1459.80,
        "payment_terms": "Net 30",
        "line_items_count": 3,
    },
    "Purchase_Order": {
        "po_id": "PO-2025-1138",
        "vendor_name": "Henderson & Clarke Engineering Ltd",
        "order_date": date(2025, 1, 22),
        "expected_delivery_date": date(2025, 2, 28),
        "currency": "GBP",
        "total_amount": 4274.00,
        "payment_terms": "Net 45",
        "line_items_count": 2,
    },
    "Quote": {
        "quote_id": "QUT-2025-0553",
        "supplier_id": "Northgate Technical Services Ltd",
        "buyer_id": "Whitfield & Sons Construction",
        "quote_date": date(2025, 2, 10),
        "validity_date": date(2025, 3, 10),
        "currency": "GBP",
        "total_amount": 5340.00,
        "tax_amount": 1068.00,
        "total_amount_incl_tax": 6408.00,
        "line_items_count": 3,
    },
}


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

def _check_field(header: dict, field: str, expected, tolerance: float = 0.02):
    """Compare an extracted field to its expected value."""
    actual = header.get(field)

    # Normalize for comparison
    if actual is None:
        return False, f"MISSING (expected {expected})"

    if isinstance(expected, date):
        if isinstance(actual, date):
            if actual == expected:
                return True, f"{actual}"
            return False, f"{actual} (expected {expected})"
        # String comparison
        actual_str = str(actual).strip()
        expected_str = expected.isoformat()
        if actual_str == expected_str:
            return True, f"{actual_str}"
        return False, f"{actual_str} (expected {expected_str})"

    if isinstance(expected, float):
        try:
            actual_num = float(actual)
        except (ValueError, TypeError):
            return False, f"{actual!r} (expected {expected}, not numeric)"
        if abs(actual_num - expected) <= tolerance:
            return True, f"{actual_num}"
        return False, f"{actual_num} (expected {expected})"

    # String comparison (case-insensitive, strip)
    actual_str = str(actual).strip()
    expected_str = str(expected).strip()
    if actual_str.lower() == expected_str.lower():
        return True, f"{actual_str}"
    # Partial match — extracted value contains expected
    if expected_str.lower() in actual_str.lower() or actual_str.lower() in expected_str.lower():
        return True, f"{actual_str} (~partial match)"
    return False, f"{actual_str} (expected {expected_str})"


def run_extraction(agent, pdf_bytes: bytes, doc_type: str, label: str):
    """Run extraction and compare to expected values."""
    print(f"\n{'='*70}")
    print(f" {label}: {doc_type}")
    print(f"{'='*70}")

    # Extract text from PDF
    text_bundle = agent._extract_text(pdf_bytes, f"test_{doc_type.lower()}.pdf")
    text = text_bundle.full_text
    print(f"  Extracted text length: {len(text)} chars")
    print(f"  First 200 chars: {text[:200]!r}")

    # Classify document type
    detected_type = agent._classify_doc_type(text)
    print(f"  Detected doc type: {detected_type}")

    # Run structured extraction
    start = time.time()
    result = agent._extract_structured_data(
        text=text,
        file_bytes=pdf_bytes,
        doc_type=doc_type,
        source_hint=f"test_{doc_type.lower()}.pdf",
    )
    elapsed = time.time() - start
    print(f"  Extraction time: {elapsed:.1f}s")

    header = result.header
    line_items = result.line_items

    # Print raw header
    print(f"\n  --- Extracted Header ({len(header)} fields) ---")
    for k, v in sorted(header.items()):
        if k.startswith("_"):
            continue
        print(f"    {k}: {v!r}")

    print(f"\n  --- Line Items ({len(line_items)} rows) ---")
    for i, item in enumerate(line_items[:5]):
        desc = item.get("item_description", item.get("description", "?"))
        qty = item.get("quantity", "?")
        price = item.get("unit_price", "?")
        total = item.get("line_amount", item.get("line_total", item.get("total_amount", "?")))
        print(f"    [{i+1}] {desc} | qty={qty} | price={price} | total={total}")

    # Validate against expected values
    expected = EXPECTED.get(doc_type, {})
    passed = 0
    failed = 0
    missing = 0

    print(f"\n  --- Validation ---")
    for field, exp_value in expected.items():
        if field == "line_items_count":
            actual_count = len(line_items)
            if actual_count == exp_value:
                print(f"    [PASS] line_items_count: {actual_count}")
                passed += 1
            else:
                print(f"    [FAIL] line_items_count: {actual_count} (expected {exp_value})")
                failed += 1
            continue

        ok, detail = _check_field(header, field, exp_value)
        if ok:
            print(f"    [PASS] {field}: {detail}")
            passed += 1
        else:
            if "MISSING" in detail:
                print(f"    [MISS] {field}: {detail}")
                missing += 1
            else:
                print(f"    [FAIL] {field}: {detail}")
                failed += 1

    total = passed + failed + missing
    print(f"\n  Results: {passed}/{total} passed, {failed} failed, {missing} missing")

    # Check confidence metadata
    confidence = header.get("_field_confidence", {})
    if confidence:
        avg_conf = sum(confidence.values()) / len(confidence)
        print(f"  Average field confidence: {avg_conf:.2f}")

    methods = header.get("_field_methods", {})
    if methods:
        all_methods = set()
        for m_list in methods.values():
            if isinstance(m_list, list):
                all_methods.update(m_list)
        print(f"  Extraction methods used: {sorted(all_methods)}")

    return passed, failed, missing


def main():
    print("=" * 70)
    print(" LIVE EXTRACTION TEST")
    print(" Testing schema-driven LLM extraction pipeline")
    print("=" * 70)

    # Check Ollama availability
    import requests
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        models = [m["name"] for m in resp.json().get("models", [])]
        print(f"\nOllama models available: {models}")
    except Exception as e:
        print(f"\nERROR: Ollama not reachable at localhost:11434: {e}")
        print("Start Ollama and try again.")
        sys.exit(1)

    # Check reportlab
    try:
        from reportlab.pdfgen import canvas
    except ImportError:
        print("\nERROR: reportlab not installed. Run: pip install reportlab")
        sys.exit(1)

    # Create the agent with minimal mock
    print("\nInitializing DataExtractionAgent...")

    # Build a lightweight agent_nick that connects to real Ollama
    from services.ollama_client import ollama_generate
    import numpy as np

    class MockEmbeddingModel:
        def encode(self, texts, **kwargs):
            return [np.zeros(384) for _ in texts]
        def get_sentence_embedding_dimension(self):
            return 384

    nick = SimpleNamespace(
        embedding_model=MockEmbeddingModel(),
        settings=SimpleNamespace(
            extraction_model="BeyondProcwise/AgentNick:latest",
            document_extraction_model="BeyondProcwise/AgentNick:latest",
            qdrant_collection_name="test_extraction",
            ollama_base_url="http://localhost:11434",
            llamaparse_api_key=None,
            llm_backend="ollama",
            embedding_model_name=None,
            local_primary_model="BeyondProcwise/AgentNick:latest",
            local_fallback_model="BeyondProcwise/AgentNick:latest",
        ),
        get_db_connection=lambda: None,
        get_agent_model=lambda name, fallback=None: "BeyondProcwise/AgentNick:latest",
        ollama_options=lambda: {
            "temperature": 0.1,
            "num_predict": 4096,
        },
        # Pre-cache the model list so _get_available_ollama_models doesn't
        # fall back to qwen2.5:32b
        _available_ollama_models=models,
        _available_ollama_models_ts=time.monotonic(),
    )

    agent = DataExtractionAgent(nick)
    print(f"  Extraction model: {agent.extraction_model}")

    # Generate test PDFs
    print("\nGenerating test PDFs...")
    invoice_pdf = _create_invoice_pdf()
    po_pdf = _create_po_pdf()
    quote_pdf = _create_quote_pdf()
    print(f"  Invoice PDF: {len(invoice_pdf):,} bytes")
    print(f"  PO PDF: {len(po_pdf):,} bytes")
    print(f"  Quote PDF: {len(quote_pdf):,} bytes")

    # Run extractions
    total_passed = 0
    total_failed = 0
    total_missing = 0

    for pdf_bytes, doc_type, label in [
        (invoice_pdf, "Invoice", "Test 1"),
        (po_pdf, "Purchase_Order", "Test 2"),
        (quote_pdf, "Quote", "Test 3"),
    ]:
        p, f, m = run_extraction(agent, pdf_bytes, doc_type, label)
        total_passed += p
        total_failed += f
        total_missing += m

    # Summary
    total = total_passed + total_failed + total_missing
    print(f"\n{'='*70}")
    print(f" OVERALL: {total_passed}/{total} passed, {total_failed} failed, {total_missing} missing")
    accuracy = (total_passed / total * 100) if total > 0 else 0
    print(f" Accuracy: {accuracy:.1f}%")
    print(f"{'='*70}")

    if total_failed > 0 or total_missing > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
