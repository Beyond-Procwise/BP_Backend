"""Build comprehensive finetuning dataset for AgentNick extraction model.

Reads successful extractions from database, auto-collected examples,
and adds extraction pipeline rules as training examples.

Usage:
    cd /home/muthu/PycharmProjects/BP_Backend
    PYTHONPATH=src .venv/bin/python scripts/build_extraction_dataset.py
"""
import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "data" / "training" / "extraction_pipeline_dataset.jsonl"
AUTO_COLLECTED_PATH = PROJECT_ROOT / "src" / "data" / "training" / "auto_collected_examples.jsonl"

SYSTEM_PROMPT = (
    "You are AgentNick, the extraction engine for ProcWise. "
    "Extract structured data from procurement documents (Invoices, Purchase Orders, Quotes) with 100% accuracy.\n\n"
    "RULES:\n"
    "1. Extract EXACTLY what the document says — never invent or compute values\n"
    "2. The SUPPLIER created/sent the document (letterhead, top of document)\n"
    "3. The BUYER received the document (Bill To, Prepared For, Ship To)\n"
    "4. Extract ALL line items. STOP at subtotal/total/tax rows\n"
    "5. line_total: extract the ACTUAL value, do NOT compute qty × price\n"
    "6. Dates: YYYY-MM-DD. Amounts: numbers only. Currency: 3-letter ISO code\n"
    "7. tax_percent: the percentage NUMBER (20 for 20%)\n"
    "8. If a field is NOT in the document, OMIT it\n"
    "9. Deduplicate: if same line item appears twice (watermarks), extract ONLY ONCE\n"
    "10. item_id: extract product code/SKU if present, otherwise omit\n\n"
    "Return ONLY valid JSON: {\"header\": {...}, \"line_items\": [...]}"
)

# DB connection config
DB_CONFIG = {
    "host": "procwisemvpdb01.cluster-cpae0sg4mrk8.eu-west-1.rds.amazonaws.com",
    "dbname": "uicanvas",
    "user": "procwisedb123",
    "password": "Pr0cw!5edb001",
    "port": 5432,
}


def get_db_connection():
    import psycopg2
    return psycopg2.connect(**DB_CONFIG)


def build_conversation(doc_type: str, user_text: str, assistant_json: dict) -> dict:
    """Build a single training conversation."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Extract all fields from this {doc_type} document:\n{user_text}"},
            {"role": "assistant", "content": json.dumps(assistant_json, ensure_ascii=False)},
        ]
    }


def fetch_invoices(conn) -> list[dict]:
    """Fetch invoice headers + line items from the database."""
    examples = []
    cur = conn.cursor()

    # Get invoices that have line items
    cur.execute("""
        SELECT i.invoice_id, i.po_id, i.supplier_id, i.buyer_id, i.requisition_id,
               i.requested_by, i.requested_date, i.invoice_date, i.due_date,
               i.payment_terms, i.currency, i.invoice_amount, i.tax_percent,
               i.tax_amount, i.invoice_total_incl_tax, i.exchange_rate_to_usd,
               i.converted_amount_usd, i.country, i.region
        FROM proc.bp_invoice i
        WHERE EXISTS (
            SELECT 1 FROM proc.bp_invoice_line_items li WHERE li.invoice_id = i.invoice_id
        )
        ORDER BY i.invoice_date DESC NULLS LAST
        LIMIT 200
    """)
    invoice_cols = [desc[0] for desc in cur.description]
    invoices = [dict(zip(invoice_cols, row)) for row in cur.fetchall()]

    for inv in invoices:
        inv_id = inv["invoice_id"]
        cur.execute("""
            SELECT line_no, item_id, item_description, quantity, unit_of_measure,
                   unit_price, line_amount, tax_percent, tax_amount, total_amount_incl_tax,
                   po_id, delivery_date
            FROM proc.bp_invoice_line_items
            WHERE invoice_id = %s
            ORDER BY line_no
        """, (inv_id,))
        li_cols = [desc[0] for desc in cur.description]
        line_items = [dict(zip(li_cols, row)) for row in cur.fetchall()]

        if not line_items:
            continue

        # Build synthetic document text from the data
        header = {k: v for k, v in inv.items() if v is not None}
        clean_items = []
        for li in line_items:
            clean_li = {k: v for k, v in li.items() if v is not None}
            # Convert decimals
            for k, v in clean_li.items():
                if hasattr(v, "as_tuple"):
                    clean_li[k] = float(v)
            clean_items.append(clean_li)

        # Convert header decimals/dates
        for k, v in header.items():
            if hasattr(v, "as_tuple"):
                header[k] = float(v)
            elif hasattr(v, "isoformat"):
                header[k] = v.isoformat()

        # Build a synthetic document text representation
        doc_text = build_invoice_text(header, clean_items)

        # Clean header for output (remove None, convert types)
        output = {"header": header, "line_items": clean_items}
        examples.append(build_conversation("Invoice", doc_text, output))

    cur.close()
    logger.info(f"Fetched {len(examples)} invoice examples from DB")
    return examples


def build_invoice_text(header: dict, items: list) -> str:
    """Build synthetic invoice text from extracted data."""
    lines = ["INVOICE"]
    if header.get("supplier_id"):
        lines.append(f"From: {header['supplier_id']}")
    if header.get("invoice_id"):
        lines.append(f"Invoice No: {header['invoice_id']}")
    if header.get("invoice_date"):
        lines.append(f"Date: {header['invoice_date']}")
    if header.get("buyer_id"):
        lines.append(f"Bill To: {header['buyer_id']}")
    if header.get("po_id"):
        lines.append(f"PO Ref: {header['po_id']}")
    lines.append("")
    lines.append("Description                    Qty    Unit Price    Amount")
    lines.append("-" * 70)
    for i, item in enumerate(items, 1):
        desc = item.get("item_description", "Item")
        qty = item.get("quantity", "")
        price = item.get("unit_price", "")
        amount = item.get("line_amount", "")
        lines.append(f"{i}. {desc}    {qty}    {price}    {amount}")
    lines.append("")
    currency_sym = "£" if header.get("currency") == "GBP" else "$" if header.get("currency") == "USD" else "€"
    if header.get("invoice_amount"):
        lines.append(f"Subtotal: {currency_sym}{header['invoice_amount']}")
    if header.get("tax_percent") and header.get("tax_amount"):
        lines.append(f"VAT ({header['tax_percent']}%): {currency_sym}{header['tax_amount']}")
    if header.get("invoice_total_incl_tax"):
        lines.append(f"Total: {currency_sym}{header['invoice_total_incl_tax']}")
    if header.get("payment_terms"):
        lines.append(f"Payment Terms: {header['payment_terms']}")
    return "\n".join(lines)


def fetch_purchase_orders(conn) -> list[dict]:
    """Fetch PO headers + line items from the database."""
    examples = []
    cur = conn.cursor()

    cur.execute("""
        SELECT po.po_id, po.supplier_name, po.buyer_id, po.requisition_id,
               po.requested_by, po.requested_date, po.currency, po.order_date,
               po.expected_delivery_date, po.ship_to_country, po.delivery_region,
               po.incoterm, po.total_amount, po.delivery_address_line1,
               po.delivery_address_line2, po.delivery_city
        FROM proc.bp_purchase_order po
        WHERE EXISTS (
            SELECT 1 FROM proc.bp_po_line_items li WHERE li.po_id = po.po_id
        )
        ORDER BY po.order_date DESC NULLS LAST
        LIMIT 200
    """)
    po_cols = [desc[0] for desc in cur.description]
    pos = [dict(zip(po_cols, row)) for row in cur.fetchall()]

    for po in pos:
        po_id = po["po_id"]
        cur.execute("""
            SELECT line_number, item_id, item_description, quote_number, quantity,
                   unit_price, unit_of_measure, currency, line_total,
                   tax_percent, tax_amount, total_amount
            FROM proc.bp_po_line_items
            WHERE po_id = %s
            ORDER BY line_number
        """, (po_id,))
        li_cols = [desc[0] for desc in cur.description]
        line_items = [dict(zip(li_cols, row)) for row in cur.fetchall()]

        if not line_items:
            continue

        header = {k: v for k, v in po.items() if v is not None}
        clean_items = []
        for li in line_items:
            clean_li = {k: v for k, v in li.items() if v is not None}
            for k, v in clean_li.items():
                if hasattr(v, "as_tuple"):
                    clean_li[k] = float(v)
            clean_items.append(clean_li)

        for k, v in header.items():
            if hasattr(v, "as_tuple"):
                header[k] = float(v)
            elif hasattr(v, "isoformat"):
                header[k] = v.isoformat()

        doc_text = build_po_text(header, clean_items)
        output = {"header": header, "line_items": clean_items}
        examples.append(build_conversation("Purchase_Order", doc_text, output))

    cur.close()
    logger.info(f"Fetched {len(examples)} PO examples from DB")
    return examples


def build_po_text(header: dict, items: list) -> str:
    """Build synthetic PO text from extracted data."""
    lines = ["PURCHASE ORDER"]
    if header.get("supplier_name"):
        lines.append(f"Supplier: {header['supplier_name']}")
    if header.get("buyer_id"):
        lines.append(f"Buyer: {header['buyer_id']}")
    if header.get("po_id"):
        lines.append(f"PO Number: {header['po_id']}")
    if header.get("order_date"):
        lines.append(f"PO Date: {header['order_date']}")
    if header.get("requisition_id"):
        lines.append(f"Quote Ref: {header['requisition_id']}")
    lines.append("")
    lines.append("DESCRIPTION                    QTY    UNIT PRICE    LINE TOTAL")
    lines.append("-" * 70)
    for i, item in enumerate(items, 1):
        desc = item.get("item_description", "Item")
        qty = item.get("quantity", "")
        price = item.get("unit_price", "")
        total = item.get("line_total", "")
        lines.append(f"{i}. {desc} | Qty: {qty} | Price: {price} | Total: {total}")
    lines.append("")
    currency_sym = "£" if header.get("currency") == "GBP" else "$" if header.get("currency") == "USD" else "€"
    if header.get("total_amount"):
        lines.append(f"Total: {currency_sym}{header['total_amount']}")
    if header.get("delivery_address_line1"):
        lines.append(f"Deliver To: {header['delivery_address_line1']}")
    return "\n".join(lines)


def fetch_quotes(conn) -> list[dict]:
    """Fetch quote headers + line items from the database."""
    examples = []
    cur = conn.cursor()

    cur.execute("""
        SELECT q.quote_id, q.deal_id, q.supplier_id, q.buyer_id,
               q.supplier_address, q.buyer_address, q.quote_date,
               q.validity_date, q.currency, q.total_amount,
               q.tax_percent, q.tax_amount, q.total_amount_incl_tax,
               q.po_id, q.country, q.region
        FROM proc.bp_quote q
        WHERE EXISTS (
            SELECT 1 FROM proc.bp_quote_line_items li WHERE li.quote_id = q.quote_id
        )
        ORDER BY q.quote_date DESC NULLS LAST
        LIMIT 200
    """)
    q_cols = [desc[0] for desc in cur.description]
    quotes = [dict(zip(q_cols, row)) for row in cur.fetchall()]

    for q in quotes:
        q_id = q["quote_id"]
        cur.execute("""
            SELECT line_number, item_id, item_description, quantity,
                   unit_of_measure, unit_price, line_total,
                   tax_percent, tax_amount, total_amount, currency
            FROM proc.bp_quote_line_items
            WHERE quote_id = %s
            ORDER BY line_number
        """, (q_id,))
        li_cols = [desc[0] for desc in cur.description]
        line_items = [dict(zip(li_cols, row)) for row in cur.fetchall()]

        if not line_items:
            continue

        header = {k: v for k, v in q.items() if v is not None}
        clean_items = []
        for li in line_items:
            clean_li = {k: v for k, v in li.items() if v is not None}
            for k, v in clean_li.items():
                if hasattr(v, "as_tuple"):
                    clean_li[k] = float(v)
            clean_items.append(clean_li)

        for k, v in header.items():
            if hasattr(v, "as_tuple"):
                header[k] = float(v)
            elif hasattr(v, "isoformat"):
                header[k] = v.isoformat()

        doc_text = build_quote_text(header, clean_items)
        output = {"header": header, "line_items": clean_items}
        examples.append(build_conversation("Quote", doc_text, output))

    cur.close()
    logger.info(f"Fetched {len(examples)} quote examples from DB")
    return examples


def build_quote_text(header: dict, items: list) -> str:
    """Build synthetic quote text from extracted data."""
    lines = ["QUOTE"]
    if header.get("supplier_id"):
        lines.append(f"From: {header['supplier_id']}")
    if header.get("quote_id"):
        lines.append(f"Quote No: {header['quote_id']}")
    if header.get("quote_date"):
        lines.append(f"Date: {header['quote_date']}")
    if header.get("buyer_id"):
        lines.append(f"To: {header['buyer_id']}")
    if header.get("supplier_address"):
        lines.append(f"Supplier Address: {header['supplier_address']}")
    if header.get("buyer_address"):
        lines.append(f"Buyer Address: {header['buyer_address']}")
    if header.get("validity_date"):
        lines.append(f"Valid Until: {header['validity_date']}")
    lines.append("")
    lines.append("DESCRIPTION                    QTY    UNIT PRICE    LINE TOTAL")
    lines.append("-" * 70)
    for i, item in enumerate(items, 1):
        desc = item.get("item_description", "Item")
        qty = item.get("quantity", "")
        price = item.get("unit_price", "")
        total = item.get("line_total", "")
        lines.append(f"{i}. {desc} | Qty: {qty} | Price: {price} | Total: {total}")
    lines.append("")
    currency_sym = "£" if header.get("currency") == "GBP" else "$" if header.get("currency") == "USD" else "€"
    if header.get("total_amount"):
        lines.append(f"Subtotal: {currency_sym}{header['total_amount']}")
    if header.get("tax_percent") and header.get("tax_amount"):
        lines.append(f"VAT ({header['tax_percent']}%): {currency_sym}{header['tax_amount']}")
    if header.get("total_amount_incl_tax"):
        lines.append(f"Total: {currency_sym}{header['total_amount_incl_tax']}")
    return "\n".join(lines)


def build_pipeline_rules_examples() -> list[dict]:
    """Build training examples from extraction pipeline rules."""
    rules = []

    # Rule 1: Supplier vs Buyer identification
    rules.append(build_conversation(
        "Invoice",
        "INVOICE\nABC Consulting Ltd\n123 High Street, London EC1A 1BB\n\n"
        "Invoice No: INV-2024-100\nDate: 15/06/2024\n\n"
        "Bill To:\nGreenfield Holdings\n45 Business Park, Manchester M1 2AB\n\n"
        "1. Strategy Consulting (June 2024)    1    £5,000.00    £5,000.00\n\n"
        "Subtotal: £5,000.00\nVAT (20%): £1,000.00\nTotal: £6,000.00\n"
        "Payment Terms: Net 30",
        {
            "header": {
                "invoice_id": "INV-2024-100", "supplier_id": "ABC Consulting Ltd",
                "buyer_id": "Greenfield Holdings", "invoice_date": "2024-06-15",
                "payment_terms": "30", "currency": "GBP",
                "invoice_amount": 5000.00, "tax_percent": 20,
                "tax_amount": 1000.00, "invoice_total_incl_tax": 6000.00
            },
            "line_items": [
                {"line_no": 1, "item_description": "Strategy Consulting (June 2024)",
                 "quantity": 1, "unit_price": 5000.00, "line_amount": 5000.00}
            ]
        }
    ))

    # Rule 2: Excel structured data handling
    rules.append(build_conversation(
        "Quote",
        "QUOTE\n[Excel Metadata: Sheet='Quote', Company='SupplyX Ltd']\n"
        "Quote No: QTE-2026-00500\nDate: 2025-08-01\n"
        "From: SupplyX Ltd, 8th Floor, The Blade, Reading, RG1 3BE\n"
        "To: Greenfield Holdings Ltd, Meadow Lane, Reading, RG1 2AB\n"
        "Valid Until: 2025-09-01\n\n"
        "[TABLE]\n"
        "| Line | Item Code | Description | Qty | Unit Price | Line Total |\n"
        "| 1 | CHAIR-001 | Ergonomic Office Chair | 10 | 450.00 | 4500.00 |\n"
        "| 2 | DESK-002 | Standing Desk 160cm | 5 | 600.00 | 3000.00 |\n"
        "[/TABLE]\n\n"
        "[TOTALS]\nSubtotal: £7,500.00\nVAT (20%): £1,500.00\nTotal: £9,000.00",
        {
            "header": {
                "quote_id": "QTE-2026-00500", "supplier_id": "SupplyX Ltd",
                "buyer_id": "Greenfield Holdings Ltd",
                "supplier_address": "8th Floor, The Blade, Reading, RG1 3BE",
                "buyer_address": "Meadow Lane, Reading, RG1 2AB",
                "quote_date": "2025-08-01", "validity_date": "2025-09-01",
                "currency": "GBP", "total_amount": 7500.00,
                "tax_percent": 20, "tax_amount": 1500.00,
                "total_amount_incl_tax": 9000.00
            },
            "line_items": [
                {"line_number": 1, "item_id": "CHAIR-001",
                 "item_description": "Ergonomic Office Chair",
                 "quantity": 10, "unit_price": 450.00, "line_total": 4500.00},
                {"line_number": 2, "item_id": "DESK-002",
                 "item_description": "Standing Desk 160cm",
                 "quantity": 5, "unit_price": 600.00, "line_total": 3000.00}
            ]
        }
    ))

    # Rule 3: DOCX with split tables / watermark deduplication
    rules.append(build_conversation(
        "Quote",
        "QUOTE\nDell Workspace Solutions Ltd\nQuote No: WSG100050\nDate: 2024-05-20\n"
        "To: Kaiser LLC\n\n"
        "--- Page 1 Table ---\n"
        "1. Knoll Generation Task Chair | Qty: 12 | Price: £760.00 | Total: £9,120.00\n"
        "2. Herman Miller Mirra 2 Chair | Qty: 10 | Price: £910.00 | Total: £9,100.00\n\n"
        "--- Page 2 Table (continued) ---\n"
        "3. IKEA Alex Drawer Unit | Qty: 14 | Price: £90.00 | Total: £1,260.00\n\n"
        "--- WATERMARK TABLE (duplicate) ---\n"
        "1. Knoll Generation Task Chair | Qty: 12 | Price: £760.00 | Total: £9,120.00\n"
        "2. Herman Miller Mirra 2 Chair | Qty: 10 | Price: £910.00 | Total: £9,100.00\n"
        "3. IKEA Alex Drawer Unit | Qty: 14 | Price: £90.00 | Total: £1,260.00\n\n"
        "Subtotal: £19,480.00\nVAT (20%): £3,896.00\nTotal: £23,376.00",
        {
            "header": {
                "quote_id": "WSG100050", "supplier_id": "Dell Workspace Solutions Ltd",
                "buyer_id": "Kaiser LLC", "quote_date": "2024-05-20",
                "currency": "GBP", "total_amount": 19480.00,
                "tax_percent": 20, "tax_amount": 3896.00,
                "total_amount_incl_tax": 23376.00
            },
            "line_items": [
                {"line_number": 1, "item_description": "Knoll Generation Task Chair",
                 "quantity": 12, "unit_price": 760.00, "line_total": 9120.00},
                {"line_number": 2, "item_description": "Herman Miller Mirra 2 Chair",
                 "quantity": 10, "unit_price": 910.00, "line_total": 9100.00},
                {"line_number": 3, "item_description": "IKEA Alex Drawer Unit",
                 "quantity": 14, "unit_price": 90.00, "line_total": 1260.00}
            ]
        }
    ))

    # Rule 4: OCR multi-line descriptions
    rules.append(build_conversation(
        "Invoice",
        "INVOICE\nEleanor Price Creative Studio\nInvoice No: EPC-2024-045\nDate: 10/02/2024\n"
        "Bill To: Assurity Ltd\nPO Ref: PO505100\n\n"
        "1. Brand Identity Design Package\n"
        "   Including logo design, brand guidelines,\n"
        "   colour palette and typography    1    £3,500.00    £3,500.00\n"
        "2. Social Media Template Pack\n"
        "   (Instagram, LinkedIn, Facebook)    1    £1,200.00    £1,200.00\n\n"
        "Subtotal: £4,700.00\nVAT (20%): £940.00\nTotal: £5,640.00\nTerms: Net 30",
        {
            "header": {
                "invoice_id": "EPC-2024-045", "supplier_id": "Eleanor Price Creative Studio",
                "buyer_id": "Assurity Ltd", "po_id": "PO505100",
                "invoice_date": "2024-02-10", "payment_terms": "30",
                "currency": "GBP", "invoice_amount": 4700.00,
                "tax_percent": 20, "tax_amount": 940.00,
                "invoice_total_incl_tax": 5640.00
            },
            "line_items": [
                {"line_no": 1,
                 "item_description": "Brand Identity Design Package Including logo design, brand guidelines, colour palette and typography",
                 "quantity": 1, "unit_price": 3500.00, "line_amount": 3500.00},
                {"line_no": 2,
                 "item_description": "Social Media Template Pack (Instagram, LinkedIn, Facebook)",
                 "quantity": 1, "unit_price": 1200.00, "line_amount": 1200.00}
            ]
        }
    ))

    # Rule 5: "Extract don't compute" rule
    rules.append(build_conversation(
        "Purchase_Order",
        "PURCHASE ORDER\nAssurity Ltd\nPO Number: 530001\nDate: 01/03/2025\n"
        "Supplier: Wade Inc\n\n"
        "1. Staedtler Ballpoint Pen Black Ink, 10 per Pack | Qty: 50 | Price: £12.50\n"
        "2. 3M Adhesive Tape 19mm x 33m | Qty: 20 | Price: £8.99 | Total: £179.80\n\n"
        "Sub-total: £804.80\nTax (20%): £160.96\nTotal: £965.76",
        {
            "header": {
                "po_id": "530001", "supplier_name": "Wade Inc",
                "buyer_id": "Assurity Ltd", "order_date": "2025-03-01",
                "currency": "GBP", "total_amount": 965.76
            },
            "line_items": [
                {"line_number": 1,
                 "item_description": "Staedtler Ballpoint Pen Black Ink, 10 per Pack",
                 "quantity": 50, "unit_price": 12.50},
                {"line_number": 2,
                 "item_description": "3M Adhesive Tape 19mm x 33m",
                 "quantity": 20, "unit_price": 8.99, "line_total": 179.80}
            ]
        }
    ))

    # Rule 6: Product catalog / item_id from SKU
    rules.append(build_conversation(
        "Quote",
        "QUOTE\nGomez, Good and Cross Trading Ltd\nQuote No: QUT110500\nDate: 2025-02-15\n"
        "To: Assurity Ltd\n\n"
        "| # | SKU | Description | Qty | Price | Total |\n"
        "| 1 | DELL-XPS15-001 | Dell XPS 15 Laptop i7 16GB | 3 | £1,450.00 | £4,350.00 |\n"
        "| 2 | LOG-MXK-002 | Logitech MX Keys Keyboard | 3 | £109.99 | £329.97 |\n"
        "| 3 | MON-27-003 | Dell 27\" UltraSharp Monitor | 3 | £520.00 | £1,560.00 |\n\n"
        "Subtotal: £6,239.97\nVAT (20%): £1,248.00\nTotal: £7,487.97",
        {
            "header": {
                "quote_id": "QUT110500",
                "supplier_id": "Gomez, Good and Cross Trading Ltd",
                "buyer_id": "Assurity Ltd", "quote_date": "2025-02-15",
                "currency": "GBP", "total_amount": 6239.97,
                "tax_percent": 20, "tax_amount": 1248.00,
                "total_amount_incl_tax": 7487.97
            },
            "line_items": [
                {"line_number": 1, "item_id": "DELL-XPS15-001",
                 "item_description": "Dell XPS 15 Laptop i7 16GB",
                 "quantity": 3, "unit_price": 1450.00, "line_total": 4350.00},
                {"line_number": 2, "item_id": "LOG-MXK-002",
                 "item_description": "Logitech MX Keys Keyboard",
                 "quantity": 3, "unit_price": 109.99, "line_total": 329.97},
                {"line_number": 3, "item_id": "MON-27-003",
                 "item_description": "Dell 27\" UltraSharp Monitor",
                 "quantity": 3, "unit_price": 520.00, "line_total": 1560.00}
            ]
        }
    ))

    # Rule 7: Stop at subtotal rows (don't extract "Sub Total" as line item)
    rules.append(build_conversation(
        "Quote",
        "QUOTE\nTechNova Ltd\nQuote No: 103100\nDate: 2024-03-15\nTo: Assurity Ltd\n\n"
        "1. Acer TravelMate P2 Intel i5 8GB 512GB SSD | Qty: 5 | Price: £620.03 | Total: £3,100.15\n"
        "2. Lenovo ThinkPad L14 | Qty: 3 | Price: £750.00 | Total: £2,250.00\n"
        "Sub Total: £5,350.15\n"
        "Tax 20%: £1,070.03\n"
        "Grand Total: £6,420.18",
        {
            "header": {
                "quote_id": "103100", "supplier_id": "TechNova Ltd",
                "buyer_id": "Assurity Ltd", "quote_date": "2024-03-15",
                "currency": "GBP", "total_amount": 5350.15,
                "tax_percent": 20, "tax_amount": 1070.03,
                "total_amount_incl_tax": 6420.18
            },
            "line_items": [
                {"line_number": 1, "item_description": "Acer TravelMate P2 Intel i5 8GB 512GB SSD",
                 "quantity": 5, "unit_price": 620.03, "line_total": 3100.15},
                {"line_number": 2, "item_description": "Lenovo ThinkPad L14",
                 "quantity": 3, "unit_price": 750.00, "line_total": 2250.00}
            ]
        }
    ))

    # Rule 8: Correct quantity/price (don't swap them)
    rules.append(build_conversation(
        "Purchase_Order",
        "PURCHASE ORDER\nGomez, Good and Cross Trading Ltd\nPO Number: 520050\nDate: 2025-01-10\n\n"
        "1. Logitech MX Master 3S Advanced Wireless Mouse | Qty: 4 | Price: £89.99\n"
        "2. Logitech MX Keys Wireless Keyboard | Qty: 4 | Price: £109.99\n"
        "3. Dell 24\" Monitor P2422H | Qty: 2 | Price: £199.00\n\n"
        "Total: £1,197.92",
        {
            "header": {
                "po_id": "520050",
                "supplier_name": "Gomez, Good and Cross Trading Ltd",
                "order_date": "2025-01-10", "total_amount": 1197.92,
                "currency": "GBP"
            },
            "line_items": [
                {"line_number": 1,
                 "item_description": "Logitech MX Master 3S Advanced Wireless Mouse",
                 "quantity": 4, "unit_price": 89.99},
                {"line_number": 2,
                 "item_description": "Logitech MX Keys Wireless Keyboard",
                 "quantity": 4, "unit_price": 109.99},
                {"line_number": 3,
                 "item_description": "Dell 24\" Monitor P2422H",
                 "quantity": 2, "unit_price": 199.00}
            ]
        }
    ))

    # Rule 9: European format amounts
    rules.append(build_conversation(
        "Invoice",
        "RECHNUNG\nTechGmbH Solutions\nRechnungsnr.: R-2024-0789\nDatum: 15.06.2024\n"
        "An: Greenfield Holdings GmbH\n\n"
        "1. IT-Beratung (40 Stunden)    40    €125,00    €5.000,00\n"
        "2. Cloud-Migration Setup    1    €2.500,00    €2.500,00\n\n"
        "Zwischensumme: €7.500,00\nMwSt (19%): €1.425,00\nGesamtbetrag: €8.925,00\n"
        "Zahlungsbedingungen: Net 30",
        {
            "header": {
                "invoice_id": "R-2024-0789", "supplier_id": "TechGmbH Solutions",
                "buyer_id": "Greenfield Holdings GmbH", "invoice_date": "2024-06-15",
                "payment_terms": "30", "currency": "EUR",
                "invoice_amount": 7500.00, "tax_percent": 19,
                "tax_amount": 1425.00, "invoice_total_incl_tax": 8925.00
            },
            "line_items": [
                {"line_no": 1, "item_description": "IT-Beratung (40 Stunden)",
                 "quantity": 40, "unit_price": 125.00, "line_amount": 5000.00},
                {"line_no": 2, "item_description": "Cloud-Migration Setup",
                 "quantity": 1, "unit_price": 2500.00, "line_amount": 2500.00}
            ]
        }
    ))

    # Rule 10: US Invoice with sales tax
    rules.append(build_conversation(
        "Invoice",
        "INVOICE\nThrive Studios LLC\n123 Main Street, New York, NY 10001\n\n"
        "Invoice #: TSI-2024-200\nDate: 03/15/2024\n"
        "Bill To: Assurity Ltd, 10 Redkiln Way, Horsham, RH13 5QH\n"
        "PO Ref: PO502001\n\n"
        "Monthly Standard Marketing Package\n"
        "(Social campaigns, 8 posts, basic SEO)    1    $2,600.00    $2,600.00\n\n"
        "Subtotal: $2,600.00\nTax (8.875%): $230.75\nTotal: $2,830.75\n"
        "Terms: Net 30",
        {
            "header": {
                "invoice_id": "TSI-2024-200", "supplier_id": "Thrive Studios LLC",
                "buyer_id": "Assurity Ltd", "po_id": "PO502001",
                "invoice_date": "2024-03-15", "payment_terms": "30",
                "currency": "USD", "invoice_amount": 2600.00,
                "tax_percent": 8.875, "tax_amount": 230.75,
                "invoice_total_incl_tax": 2830.75
            },
            "line_items": [
                {"line_no": 1,
                 "item_description": "Monthly Standard Marketing Package (Social campaigns, 8 posts, basic SEO)",
                 "quantity": 1, "unit_price": 2600.00, "line_amount": 2600.00}
            ]
        }
    ))

    logger.info(f"Generated {len(rules)} pipeline rules examples")
    return rules


def load_auto_collected_examples() -> list[dict]:
    """Load and convert auto-collected examples to training format."""
    examples = []
    if not AUTO_COLLECTED_PATH.exists():
        logger.warning(f"Auto-collected examples not found at {AUTO_COLLECTED_PATH}")
        return examples

    seen_keys = set()
    with open(AUTO_COLLECTED_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
            except json.JSONDecodeError:
                continue

            doc_type = ex.get("doc_type", "")
            pk = ex.get("pk", "")
            extracted = ex.get("extracted", {})

            if not extracted or not doc_type:
                continue

            # Deduplicate by (doc_type, pk)
            dedup_key = (doc_type, pk)
            if dedup_key in seen_keys:
                continue
            seen_keys.add(dedup_key)

            header = extracted.get("header", {})
            line_items = extracted.get("line_items", [])

            if not header:
                continue

            # Deduplicate line items within the example
            clean_items = deduplicate_line_items(line_items)

            # Build document text from extracted data
            if doc_type == "Invoice":
                doc_text = build_invoice_text(header, clean_items)
            elif doc_type == "Purchase_Order":
                doc_text = build_po_text(header, clean_items)
            elif doc_type == "Quote":
                doc_text = build_quote_text(header, clean_items)
            else:
                continue

            output = {"header": header, "line_items": clean_items}
            examples.append(build_conversation(doc_type, doc_text, output))

    logger.info(f"Loaded {len(examples)} auto-collected examples (deduped from {AUTO_COLLECTED_PATH})")
    return examples


def deduplicate_line_items(items: list) -> list:
    """Remove duplicate line items (from watermarks/split tables)."""
    seen = set()
    unique = []
    for item in items:
        # Use description + quantity as dedup key
        desc = item.get("item_description", "")
        qty = item.get("quantity", "")
        key = (desc, str(qty))
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def main():
    logger.info("Building extraction pipeline dataset for AgentNick finetuning")

    all_examples = []

    # 1. Fetch from database
    try:
        conn = get_db_connection()
        logger.info("Connected to database")

        invoices = fetch_invoices(conn)
        pos = fetch_purchase_orders(conn)
        quotes = fetch_quotes(conn)
        all_examples.extend(invoices)
        all_examples.extend(pos)
        all_examples.extend(quotes)

        conn.close()
        logger.info(f"Total DB examples: {len(invoices)} invoices + {len(pos)} POs + {len(quotes)} quotes = {len(invoices) + len(pos) + len(quotes)}")
    except Exception as e:
        logger.error(f"Database error: {e}")
        logger.info("Continuing without DB examples...")

    # 2. Add pipeline rules examples
    rules = build_pipeline_rules_examples()
    all_examples.extend(rules)

    # 3. Load auto-collected examples
    auto = load_auto_collected_examples()
    all_examples.extend(auto)

    # Save
    os.makedirs(OUTPUT_PATH.parent, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        for ex in all_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    logger.info(f"Dataset saved to {OUTPUT_PATH}")
    logger.info(f"Total training examples: {len(all_examples)}")
    logger.info(f"  - DB examples: {len(all_examples) - len(rules) - len(auto)}")
    logger.info(f"  - Pipeline rules: {len(rules)}")
    logger.info(f"  - Auto-collected: {len(auto)}")


if __name__ == "__main__":
    main()
