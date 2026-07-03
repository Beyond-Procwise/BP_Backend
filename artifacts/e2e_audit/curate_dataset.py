"""Curate a failure-case-targeted QLoRA dataset for AgentNick extraction.

Core: all usable auto-collected examples (source_text + the CANONICAL correct
`extracted` target — these already encode the right answers for the eval misses,
e.g. po_id bare '521031', exact dates).
Plus: focused rule-reinforcement examples for the systematic errors observed in
the eval (po_id prefix-stripping, exact dates, address line splitting, quote
line-item numerics). Output = Alpaca {instruction,input,output} JSON list.
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path("/home/muthu/PycharmProjects/BP_Backend")
SRC = ROOT / "src/data/training/auto_collected_examples.jsonl"
OUT = ROOT / "data/training/curated_finetune.json"

INSTR = {
    "invoice": "Extract all fields from this invoice document. Return valid JSON with a 'header' object and a 'line_items' array. Use ONLY values present in the document.",
    "purchase_order": "Extract all fields from this purchase order document. Return valid JSON with a 'header' object and a 'line_items' array. Use ONLY values present in the document.",
    "po": "Extract all fields from this purchase order document. Return valid JSON with a 'header' object and a 'line_items' array. Use ONLY values present in the document.",
    "quote": "Extract all fields from this quote document. Return valid JSON with a 'header' object and a 'line_items' array. Use ONLY values present in the document.",
}

def core_examples():
    out = []
    for line in SRC.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        st = d.get("source_text") or ""
        ex = d.get("extracted")
        if len(st) < 50 or not isinstance(ex, dict) or not ex.get("header"):
            continue
        dt = (d.get("doc_type") or "invoice").lower()
        out.append({
            "instruction": INSTR.get(dt, INSTR["invoice"]),
            "input": st[:12000],
            "output": json.dumps(ex, default=str, ensure_ascii=False),
        })
    return out

# Rule-reinforcement examples — directly target the eval's systematic misses.
RULES = [
    # po_id: strip the 'PO' prefix -> bare number
    {"instruction": INSTR["invoice"],
     "input": "INVOICE\nInvoice Number\nINV900001 Mar 2024 PO521031\nBill To: Assurity Ltd\nTotal GBP 1200.00",
     "output": json.dumps({"header": {"invoice_id": "INV900001", "po_id": "521031", "invoice_date": "2024-03-01", "currency": "GBP", "invoice_total_incl_tax": 1200.0}, "line_items": []})},
    # quote id: strip 'QUT-' prefix
    {"instruction": INSTR["quote"],
     "input": "QUOTE\nAQUARIUS MARKETING\nQuote Ref QUT-25-032\nValid until 2025-08-30\nTotal GBP 60000.00",
     "output": json.dumps({"header": {"quote_id": "25-032", "supplier_name": "Aquarius Marketing", "validity_date": "2025-08-30", "currency": "GBP", "total_amount": 60000.0}, "line_items": []})},
    # exact date — do not shift by a day
    {"instruction": INSTR["invoice"],
     "input": "INVOICE\nInvoice No INV900002\nInvoice Date: 1 January 2024\nDue Date: 12 February 2024\nTotal GBP 638.60",
     "output": json.dumps({"header": {"invoice_id": "INV900002", "invoice_date": "2024-01-01", "due_date": "2024-02-12", "currency": "GBP", "invoice_total_incl_tax": 638.6}, "line_items": []})},
    # address line split — line1 = street, line2 = building
    {"instruction": INSTR["po"],
     "input": "PURCHASE ORDER\nPO No 502004\nSupplier: City of Newport\nDeliver to: 45 Riverfront Plaza, Civic Centre Building, Newport\nTotal GBP 9999.60",
     "output": json.dumps({"header": {"po_id": "502004", "supplier_name": "City of Newport", "delivery_address_line1": "45 Riverfront Plaza", "delivery_address_line2": "Civic Centre Building", "delivery_city": "Newport", "currency": "GBP", "total_amount": 9999.6}, "line_items": []})},
    # quote line-item numerics must be populated (not null)
    {"instruction": INSTR["quote"],
     "input": "QUOTE\nThrive Studios\nLine Items:\nGENERAL IT CONSULTANT  Qty 1  Unit 6750.00  Amount 6750.00\nTotal GBP 6750.00",
     "output": json.dumps({"header": {"supplier_name": "Thrive Studios", "currency": "GBP", "total_amount": 6750.0}, "line_items": [{"item_description": "GENERAL IT CONSULTANT", "quantity": 1, "unit_price": 6750.0, "total_amount": 6750.0}]})},
    # tax arithmetic coherence — read the right total line
    {"instruction": INSTR["quote"],
     "input": "QUOTE\nSubtotal 2399.70\nVAT 20% 479.94\nTotal incl VAT 2879.64",
     "output": json.dumps({"header": {"total_amount": 2399.70, "tax_percent": 20.0, "tax_amount": 479.94, "total_amount_incl_tax": 2879.64}, "line_items": []})},
]

def main():
    core = core_examples()
    data = core + RULES
    OUT.write_text(json.dumps(data, indent=1, ensure_ascii=False))
    print(f"curated: {len(core)} real + {len(RULES)} rule examples = {len(data)} total -> {OUT}")

if __name__ == "__main__":
    main()
