#!/usr/bin/env python3
"""Build comprehensive fine-tuning dataset for AgentNick.

Combines:
1. Auto-collected verified extractions (conf >= 0.90)
2. Existing QLoRA dataset (corrected for accuracy)
3. Multi-agent awareness training examples
4. Procurement domain knowledge examples

Output: data/training/overnight_finetune.jsonl (Alpaca format)
"""

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT = PROJECT_ROOT / "data" / "training" / "overnight_finetune.jsonl"

SYSTEM_PROMPT = (
    "You are AgentNick, the core AI engine for ProcWise — an enterprise procurement "
    "intelligence platform. You power 13 specialized agents across the procurement lifecycle: "
    "DataExtraction, SupplierRanking, QuoteEvaluation, QuoteComparison, OpportunityMiner, "
    "EmailDrafting, Negotiation, SupplierInteraction, EmailDispatch, EmailWatcher, "
    "Approvals, DiscrepancyDetection, and RAG. Your primary role is accurate document "
    "extraction — extracting EXACTLY what documents say, never modifying source values."
)


def load_auto_collected():
    """Load auto-collected verified extractions."""
    path = PROJECT_ROOT / "src" / "data" / "training" / "auto_collected_examples.jsonl"
    examples = []
    if not path.exists():
        return examples
    seen = set()
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            key = f"{ex.get('doc_type')}_{ex.get('pk')}"
            if key in seen:
                continue
            seen.add(key)

            doc_type = ex.get("doc_type", "")
            source = ex.get("source_text", "")
            extracted = ex.get("extracted", {})

            if not source or not extracted.get("header"):
                continue

            examples.append({
                "instruction": f"Extract all fields from this {doc_type} document. Return valid JSON with header and line_items.",
                "input": source[:4000],
                "output": json.dumps({"document_type": doc_type, **extracted}, default=str),
            })
    print(f"  Auto-collected: {len(examples)} examples")
    return examples


def load_existing_qlora():
    """Load existing QLoRA dataset."""
    path = PROJECT_ROOT / "data" / "training" / "qlora_dataset.json"
    if not path.exists():
        return []
    with open(path) as f:
        data = json.load(f)
    print(f"  Existing QLoRA: {len(data)} examples")
    return data


def generate_multi_agent_examples():
    """Generate training examples that teach AgentNick about all 13 agents."""
    examples = []

    # Agent awareness examples
    agent_examples = [
        {
            "instruction": "What agents are part of the ProcWise platform and what does each do?",
            "input": "",
            "output": json.dumps({
                "agents": [
                    {"name": "DataExtractionAgent", "role": "Extract structured data from procurement documents (invoices, POs, quotes, contracts) in any format (PDF, DOCX, Excel, images). This is the foundational first step — all downstream agents depend on accurate extraction."},
                    {"name": "SupplierRankingAgent", "role": "Rank and score suppliers using policy-weighted criteria: price competitiveness, quality metrics, delivery reliability, risk score, and ESG certifications."},
                    {"name": "QuoteEvaluationAgent", "role": "Evaluate individual supplier quotes against historical pricing data, policy thresholds, and business rules to determine value."},
                    {"name": "QuoteComparisonAgent", "role": "Normalize and compare multiple quotes side-by-side. Generate recommendations based on total cost of ownership."},
                    {"name": "OpportunityMinerAgent", "role": "Detect procurement anomalies and savings opportunities: duplicate vendors, maverick spend, contract consolidation, volume discount potential."},
                    {"name": "EmailDraftingAgent", "role": "Draft professional supplier communications: RFQs, negotiation messages, follow-ups, using context from ranking and opportunity findings."},
                    {"name": "NegotiationAgent", "role": "Conduct multi-round supplier negotiations (max 3 rounds) with strategy-aware messaging, concession tracking, and timeout management."},
                    {"name": "SupplierInteractionAgent", "role": "Gateway for all supplier workflows. Routes communications, normalizes responses, tracks interaction history."},
                    {"name": "EmailDispatchAgent", "role": "Send approved emails via Amazon SES. Record dispatch metadata, track delivery status."},
                    {"name": "EmailWatcherAgent", "role": "Monitor inbound supplier responses via IMAP. Map replies to active workflow threads and trigger downstream processing."},
                    {"name": "ApprovalsAgent", "role": "Determine approval decisions based on procurement policy thresholds, delegation of authority, and business rules."},
                    {"name": "DiscrepancyDetectionAgent", "role": "Detect data quality issues: field mismatches, validation errors, math inconsistencies, missing required fields in extracted data."},
                    {"name": "RAGAgent", "role": "Procurement knowledge Q&A using retrieval-augmented generation. Answers policy questions, provides sourcing guidance from the Qdrant vector store."},
                ]
            }),
        },
        {
            "instruction": "How does DataExtractionAgent output feed into SupplierRankingAgent?",
            "input": "",
            "output": "DataExtractionAgent extracts structured data from procurement documents and persists to bp_ tables (bp_invoice, bp_purchase_order, bp_quote). SupplierRankingAgent then queries these tables to: 1) Calculate spend per supplier from invoice totals, 2) Compare quoted prices across suppliers from quote data, 3) Assess delivery performance from PO expected vs actual dates, 4) Build a composite ranking score weighted by procurement policy criteria. Accurate extraction is critical — wrong amounts or misidentified suppliers would corrupt the entire ranking.",
        },
        {
            "instruction": "What happens when extracted data has quality issues?",
            "input": "",
            "output": "DiscrepancyDetectionAgent monitors extraction quality. It checks: 1) Math coherence — subtotal + tax should equal total, 2) Field completeness — required fields like invoice_id and supplier must be present, 3) Cross-reference validity — PO numbers should match existing POs, 4) Value reasonableness — tax_percent should be 0-30%, quantities should be positive. Discrepancies are logged to bp_discrepancy_data with severity (error/warning/info) but source values are NEVER modified. Human review is triggered for error-severity issues.",
        },
        {
            "instruction": "How does the procurement lifecycle flow through the agents?",
            "input": "",
            "output": "The procurement lifecycle flows: 1) CAPTURE: Documents uploaded to S3 → ProcessMonitorWatcher detects them. 2) EXTRACT: DataExtractionAgent extracts structured data → persists to bp_ tables → syncs to Knowledge Graph. 3) RANK: SupplierRankingAgent scores suppliers based on extracted data + policies. 4) EVALUATE: QuoteEvaluationAgent + QuoteComparisonAgent analyze quotes. 5) MINE: OpportunityMinerAgent finds savings opportunities. 6) NEGOTIATE: NegotiationAgent conducts supplier negotiations via EmailDraftingAgent. 7) APPROVE: ApprovalsAgent checks against policy thresholds. 8) DISPATCH: EmailDispatchAgent sends approved communications. 9) MONITOR: EmailWatcherAgent tracks responses, RAGAgent answers queries.",
        },
        {
            "instruction": "Why is extraction accuracy the highest priority?",
            "input": "",
            "output": "Extraction is the foundational first step in the procurement pipeline. Every downstream agent depends on it: SupplierRankingAgent needs accurate spend amounts to rank correctly. QuoteComparisonAgent needs precise line item prices for comparison. OpportunityMinerAgent needs correct supplier names to detect duplicates. NegotiationAgent needs accurate current pricing for negotiation strategy. ApprovalsAgent needs correct totals for threshold checks. If extraction gets a supplier name wrong, the entire ranking is invalid. If it misreads an amount, approvals apply wrong thresholds. Garbage in = garbage out across all 12 downstream agents.",
        },
    ]

    # Extraction accuracy training
    accuracy_examples = [
        {
            "instruction": "Extract all fields from this Invoice document. Return valid JSON with header and line_items.",
            "input": "INVOICE\nFrom: TechWorld Solutions Ltd\n20 Innovation Park, Reading RG1 3AD\nInvoice No: INV-2025-0847\nDate: 15 March 2025\nPO Reference: PO-507291\n\nBill To:\nAssurity Ltd\n10 Redkiln Way, Horsham RH13 5QH\n\nDescription | Qty | Unit Price | Amount\nIT Consulting Services (Senior) | 5 | £450.00 | £2,250.00\nSoftware License - Annual | 1 | £1,200.00 | £1,200.00\nTraining Workshop (half day) | 2 | £350.00 | £700.00\n\nSubtotal: £4,150.00\nVAT (20%): £830.00\nTotal Due: £4,980.00\n\nPayment Terms: Net 30\nBank: HSBC Business\nSort: 40-12-78\nAccount: 91827364",
            "output": json.dumps({
                "document_type": "Invoice",
                "header": {
                    "invoice_id": "INV-2025-0847",
                    "supplier_id": "TechWorld Solutions Ltd",
                    "po_id": "PO-507291",
                    "buyer_id": "Assurity Ltd",
                    "invoice_date": "2025-03-15",
                    "payment_terms": "Net 30",
                    "currency": "GBP",
                    "invoice_amount": 4150.00,
                    "tax_percent": 20,
                    "tax_amount": 830.00,
                    "invoice_total_incl_tax": 4980.00
                },
                "line_items": [
                    {"item_description": "IT Consulting Services (Senior)", "quantity": 5, "unit_price": 450.00, "line_amount": 2250.00},
                    {"item_description": "Software License - Annual", "quantity": 1, "unit_price": 1200.00, "line_amount": 1200.00},
                    {"item_description": "Training Workshop (half day)", "quantity": 2, "unit_price": 350.00, "line_amount": 700.00},
                ]
            }),
        },
        {
            "instruction": "Extract all fields from this Purchase_Order document. Return valid JSON with header and line_items.",
            "input": "PURCHASE ORDER\n\nPO Number: PO-526900\nDate: 1 April 2025\n\nVendor: Duncan Office Supplies LLC\n28 Victoria Works, Birmingham B7 4AX\n\nShip To: Assurity Ltd\n10 Redkiln Way, Horsham RH13 5QH\n\nDelivery Expected: 15 April 2025\nPayment Terms: Net 60\n\nItem | Qty | Unit Price | Total\nStaedtler Ballpoint Pen (Black, 10pk) | 50 | £12.49 | £624.50\n3M Adhesive Tape (19mm) | 30 | £8.99 | £269.70\nHP A4 Paper (500 sheets) | 100 | £4.50 | £450.00\n\nSubtotal: £1,344.20\nVAT (20%): £268.84\nTotal Including VAT: £1,613.04\n\nAuthorised by: Procurement Manager",
            "output": json.dumps({
                "document_type": "Purchase_Order",
                "header": {
                    "po_id": "PO-526900",
                    "supplier_name": "Duncan Office Supplies LLC",
                    "buyer_id": "Assurity Ltd",
                    "order_date": "2025-04-01",
                    "expected_delivery_date": "2025-04-15",
                    "payment_terms": "Net 60",
                    "currency": "GBP",
                    "total_amount": 1344.20,
                    "tax_amount": 268.84,
                    "total_amount_incl_tax": 1613.04
                },
                "line_items": [
                    {"item_description": "Staedtler Ballpoint Pen (Black, 10pk)", "quantity": 50, "unit_price": 12.49, "line_total": 624.50},
                    {"item_description": "3M Adhesive Tape (19mm)", "quantity": 30, "unit_price": 8.99, "line_total": 269.70},
                    {"item_description": "HP A4 Paper (500 sheets)", "quantity": 100, "unit_price": 4.50, "line_total": 450.00},
                ]
            }),
        },
        {
            "instruction": "Extract all fields from this Quote document. Return valid JSON with header and line_items.",
            "input": "QUOTATION\n\nQuote Ref: QTE-2026-00512\nDate: 18 July 2025\nValid Until: 18 August 2025\n\nFrom: SupplyX1 Ltd\n8th Floor, The Blade, Reading RG1 3BE\nTel: +44 (0)118 405 6723\n\nPrepared For:\nGreenfield Holdings Ltd\nFinance Department\n\nItem | Qty | Unit Price | Total\nErgonomic Office Chair (Mesh) | 20 | £289.00 | £5,780.00\nStanding Desk (Electric) | 10 | £549.00 | £5,490.00\nMonitor Arm (Dual) | 20 | £89.00 | £1,780.00\nCable Management Kit | 30 | £24.50 | £735.00\nDesk Lamp (LED, Adjustable) | 20 | £45.00 | £900.00\n\nSubtotal: £14,685.00\nVAT (20%): £2,937.00\nTotal: £17,622.00",
            "output": json.dumps({
                "document_type": "Quote",
                "header": {
                    "quote_id": "QTE-2026-00512",
                    "supplier_id": "SupplyX1 Ltd",
                    "buyer_id": "Greenfield Holdings Ltd",
                    "quote_date": "2025-07-18",
                    "validity_date": "2025-08-18",
                    "currency": "GBP",
                    "total_amount": 14685.00,
                    "tax_percent": 20,
                    "tax_amount": 2937.00,
                    "total_amount_incl_tax": 17622.00
                },
                "line_items": [
                    {"item_description": "Ergonomic Office Chair (Mesh)", "quantity": 20, "unit_price": 289.00, "line_total": 5780.00},
                    {"item_description": "Standing Desk (Electric)", "quantity": 10, "unit_price": 549.00, "line_total": 5490.00},
                    {"item_description": "Monitor Arm (Dual)", "quantity": 20, "unit_price": 89.00, "line_total": 1780.00},
                    {"item_description": "Cable Management Kit", "quantity": 30, "unit_price": 24.50, "line_total": 735.00},
                    {"item_description": "Desk Lamp (LED, Adjustable)", "quantity": 20, "unit_price": 45.00, "line_total": 900.00},
                ]
            }),
        },
    ]

    examples.extend(agent_examples)
    examples.extend(accuracy_examples)
    print(f"  Multi-agent + accuracy: {len(examples)} examples")
    return examples


def main():
    print("Building overnight fine-tuning dataset...")

    all_examples = []

    # 1. Auto-collected verified extractions
    all_examples.extend(load_auto_collected())

    # 2. Existing QLoRA dataset
    all_examples.extend(load_existing_qlora())

    # 3. Multi-agent awareness + accuracy examples
    all_examples.extend(generate_multi_agent_examples())

    # Convert to chat format (Alpaca style)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        for ex in all_examples:
            chat = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": ex.get("instruction", "") + ("\n\n" + ex["input"] if ex.get("input") else "")},
                    {"role": "assistant", "content": ex.get("output", "")},
                ]
            }
            f.write(json.dumps(chat, default=str) + "\n")

    print(f"\nTotal: {len(all_examples)} training examples → {OUTPUT}")
    print(f"File size: {OUTPUT.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
