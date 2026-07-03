"""ProcWise-product knowledge training examples for AgentNick fine-tune.

This module contributes the *product-awareness* slice of the daily
fine-tune dataset. The goal is to make AgentNick understand ProcWise as
a complete procurement intelligence platform — not just a single
extraction skill.

Each example is one (instruction, input, output) triple in the same
Alpaca format used by build_finetune_dataset.py. The set covers:

  1. Agent-by-agent capability deep dives (13 agents).
  2. Data model mapping (which bp_* table receives which field).
  3. Extraction discipline (no-fabrication-NULL-when-absent rules
     observed during live production runs on 2026-05-21).
  4. Layout patterns AgentNick must handle (structural tables,
     markdown pipe-tables, paragraph-layout single-item docs).
  5. Cross-agent workflow context.
  6. Negative examples — what NOT to do (fabrication, prefix mutation).

The examples are deliberately concrete (real schema field names, real
table names, real workflows) so the fine-tuned model gains operational
fluency rather than vague awareness.
"""
from __future__ import annotations

import json
from typing import Any


def _ex(instruction: str, output: Any, input_text: str = "") -> dict:
    """Build one Alpaca-style example."""
    return {
        "instruction": instruction,
        "input": input_text,
        "output": output if isinstance(output, str) else json.dumps(output, default=str),
    }


# ---------------------------------------------------------------------------
# 1. Per-agent capability deep dives (13 entries — one per agent)
# ---------------------------------------------------------------------------
_AGENT_DEEP_DIVES = [
    _ex(
        "Explain DataExtractionAgent's responsibility, inputs, outputs, and downstream dependents.",
        (
            "DataExtractionAgent is ProcWise's foundational ingestion agent. It runs the L0→L1→L2→L3 "
            "single-flow renovation pipeline:\n\n"
            "- L0 (parser): Docling for native PDF/DOCX; PaddleOCR with Donut fallback for scanned PDFs and images. "
            "Produces a ParsedDocument with full_text, tables, page tokens.\n"
            "- L1 (regex): PatternRegistry-driven extraction of header fields (invoice_id, dates, amounts, currency).\n"
            "- L2 (engineered): table_extractor for line items, NER gap-fill, structural fallbacks "
            "(markdown pipe-tables, paragraph-layout heading+amount).\n"
            "- L3 (judge + context_layer): Qwen2.5-VL grounded-last-resort for remaining gaps, then "
            "AgentNick (BeyondProcwise/AgentNick:extract) context_layer synthesises a clean stg row.\n\n"
            "Persistence: per-doc INSERT into proc.bp_<doc_type>_raw + provenance_v3 rows, then promote() "
            "copies to proc.bp_<doc_type>_stg. Line items land in proc.bp_<doc_type>_line_items_stg.\n\n"
            "Downstream agents that depend on DataExtractionAgent: SupplierRankingAgent, QuoteEvaluationAgent, "
            "QuoteComparisonAgent, OpportunityMinerAgent, ApprovalsAgent, DiscrepancyDetectionAgent. "
            "Accuracy here is non-negotiable — any field error propagates to every downstream calculation."
        ),
    ),
    _ex(
        "Explain SupplierRankingAgent's algorithm and how it consumes extracted data.",
        (
            "SupplierRankingAgent scores suppliers using policy-weighted criteria. It reads from "
            "proc.bp_invoice_stg (spend), proc.bp_purchase_order_stg (delivery), proc.bp_quote_stg "
            "(price competitiveness) and proc.bp_supplier (master data).\n\n"
            "Weighted dimensions (weights come from policy_engine):\n"
            "- Price competitiveness: median quoted unit_price vs market\n"
            "- Delivery reliability: expected_delivery_date vs actual receipt\n"
            "- Quality: discrepancy_count from bp_extraction_discrepancy per supplier\n"
            "- Risk: TPRM score from risk_intelligence_service\n"
            "- ESG: certifications, region, financial stability\n\n"
            "Output: ranked list with composite score + justification. Used by EmailDraftingAgent to "
            "select shortlists for RFQs and by NegotiationAgent to anchor BATNAs."
        ),
    ),
    _ex(
        "Explain QuoteEvaluationAgent.",
        (
            "QuoteEvaluationAgent evaluates one supplier quote against benchmarks. It pulls the quote "
            "header from proc.bp_quote_stg and lines from proc.bp_quote_line_items_stg, then:\n"
            "1. Compares quoted total to historical median for the same supplier+category.\n"
            "2. Checks tax + sum invariants: subtotal × tax_percent ≈ tax_amount; subtotal + tax ≈ total_incl_tax.\n"
            "3. Verifies validity_date is in the future.\n"
            "4. Cross-references quote_id with active RFQs.\n\n"
            "Returns a per-quote score (0-100) + flags. Wrong unit_price extraction here causes "
            "false competitive ranking; wrong currency causes wrong USD conversion."
        ),
    ),
    _ex(
        "Explain QuoteComparisonAgent.",
        (
            "QuoteComparisonAgent normalises multiple quotes for the same scope into a comparable view. "
            "Reads proc.bp_quote_stg + proc.bp_quote_line_items_stg joined to proc.bp_supplier. "
            "Normalises currencies via converted_amount_usd, aligns line items by canonical product "
            "(item_id when available, fuzzy match on item_description otherwise), then emits a "
            "side-by-side matrix with totals, delivery terms, payment terms and recommended winner."
        ),
    ),
    _ex(
        "Explain OpportunityMinerAgent.",
        (
            "OpportunityMinerAgent finds savings opportunities by scanning proc.bp_invoice_stg and "
            "proc.bp_purchase_order_stg history:\n"
            "- Duplicate suppliers (same VAT/registration number, different supplier_id).\n"
            "- Maverick spend (invoices without a referenced PO).\n"
            "- Volume discount potential (suppliers > 80% of category spend without negotiated rate).\n"
            "- Contract consolidation (multiple POs to same supplier within 30 days).\n\n"
            "Outputs go into proc.procurement_patterns and surface as actionable findings."
        ),
    ),
    _ex(
        "Explain EmailDraftingAgent.",
        (
            "EmailDraftingAgent composes professional supplier communications: RFQs, negotiation messages, "
            "follow-ups. It conditions on ranking output, opportunity findings, and the active workflow's "
            "stage. Uses ProcWise's prompt_engine for template+context; never includes internal data "
            "(rankings, BATNAs, policy thresholds) in the outbound copy."
        ),
    ),
    _ex(
        "Explain NegotiationAgent.",
        (
            "NegotiationAgent runs multi-round negotiations capped at 3 rounds. Strategy comes from "
            "negotiation_strategy_engine (anchor low, mirror, walk-away). State persists in "
            "proc.workflow_email_tracking and negotiation_session. Each round emits drafts via "
            "EmailDraftingAgent → reviewed via ApprovalsAgent → dispatched via EmailDispatchAgent."
        ),
    ),
    _ex(
        "Explain SupplierInteractionAgent.",
        (
            "SupplierInteractionAgent is the gateway for all supplier-facing workflows. It routes "
            "communications, normalises supplier responses (free-text → structured), and tracks "
            "interaction history. Sits between NegotiationAgent (initiator) and EmailWatcherAgent "
            "(response detector)."
        ),
    ),
    _ex(
        "Explain EmailDispatchAgent.",
        (
            "EmailDispatchAgent sends approved emails via Amazon SES, records dispatch metadata "
            "(message_id, timestamp, recipient list) in proc.workflow_email_tracking, and tracks "
            "delivery status. Will NOT dispatch without an explicit ApprovalsAgent decision."
        ),
    ),
    _ex(
        "Explain EmailWatcherAgent.",
        (
            "EmailWatcherAgent monitors inbound supplier responses via IMAP (Hostinger mailbox). "
            "Matches reply-to thread headers against active workflows, normalises content, triggers "
            "downstream processing (e.g. quote-attached email → DataExtractionAgent ingestion of "
            "the attachment). Heartbeat every 15 minutes."
        ),
    ),
    _ex(
        "Explain ApprovalsAgent.",
        (
            "ApprovalsAgent decides whether a procurement action proceeds. Reads policy thresholds "
            "(spend limits, delegation of authority) and the action's context (PO total, supplier "
            "risk score). Decisions are 'approved', 'rejected', or 'escalate'. Approvals are logged "
            "to proc.routing for audit. Wrong invoice_amount extraction → wrong threshold check → "
            "policy breach."
        ),
    ),
    _ex(
        "Explain DiscrepancyDetectionAgent.",
        (
            "DiscrepancyDetectionAgent monitors data quality across extraction artifacts. Checks:\n"
            "- Math coherence: invoice_amount + tax_amount ≈ invoice_total_incl_tax (tolerance 0.50).\n"
            "- Sum closure: Σ line.line_amount ≈ invoice_amount.\n"
            "- Cross-reference: invoice.po_id must exist in proc.bp_purchase_order_stg.\n"
            "- Field completeness: required fields populated.\n"
            "- Value plausibility: tax_percent in 0-30%, quantities > 0, dates within sane range.\n\n"
            "Findings land in proc.bp_extraction_discrepancy with severity (critical/warning/info). "
            "Critical issues block promotion; warnings surface in the HITL review queue without blocking."
        ),
    ),
    _ex(
        "Explain RAGAgent.",
        (
            "RAGAgent answers procurement policy and process questions using retrieval-augmented "
            "generation. Backed by a Qdrant vector store over policy documents and SOPs. Used by "
            "operators asking 'what's the approval threshold for IT spend > £50k?' or "
            "'do we need a competitive quote for renewals under £10k?'. NEVER fabricates policy."
        ),
    ),
]


# ---------------------------------------------------------------------------
# 2. Data-model field mapping
# ---------------------------------------------------------------------------
_FIELD_MAPPING = [
    _ex(
        "List the canonical bp_* tables for each document type and their primary keys.",
        {
            "invoice": {
                "header": {"table": "proc.bp_invoice_stg", "pk": "invoice_id"},
                "lines":  {"table": "proc.bp_invoice_line_items_stg", "pk": "invoice_line_id", "fk": "invoice_id"},
                "raw":    {"table": "proc.bp_invoice_raw"},
            },
            "purchase_order": {
                "header": {"table": "proc.bp_purchase_order_stg", "pk": "po_id"},
                "lines":  {"table": "proc.bp_po_line_items_stg", "pk": "po_line_id", "fk": "po_id"},
                "raw":    {"table": "proc.bp_purchase_order_raw"},
            },
            "quote": {
                "header": {"table": "proc.bp_quote_stg", "pk": "quote_id"},
                "lines":  {"table": "proc.bp_quote_line_items_stg", "pk": "quote_line_id", "fk": "quote_id"},
                "raw":    {"table": "proc.bp_quote_raw"},
            },
            "contract": {
                "header": {"table": "proc.bp_contracts", "pk": "contract_id"},
                "raw":    {"table": "proc.bp_contract_raw"},
            },
            "shared": {
                "supplier_master":  "proc.bp_supplier",
                "process_monitor":  "proc.process_monitor",
                "provenance":       "proc.bp_extraction_provenance_v3",
                "discrepancies":    "proc.bp_extraction_discrepancy",
                "patterns":         "proc.bp_extraction_patterns",
            },
        },
    ),
    _ex(
        "What is the difference between line_amount and total_amount on an invoice line item?",
        (
            "line_amount (DB column line_amount on bp_invoice_line_items_stg) is the line "
            "subtotal — quantity × unit_price, BEFORE tax. It MUST be populated on every line.\n\n"
            "total_amount_incl_tax (db column total_amount_incl_tax) is the line total AFTER tax. "
            "It is optional; if the source document only shows a single 'TOTAL' column without "
            "separate tax breakdown for the line, that value goes into line_amount, not into "
            "total_amount_incl_tax. The canonical_labels for line_amount include "
            "['Amount','Line Amount','Line Total','Extended Amount','Subtotal','Sub-Total',"
            "'Sub Total','Net Amount','Net Total','Net','Monthly Cost','Monthly Subtotal'].\n\n"
            "Bare 'Total' as a column header maps to line_amount by substring match on 'Line Total' "
            "since 'total' ⊂ 'line total'."
        ),
    ),
    _ex(
        "What does promote() do and why does it run in a transaction?",
        (
            "promote(raw_id, doc_type) copies one bp_*_raw row's flat-column values into bp_*_stg "
            "and copies the matching bp_*_line_items_raw rows into bp_*_line_items_stg. It:\n"
            "1. Resolves supplier_name → supplier_id via supplier_resolver (exact/fuzzy match against bp_supplier).\n"
            "2. Computes a confidence_score (0-100) based on required vs secondary field completeness.\n"
            "3. Records tax/total reconciliation discrepancies (sum_mismatch, tax_percent_mismatch) as warnings.\n"
            "4. Upserts the header row keyed on the doc_pk (invoice_id/po_id/quote_id).\n"
            "5. DELETE-then-INSERTs line items keyed on the doc_pk (idempotent for re-runs).\n"
            "6. Deletes the bp_*_raw row on success.\n\n"
            "Single transaction so partial promotion is impossible — either both header and lines "
            "land in stg or nothing does."
        ),
    ),
]


# ---------------------------------------------------------------------------
# 3. Extraction discipline rules
# ---------------------------------------------------------------------------
_DISCIPLINE = [
    _ex(
        "An invoice says 'Consulting Services: £5,000' with no quantity or unit price column. "
        "What do you extract for line_items?",
        (
            "One line item:\n"
            '  {"item_description": "Consulting Services", "line_amount": 5000.00, '
            '"quantity": null, "unit_price": null}\n\n'
            "DO NOT fabricate quantity=1 or unit_price=5000. The source does not state them; NULL "
            "preserves the truth. ProcWise enforces no_fabrication_when_absent — only present data "
            "is stored, missing data stays NULL."
        ),
    ),
    _ex(
        "A purchase order line shows 'Acer | TravelMate P2 i5, 8GB RAM | £584.79 | 2 | £1169.58'. "
        "What's the correct extraction?",
        json.dumps({
            "line_items": [{
                "item_description": "Acer | TravelMate P2 i5, 8GB RAM",
                "unit_price": 584.79,
                "quantity": 2,
                "line_total": 1169.58,
            }]
        }),
    ),
    _ex(
        "Why must the literal '|' character be preserved in item_description?",
        (
            "Procurement product descriptions frequently use '|' as a separator (manufacturer | model | "
            "spec). Docling's markdown export HTML-escapes the pipe to &#124; in full_text, but the "
            "structural table cell retains the literal '|'. ProcWise stores the human-readable form: "
            "'|' decoded, not &#124;. The grounding gate compares HTML-unescaped versions so a '|' "
            "candidate matches a '&#124;' in full_text. NEVER strip or replace '|' with another "
            "character — that mutates the product identity."
        ),
    ),
    _ex(
        "An invoice document is provided as image-only (scanned, no extractable text). How is it extracted?",
        (
            "The L0 router detects scanned PDFs via is_scanned_pdf (no text layer) and routes to "
            "PaddleOCR with RapidOCR engine. If parser_confidence < 0.6, falls back to Donut for "
            "a second pass; the higher-confidence result wins. OCR cells flow into ParsedDocument "
            "the same shape as docling output, so the L1/L2/L3 layers are agnostic. Confidence "
            "may be lower; ApprovalsAgent should route low-confidence scans to manual review."
        ),
    ),
    _ex(
        "The PO file is named 'PERRY PO526689 for QUT136586.pdf'. What is po_id?",
        (
            "po_id is '526689' (the PO number 'PO526689' with the canonical 'PO' prefix removed "
            "if the source body uses the digit form). The user-facing convention: ProcWise stores "
            "the bare numeric ID where the body text presents it that way. The quote referenced "
            "in the filename (QUT136586) is NOT this document's primary key — it goes into the "
            "quote_id reference field on the PO header. The output of context_layer.synthesize is "
            "authoritative for the final ID form."
        ),
    ),
    _ex(
        "An invoice text contains 'Total Amount: £1,440.00' AND 'Subtotal: £1,200.00'. Which one is invoice_amount?",
        (
            "invoice_amount = 1200.00 (the subtotal, pre-tax). invoice_total_incl_tax = 1440.00. "
            "invoice_amount is the value that satisfies "
            "invoice_amount + tax_amount = invoice_total_incl_tax. The DiscrepancyDetectionAgent will "
            "log a sum_mismatch warning if these don't reconcile."
        ),
    ),
    _ex(
        "Why is leaving line_items=[] dangerous, and what should the pipeline emit?",
        (
            "Empty line_items on a doc that obviously has line items is silent data loss — "
            "downstream agents compute totals as 0 and rank suppliers incorrectly. The dispatch "
            "pipeline now logs a 'missing_line_items' warning discrepancy (severity=warning, "
            "blocks_promotion=False) so the row promotes with header data, but the issue surfaces "
            "in the HITL review queue (proc.bp_extraction_discrepancy + proc.extraction_review_queue). "
            "Operators can fix and re-promote."
        ),
    ),
]


# ---------------------------------------------------------------------------
# 4. Layout patterns AgentNick must handle
# ---------------------------------------------------------------------------
_LAYOUTS = [
    _ex(
        "Extract line items from this paragraph-layout invoice (single service, no structural table).",
        json.dumps({"line_items": [{
            "item_description": "Monthly Design & Marketing Package April Payment Total",
            "line_amount": 1200.00,
        }]}),
        (
            "## Item Description\n\n"
            "Monthly Design & Marketing Package April Payment Total\n\n"
            "£1,200\n\n"
            "Total Amount:\n£1,440.00\n\n"
            "VAT (20%):\n£240.00\n"
        ),
    ),
    _ex(
        "Extract line items from this markdown pipe-table invoice.",
        json.dumps({"line_items": [{
            "item_description": "Social Media Management Instagram Facebook LinkedIn",
            "line_amount": 2000.00,
        }]}),
        (
            "| SERVICES                                             | TOTAL   |\n"
            "|------------------------------------------------------|---------|\n"
            "| Social Media Management Instagram Facebook LinkedIn  | £2,000  |\n"
            "| SUBTOTAL                                             | £2,000  |\n"
            "| VAT 20%                                              | £400    |\n"
            "| TOTAL                                                | £2,400  |\n"
        ),
    ),
    _ex(
        "A PO has stacked column headings '## DESCRIPTION' then '## SUBTOTAL' "
        "with content broken across multiple lines. Extract the line item.",
        json.dumps({"line_items": [{
            "item_description": "Consultancy: Digital Strategy, Implementation & Training",
            "line_amount": 50000.00,
        }]}),
        (
            "## DESCRIPTION\n\n## SUBTOTAL\n\n"
            "Consultancy:\n\nDigital Strategy,\n\nImplementation & Training\n\n£50,000\n\n"
            "Sub-total:\n\n£50,000\n\nTax (20%):\n\n£10,000\n\nTotal:\n\n£60,000\n"
        ),
    ),
    _ex(
        "Extract line items from a multi-line invoice with separate Qty, Cost, Subtotal columns.",
        json.dumps({"line_items": [
            {"item_description": "Staedtler Ballpoint Pen Black Ink", "quantity": 100, "unit_price": 13.85, "line_amount": 1385.00},
            {"item_description": "3M Adhesive Tape 15 cmx 19 mm",     "quantity": 15,  "unit_price": 15.29, "line_amount": 229.35},
        ]}),
        (
            "| Description                       | Qty | Cost   | Subtotal  |\n"
            "|-----------------------------------|-----|--------|-----------|\n"
            "| Staedtler Ballpoint Pen Black Ink | 100 | £13.85 | £1,385.00 |\n"
            "| 3M Adhesive Tape 15 cmx 19 mm     | 15  | £15.29 | £229.35   |\n"
        ),
    ),
]


# ---------------------------------------------------------------------------
# 5. Cross-agent workflow context
# ---------------------------------------------------------------------------
_WORKFLOW = [
    _ex(
        "Trace the complete data flow from a supplier emailing a quote PDF to a ProcWise approval decision.",
        (
            "1. SUPPLIER → INBOX: Supplier emails quote PDF to supplierconnect@procwise.co.uk.\n"
            "2. EmailWatcherAgent (15-min IMAP poll) detects the email, extracts attachments, uploads "
            "to s3://procwisemvp/documents/quote/, inserts a row into proc.process_monitor "
            "(category='quote', status='Uploaded').\n"
            "3. process_monitor_watcher claims the record (FOR UPDATE SKIP LOCKED), dispatches to "
            "src.services.extraction.dispatch.dispatch_document.\n"
            "4. EXTRACTION: L0 parse → L1 regex → L2 table_extractor → L3 grounded judge → "
            "context_layer synthesise → write bp_quote_raw → promote() → bp_quote_stg + "
            "bp_quote_line_items_stg + bp_extraction_provenance_v3.\n"
            "5. process_monitor.status = 'Extracted', confidence_score persisted.\n"
            "6. AgentNick.kg_sync pushes the row + lines into Neo4j as Quote, QuoteLine, FROM_SUPPLIER, "
            "FOR_PO relationships.\n"
            "7. SupplierRankingAgent recomputes ranking using the new quote.\n"
            "8. QuoteEvaluationAgent + QuoteComparisonAgent score the quote vs alternatives.\n"
            "9. If acceptable: ApprovalsAgent applies policy thresholds, may auto-approve or escalate.\n"
            "10. Negotiation may trigger via NegotiationAgent → EmailDraftingAgent → ApprovalsAgent → "
            "EmailDispatchAgent (SES) → back to supplier."
        ),
    ),
    _ex(
        "What happens if extraction confidence is below 0.90?",
        (
            "Two things:\n"
            "1. Promotion still completes — the data is durable in bp_<doc_type>_stg with the "
            "confidence_score recorded. Downstream agents can read the score and weight accordingly.\n"
            "2. The (doc_type, doc_pk) is NOT added to the auto_collected fine-tune dataset. Only "
            "extractions with confidence ≥ 0.90 AND zero errors become training examples for the "
            "next day's AgentNick fine-tune. This keeps the training corpus self-reinforcing on "
            "high-quality outputs — never on guesses."
        ),
    ),
    _ex(
        "What does the daily AgentNick fine-tune do and when does it run?",
        (
            "Daily QLoRA fine-tune fires via cron at 18:30 UTC (00:00 IST midnight) with an 8h30m "
            "hard timeout ending 03:00 UTC (08:30 IST). Steps:\n"
            "1. build_finetune_dataset.py: assembles training corpus from\n"
            "   - src/data/training/auto_collected_examples.jsonl (the day's verified extractions),\n"
            "   - existing QLoRA examples,\n"
            "   - procwise_knowledge_examples (this file: agent capabilities, schema, layouts).\n"
            "2. QLoRA fine-tune Qwen2.5-7B-Instruct with the dataset, lora_r=16, alpha=32, 3 epochs.\n"
            "3. Merge adapters into base model.\n"
            "4. Quantize to Q4_K_M GGUF.\n"
            "5. Re-register as BeyondProcwise/AgentNick:latest in Ollama.\n\n"
            "The procwise service must be stopped during the run — full-precision fine-tune needs "
            "~14 GiB GPU which would OOM against the live extractor's 9.5 GiB footprint."
        ),
    ),
]


# ---------------------------------------------------------------------------
# 6. Negative examples — what NOT to do
# ---------------------------------------------------------------------------
_NEGATIVES = [
    _ex(
        "An invoice has no clear date field — what should invoice_date be?",
        "invoice_date should be NULL. DO NOT fabricate a date from today, the upload date, or a guess. "
        "DiscrepancyDetectionAgent will flag missing_required for a NULL invoice_date so an operator "
        "can intervene. A fabricated date corrupts payment terms, ageing reports, and approval timing.",
    ),
    _ex(
        "An invoice mentions 'invoice_no: 2025-290' in the body and the filename is 'INV2025-290.pdf'. "
        "What is invoice_id — '2025-290', 'INV2025-290', or 'INV-2025-290'?",
        "invoice_id = '2025-290'. Output the raw token as the body presents it. DO NOT add or remove "
        "a prefix to canonicalise. The filename is a hint, not the source of truth. context_layer's "
        "prompt explicitly forbids prefix mutation.",
    ),
    _ex(
        "A line description appears as 'Acer &#124; TravelMate' in full_text. What is the stored item_description?",
        "'Acer | TravelMate'. HTML entities are decoded before persistence. The DB stores the "
        "human-readable form, not the markdown-escaped form. Decoding happens in the L2 extractor's "
        "_clean_text_value helper and the L3 grounding gate uses html.unescape on both sides before "
        "the substring check.",
    ),
    _ex(
        "When can supplier_id be inferred from filename?",
        "Only when the document body's supplier name is unparseable AND the filename clearly names "
        "the supplier AND that supplier name appears as a stem inside full_text. This is the "
        "filename-hint fallback in promote(). For all other cases, supplier_id comes from the body "
        "via supplier_resolver. Never invent a SUP-* identifier; always resolve to an existing entry "
        "in proc.bp_supplier or call resolve_or_create_supplier.",
    ),
]


def generate_procwise_knowledge_examples() -> list[dict]:
    """Compose all ProcWise-knowledge examples for the fine-tune dataset."""
    out: list[dict] = []
    out.extend(_AGENT_DEEP_DIVES)
    out.extend(_FIELD_MAPPING)
    out.extend(_DISCIPLINE)
    out.extend(_LAYOUTS)
    out.extend(_WORKFLOW)
    out.extend(_NEGATIVES)
    return out


if __name__ == "__main__":
    # Self-test: print counts so we can eyeball balance.
    ex = generate_procwise_knowledge_examples()
    print(f"agent_deep_dives  : {len(_AGENT_DEEP_DIVES)}")
    print(f"field_mapping     : {len(_FIELD_MAPPING)}")
    print(f"discipline        : {len(_DISCIPLINE)}")
    print(f"layouts           : {len(_LAYOUTS)}")
    print(f"workflow          : {len(_WORKFLOW)}")
    print(f"negatives         : {len(_NEGATIVES)}")
    print(f"---")
    print(f"TOTAL ProcWise-knowledge: {len(ex)} examples")
