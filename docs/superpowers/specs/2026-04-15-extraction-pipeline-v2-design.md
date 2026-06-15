# ProcWise Extraction Pipeline v2 — Design Spec

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this spec task-by-task.

**Goal:** Redesign the extraction pipeline for 100% accuracy with speed, context-awareness, and self-improving vendor intelligence.

**Architecture:** Single LLM call per document with intelligent pre-processing (section detection, smart text windowing, vendor profile injection) and post-verification (source text anchoring, line item completeness, supplier resolution, smart computation). No fixed regex patterns. Source data never modified.

**Tech Stack:** Python, Ollama (qwen3:30b MoE on GPU), PostgreSQL, Qdrant, FastAPI

---

## 1. Schema Fixes

### 1.1 PO Header — Add Missing Financial Columns

```sql
ALTER TABLE proc.bp_purchase_order ADD COLUMN IF NOT EXISTS tax_percent numeric;
ALTER TABLE proc.bp_purchase_order ADD COLUMN IF NOT EXISTS tax_amount numeric;
ALTER TABLE proc.bp_purchase_order ADD COLUMN IF NOT EXISTS total_amount_incl_tax numeric;
ALTER TABLE proc.bp_purchase_order ADD COLUMN IF NOT EXISTS supplier_id text;
```

**Why:** PO currently has only `total_amount` which is ambiguous (subtotal vs grand total). Adding `tax_amount` and `total_amount_incl_tax` matches Invoice/Quote pattern and eliminates the LINE_SUM_MISMATCH issue on 100% of POs.

### 1.2 Fix Column Typo

```sql
ALTER TABLE proc.bp_po_line_items RENAME COLUMN unit_of_measue TO unit_of_measure;
```

### 1.3 Update TABLE_SCHEMAS

Update `direct_extraction_service.py` TABLE_SCHEMAS for Purchase_Order to include new columns and fix the typo. Update the extraction prompt's PO context accordingly.

---

## 2. Three-Phase Extraction Pipeline

### Phase 1: Document Intelligence (Pre-LLM)

**2.1 Section Detector**

New module: `src/services/document_intelligence.py`

Scans raw text and identifies structural sections:
- **Header area**: document ID, dates, supplier/buyer info (top of document)
- **Line items table**: detected by table header keywords (Description, Qty, Price, Amount) and tabular row patterns
- **Summary section**: Subtotal, Tax, Total rows
- **Payment/boilerplate**: bank details, T&Cs, signatures

Returns a `DocumentStructure` dataclass with section boundaries (line ranges).

**2.2 Smart Text Builder**

Instead of `text[:6000]`, builds the LLM prompt text with priority ordering:
1. ALL line item rows (never truncated — this is non-negotiable)
2. Full header section
3. Summary/totals section
4. Remaining content (addresses, notes) if space permits
5. Boilerplate dropped last

Maximum total: 6000 chars. If line items alone exceed 6000 chars (very long documents), split into header+summary call and a separate line-items-only call.

**2.3 Vendor Profile Lookup**

Before the LLM call, search `proc.bp_supplier` for the supplier:
- Extract supplier hint from filename (e.g., "TECHWORLD" from "TECHWORLD INV-005-32")
- Search `bp_supplier.supplier_name` and `bp_supplier.trading_name`
- If found: inject vendor-specific hints into the prompt (currency, date format patterns, typical tax rate)
- Also check `proc.vendor_profile` table (created by `_learn_vendor_profile`) for extraction history patterns

---

### Phase 2: Context-Aware LLM Extraction

**2.4 Layered Prompt Construction**

Single LLM call with four context layers:

```
Layer 1: Procurement domain context (per doc type)
  - Already built: Invoice, PO, Quote, Contract contexts
  - Enhanced PO context: explicit subtotal vs total-incl-tax distinction

Layer 2: Exact DB column names and types
  - From TABLE_SCHEMAS — source of truth
  - Skip audit columns (created_date, etc.)

Layer 3: Vendor profile hints (if available)
  - "This vendor: GBP currency, DD/MM/YYYY dates, 20% VAT typical"
  - "Previous extractions: invoice_id format INV-XXX-XX"

Layer 4: Section-tagged document text
  - [HEADER] ... [LINE ITEMS] ... [SUMMARY] ... [PAYMENT] ...
  - Smart text builder output — line items never truncated
```

**2.5 Extraction Model**

- Model: `BeyondProcwise/AgentNick:extract` (lean system prompt, num_ctx=8192)
- GPU: 25/49 layers on NVIDIA A10G (12.3GB VRAM)
- Timeout: 300s default, retry: 2 attempts with 5s backoff
- Concurrent: 3 max (semaphore)

---

### Phase 3: Source Verification + Enrichment

**2.6 Value Anchoring**

For each extracted field value, check if it literally appears in the source text:
- Direct match: confidence = 0.95+
- Normalized match (whitespace/hyphen stripped): confidence = 0.90+
- Not found in source: flag as "unanchored" in discrepancy log, keep value

**2.7 Line Item Completeness Check**

Count table rows in source text (between table header and summary markers) and compare to extracted line_items count:
- Match: proceed
- Mismatch: LOG the gap. If extracted < source rows AND document was truncated, make a focused second LLM call for line items only using the full line items section text

**2.8 Supplier Resolution + Auto-Creation**

Search `proc.bp_supplier`:
1. Exact match on `supplier_name` or `trading_name` (case-insensitive)
2. Fuzzy match (ILIKE with wildcards)
3. If found: set `supplier_id` from the matched record

If NOT found (all searches miss):
- Auto-create new supplier in `proc.bp_supplier`:
  - `supplier_id`: auto-generated from name (e.g., "SUP-SupplyX1Ltd")
  - `supplier_name`: verbatim from document
  - `trading_name`: same as supplier_name
  - `default_currency`: from document's currency field
  - `country`: from document if available
  - `address_line1/city/postal_code`: from document if available
  - `contact_email_1/contact_phone_1`: from document if available
  - `created_by`: "AgentNick-AutoDiscovery"
- Log auto-creation to `bp_discrepancy_data` with `severity=info, rule_name=new_supplier_discovered`
- Also create in `proc.supplier` table (keeps both tables in sync)

**2.9 Smart Computation (Absent Fields Only)**

Derive values ONLY when the field is completely absent from the document. Source values are NEVER modified.

**Header computations:**
- `total_incl_tax` absent + `subtotal` present + `tax_amount` present → derive total = subtotal + tax
- `subtotal` absent + `total_incl_tax` present + `tax_amount` present → derive subtotal = total - tax
- `tax_amount` absent + `total_incl_tax` present + `subtotal` present → derive tax = total - subtotal
- `tax_percent` absent + `tax_amount` present + `subtotal` present (>0) → derive pct = (tax/subtotal) × 100
- `subtotal` absent + line items present → derive subtotal = SUM(line_amounts)

**Line item computations:**
- `line_amount` absent + `quantity` present + `unit_price` present → derive amount = qty × price
- Never guess quantity or unit_price from line_amount

**Currency conversion (always computed):**
- `exchange_rate_to_usd`: live API rate, fallback static table
- `converted_amount_usd`: total_incl_tax × exchange_rate

**Never computed:**
- Dates, payment_terms, supplier names, descriptions
- Never override any value present in the document
- Never "fix" math that looks wrong

**2.10 Anomaly Logging**

Log to `proc.bp_discrepancy_data` (severity levels: error, warning, info):
- AMOUNT_MISMATCH: subtotal + tax != total
- LINE_SUM_MISMATCH: line items sum != header subtotal
- LINE_MATH: qty × price != line_amount
- DATE_LOGIC: due_date before invoice_date
- TAX_PERCENT_HIGH: tax_percent > 30%
- QUANTITY_HIGH: quantity > 10000
- SUPPLIER_SHORT: supplier name < 3 chars
- NEW_SUPPLIER: auto-created supplier
- UNANCHORED_VALUE: extracted value not found in source text

All anomalies are logged but NEVER trigger data modification.

---

## 3. Vendor Profile Learning

### 3.1 Profile Structure

After each successful extraction, update the vendor profile:

```json
{
  "supplier_name": "TechWorld",
  "date_format_hint": "DD MMM YYYY",
  "id_pattern": "INV-XXX-XX",
  "typical_tax_rate": 20.0,
  "default_currency": "GBP",
  "line_item_layout": "description | qty | unit_price | amount",
  "total_extractions": 13,
  "avg_confidence": 0.94,
  "common_issues": ["StructTree fallback needed", "corrupted PDF"],
  "last_extraction": "2026-04-15T08:55:00Z"
}
```

### 3.2 Profile Storage

Use the existing `vendor_profile` service. If no DB table exists, create:

```sql
CREATE TABLE IF NOT EXISTS proc.vendor_profile (
  supplier_name text PRIMARY KEY,
  profile_data jsonb NOT NULL DEFAULT '{}',
  extraction_count integer DEFAULT 0,
  avg_confidence numeric DEFAULT 0,
  last_extraction timestamp,
  created_date timestamp DEFAULT NOW(),
  last_modified_date timestamp DEFAULT NOW()
);
```

### 3.3 Profile Injection

When extracting a document from a known vendor:
- Fetch profile from `proc.vendor_profile`
- Inject as Layer 3 in the LLM prompt:
  ```
  VENDOR CONTEXT (from previous extractions):
  - This vendor typically uses GBP currency
  - Date format: DD MMM YYYY
  - Tax rate: usually 20%
  - Document ID format: INV-XXX-XX
  ```

---

## 4. Pre-KG Validation Gate

Before syncing to Knowledge Graph:
- PK must be present
- Confidence must be >= 0.70
- Error-severity discrepancies must be <= 2
- If gate fails: data persists in bp_ tables but does NOT sync to KG

---

## 5. Training Data Collection

After each successful extraction:
- If confidence >= 0.90 AND error_count == 0 AND PK present
- Deduplicate by doc_type + PK (don't collect same document twice)
- Append to `data/training/auto_collected_examples.jsonl`
- Format: instruction (prompt) + input (document text) + output (extracted JSON)
- Use `scripts/prepare_finetune_data.py` to convert for QLoRA training

---

## 6. Files to Create/Modify

### New files:
- `src/services/document_intelligence.py` — Section detector + smart text builder
- `scripts/schema_migration_v2.sql` — Schema fix SQL

### Modified files:
- `src/services/direct_extraction_service.py` — Smart text builder integration, updated TABLE_SCHEMAS for PO
- `src/services/agent_nick_orchestrator.py` — Phase 1/3 integration, supplier auto-creation, vendor profile injection
- `src/services/extraction_validator.py` — Value anchoring, line item completeness check
- `src/services/process_monitor_watcher.py` — Already has pre-KG gate and training collection

### Existing files (no changes needed):
- `src/services/ollama_client.py` — Already configured for GPU
- `src/services/validation_gate.py` — Already has confidence scoring
- `utils/gpu.py` — Already fixed for CPU default device

---

## 7. Success Criteria

- Every extracted field value matches the source document verbatim
- Zero line items missed (completeness check catches truncation)
- PO financial fields correctly mapped (subtotal vs total-incl-tax)
- New suppliers auto-created in bp_supplier
- Vendor profiles improve accuracy over time
- Confidence consistently >= 0.92
- All anomalies logged, source data never modified
- GPU inference: < 60s per document
