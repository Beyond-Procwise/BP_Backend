# Extraction Accuracy & Product Catalog Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Achieve 100% accurate extraction from all document types (PDF, DOCX, Excel, JPEG), create a `bp_products` master catalog with auto-generated item_ids, and ensure line items are accurately captured with correct buyer/supplier identification.

**Architecture:** Three layers — (1) Improved `_tabular_to_text` for smart Excel structure detection, (2) Enhanced LLM extraction prompts with stronger procurement context, item_id extraction rules, and Excel-aware guidance, (3) New `ProductCatalogService` that fuzzy-matches products and maintains a `bp_products` table with current unit prices. Discrepancies are logged but source values are never auto-corrected.

**Tech Stack:** Python, openpyxl, difflib.SequenceMatcher, psycopg2, Ollama LLM

---

### Task 1: Create `bp_products` Table

**Files:**
- Create: `src/services/product_catalog_service.py`

- [ ] **Step 1: Create the bp_products table via SQL**

Connect to the database and run:

```sql
CREATE TABLE IF NOT EXISTS proc.bp_products (
    product_id          TEXT PRIMARY KEY,
    item_description    TEXT NOT NULL,
    current_unit_price  NUMERIC,
    currency            TEXT,
    unit_of_measure     TEXT,
    first_seen_date     TIMESTAMP DEFAULT NOW(),
    last_seen_date      TIMESTAMP DEFAULT NOW(),
    source_doc_type     TEXT,
    source_doc_id       TEXT,
    occurrence_count    INTEGER DEFAULT 1,
    created_date        TIMESTAMP DEFAULT NOW(),
    last_modified_date  TIMESTAMP DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_bp_products_description
    ON proc.bp_products USING btree (lower(item_description));
```

Run via `psycopg2` in a one-off script or inline in the service init.

- [ ] **Step 2: Create `src/services/product_catalog_service.py`**

```python
"""Product Catalog Service — master product registry with fuzzy matching.

Maintains proc.bp_products as a canonical product catalog.
Each line item extracted from any document gets matched to an existing
product or creates a new one. product_id is used as item_id in all
line item tables (bp_po_line_items, bp_quote_line_items, bp_invoice_line_items).
"""

import logging
import re
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

MATCH_THRESHOLD = 0.85  # fuzzy match ratio


class ProductCatalogService:
    def __init__(self, get_db_connection):
        self._get_conn = get_db_connection
        self._cache: Dict[str, Dict[str, Any]] = {}  # product_id -> row
        self._descriptions: List[Tuple[str, str]] = []  # (normalized, product_id)
        self._next_seq: Optional[int] = None
        self._load_cache()

    def _load_cache(self):
        """Load all products into memory for fast matching."""
        try:
            conn = self._get_conn()
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT product_id, item_description, current_unit_price, "
                    "currency, unit_of_measure, last_seen_date "
                    "FROM proc.bp_products"
                )
                for row in cur.fetchall():
                    pid, desc = row[0], row[1]
                    self._cache[pid] = {
                        "product_id": pid,
                        "item_description": desc,
                        "current_unit_price": row[2],
                        "currency": row[3],
                        "unit_of_measure": row[4],
                        "last_seen_date": row[5],
                    }
                    self._descriptions.append((self._normalize(desc), pid))
                # Determine next sequence number
                cur.execute(
                    "SELECT product_id FROM proc.bp_products "
                    "WHERE product_id LIKE 'PROD-%%' "
                    "ORDER BY product_id DESC LIMIT 1"
                )
                last = cur.fetchone()
                if last:
                    num = re.search(r"PROD-(\d+)", last[0])
                    self._next_seq = int(num.group(1)) + 1 if num else 1
                else:
                    self._next_seq = 1
            conn.close()
        except Exception:
            logger.warning("Could not load product cache", exc_info=True)
            self._next_seq = 1

    @staticmethod
    def _normalize(text: str) -> str:
        """Normalize description for comparison."""
        text = text.lower().strip()
        text = re.sub(r"[^a-z0-9\s]", "", text)
        text = re.sub(r"\s+", " ", text)
        return text

    def match_or_create(
        self,
        item_description: str,
        item_id_from_doc: Optional[str],
        unit_price: Optional[float],
        currency: Optional[str],
        unit_of_measure: Optional[str],
        doc_type: str,
        doc_id: str,
    ) -> str:
        """Match item to existing product or create new.

        Returns product_id to use as item_id in line item tables.
        """
        if not item_description or not item_description.strip():
            return self._generate_id()

        # 1. If document has an explicit item/SKU/part code, use it directly
        if item_id_from_doc and item_id_from_doc.strip():
            pid = item_id_from_doc.strip()
            if pid in self._cache:
                self._update_product(pid, unit_price, currency, unit_of_measure,
                                     doc_type, doc_id, item_description)
            else:
                self._insert_product(pid, item_description, unit_price, currency,
                                     unit_of_measure, doc_type, doc_id)
            return pid

        # 2. Fuzzy match against known products
        norm = self._normalize(item_description)
        best_ratio = 0.0
        best_pid = None
        for cached_norm, cached_pid in self._descriptions:
            ratio = SequenceMatcher(None, norm, cached_norm).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_pid = cached_pid

        if best_ratio >= MATCH_THRESHOLD and best_pid:
            self._update_product(best_pid, unit_price, currency, unit_of_measure,
                                 doc_type, doc_id, item_description)
            return best_pid

        # 3. No match — create new product with generated ID
        pid = self._generate_id()
        self._insert_product(pid, item_description, unit_price, currency,
                             unit_of_measure, doc_type, doc_id)
        return pid

    def _generate_id(self) -> str:
        pid = f"PROD-{self._next_seq:05d}"
        self._next_seq += 1
        return pid

    def _insert_product(self, product_id, description, price, currency,
                        uom, doc_type, doc_id):
        now = datetime.now(timezone.utc)
        try:
            conn = self._get_conn()
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO proc.bp_products
                    (product_id, item_description, current_unit_price, currency,
                     unit_of_measure, first_seen_date, last_seen_date,
                     source_doc_type, source_doc_id, occurrence_count,
                     created_date, last_modified_date)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,1,%s,%s)
                    ON CONFLICT (product_id) DO UPDATE SET
                        current_unit_price = EXCLUDED.current_unit_price,
                        currency = EXCLUDED.currency,
                        unit_of_measure = COALESCE(EXCLUDED.unit_of_measure, proc.bp_products.unit_of_measure),
                        last_seen_date = EXCLUDED.last_seen_date,
                        source_doc_type = EXCLUDED.source_doc_type,
                        source_doc_id = EXCLUDED.source_doc_id,
                        occurrence_count = proc.bp_products.occurrence_count + 1,
                        last_modified_date = EXCLUDED.last_modified_date,
                        item_description = CASE
                            WHEN length(EXCLUDED.item_description) > length(proc.bp_products.item_description)
                            THEN EXCLUDED.item_description
                            ELSE proc.bp_products.item_description END
                    """,
                    (product_id, description, price, currency, uom,
                     now, now, doc_type, doc_id, now, now),
                )
            conn.close()
            # Update local cache
            self._cache[product_id] = {
                "product_id": product_id,
                "item_description": description,
                "current_unit_price": price,
                "currency": currency,
                "unit_of_measure": uom,
                "last_seen_date": now,
            }
            self._descriptions.append((self._normalize(description), product_id))
            logger.info("Product catalog: created %s — %s", product_id, description[:60])
        except Exception:
            logger.exception("Failed to insert product %s", product_id)

    def _update_product(self, product_id, price, currency, uom,
                        doc_type, doc_id, description=None):
        now = datetime.now(timezone.utc)
        cached = self._cache.get(product_id, {})
        # Only update price if this is newer data
        update_price = price is not None
        try:
            conn = self._get_conn()
            conn.autocommit = True
            with conn.cursor() as cur:
                if update_price:
                    cur.execute(
                        """UPDATE proc.bp_products SET
                            current_unit_price = %s,
                            currency = COALESCE(%s, currency),
                            unit_of_measure = COALESCE(%s, unit_of_measure),
                            last_seen_date = %s,
                            source_doc_type = %s,
                            source_doc_id = %s,
                            occurrence_count = occurrence_count + 1,
                            last_modified_date = %s,
                            item_description = CASE
                                WHEN %s IS NOT NULL AND length(%s) > length(item_description)
                                THEN %s ELSE item_description END
                        WHERE product_id = %s""",
                        (price, currency, uom, now, doc_type, doc_id, now,
                         description, description, description, product_id),
                    )
                else:
                    cur.execute(
                        """UPDATE proc.bp_products SET
                            occurrence_count = occurrence_count + 1,
                            last_seen_date = %s,
                            last_modified_date = %s
                        WHERE product_id = %s""",
                        (now, now, product_id),
                    )
            conn.close()
            if update_price:
                cached["current_unit_price"] = price
            if currency:
                cached["currency"] = currency
            if uom:
                cached["unit_of_measure"] = uom
            cached["last_seen_date"] = now
        except Exception:
            logger.exception("Failed to update product %s", product_id)
```

- [ ] **Step 3: Commit**

```bash
git add src/services/product_catalog_service.py
git commit -m "feat: add bp_products table and ProductCatalogService with fuzzy matching"
```

---

### Task 2: Improve Excel `_tabular_to_text` for Structure-Aware Extraction

**Files:**
- Modify: `src/services/direct_extraction_service.py` — `_tabular_to_text` method (lines 906-980)

The current method outputs raw pipe-delimited cells. The improvement detects the header row and labels each data cell with its column header, separating metadata from the line items table.

- [ ] **Step 1: Replace `_tabular_to_text` Excel handling**

In `src/services/direct_extraction_service.py`, replace the Excel section of `_tabular_to_text` (from `# Excel: read raw cells to capture metadata above the table` to the end of the method).

The new logic:
1. Read all cells from each sheet
2. Detect the header row: first row where 3+ cells look like column names (contain words like "quantity", "description", "price", "total", "amount", "item", "unit", "product")
3. Split output into `=== DOCUMENT METADATA ===` (rows above header) and `=== LINE ITEMS TABLE ===` (header row onward)
4. In the line items section, label each cell with its column header: `Quantity: 20 | Description: Herman Miller Aeron Chair | Unit Price: 1050 | Total Price: 21000`
5. Mark subtotal/total/tax rows clearly: `--- TOTALS ---`

Key detection patterns:
- Header row keywords: `quantity, qty, description, item, product, unit price, total, amount, price, uom, unit of measure`
- Total row keywords: `subtotal, sub-total, total, vat, tax, delivery, grand total`
- Stop extracting line items when a total row is reached

```python
# Detect header row
header_keywords = {"quantity", "qty", "description", "item", "product",
                   "unit price", "total", "amount", "price", "uom",
                   "unit of measure", "part", "sku", "code", "no"}
total_keywords = {"subtotal", "sub-total", "total", "vat", "tax",
                  "delivery", "grand total", "net", "gross"}

def _is_header_row(cells_text):
    """Check if 3+ cells match header keywords."""
    matches = 0
    for ct in cells_text:
        norm = ct.lower().strip()
        if any(kw in norm for kw in header_keywords):
            matches += 1
    return matches >= 3

def _is_total_row(cells_text):
    """Check if a row is a subtotal/total/tax summary row."""
    for ct in cells_text:
        norm = ct.lower().strip()
        if any(kw in norm for kw in total_keywords):
            return True
    return False
```

The output for the HR Quote v2.xlsx example should look like:
```
=== DOCUMENT METADATA ===
Quotation
PeopleFirst HR Solutions Ltd
3rd Floor, Regent House, Birmingham, B2 5QP
Telephone: +44 (0)121 445 7821
Delivery Address | Invoice Address | Quote Information
Laura Stevens | Horizon Retail Group Ltd | Prepared For: Laura Stevens
Horizon HQ | Finance Department | Quote Number: QTE-2026-01521
45 Market Street | 45 Market Street | Quote Date: 02/10/2025
Birmingham | Birmingham | Account Number: HRG221
B3 1AA | B3 1AA | Account Manager: Sophie Turner
Email: sophie.turner@peoplefirsthr.co.uk
Company Registration: 40219876
VAT Registration: 665342198

=== LINE ITEMS TABLE ===
Quantity: 1 | Description: HR Transformation Programme (Phase 1 – Assessment & Design) | Unit Price (£): 16500 | Total Price (£): 16500
Quantity: 1 | Description: Workforce Skills Gap Analysis & Capability Mapping | Unit Price (£): 11200 | Total Price (£): 11200
...

--- TOTALS ---
Subtotal: 100000
VAT (20%): 20000
Total (GBP): 120000
```

- [ ] **Step 2: Test with sample files locally**

```bash
.venv/bin/python -c "
from services.direct_extraction_service import DirectExtractionService
import openpyxl
from io import BytesIO

with open('/home/muthu/Downloads/new/HR Quote v2.xlsx', 'rb') as f:
    data = f.read()

svc = DirectExtractionService.__new__(DirectExtractionService)
text = svc._tabular_to_text(data, '.xlsx')
print(text)
"
```

Verify the output has clear METADATA, LINE ITEMS, and TOTALS sections.

- [ ] **Step 3: Commit**

```bash
git add src/services/direct_extraction_service.py
git commit -m "feat: structure-aware Excel-to-text conversion with header detection"
```

---

### Task 3: Enhance LLM Extraction Prompts for Accuracy

**Files:**
- Modify: `src/services/direct_extraction_service.py` — `_PROCUREMENT_CONTEXT` (lines 1012-1054) and `_build_extraction_prompt` (lines 1056-1123)

- [ ] **Step 1: Expand `_PROCUREMENT_CONTEXT` with stronger buyer/supplier rules**

Add to each doc type context:
- For **Quote**: The supplier is the company whose letterhead/logo/address appears at the TOP of the document. The buyer is the company in the "Prepared For", "Customer", "Bill To" section. For Excel quotes, the supplier company name is typically in the first few rows above the table.
- For **Purchase_Order**: The buyer is the company that ISSUED/CREATED the PO (look for "From", "Issued By", letterhead). The supplier is the company the PO is addressed TO (look for "To", "Vendor", "Supplier"). In the context of this PO, the buyer is PLACING the order and the supplier is FULFILLING it.
- For **Invoice**: The supplier is the company whose letterhead/name is at the TOP (they are billing). The buyer is in "Bill To" / "Invoice To". supplier_name should be the company name from the letterhead, NOT the "Bill To" company.

- [ ] **Step 2: Add item_id and unit_of_measure extraction rules to the prompt**

Add to CRITICAL RULES section of `_build_extraction_prompt`:
```
14. item_id: If the document shows a product code, SKU, part number, catalog number, or item reference for a line item, extract it as item_id. Look for columns like "Item Code", "SKU", "Part No", "Product Code", "Ref", "Item #". If no product code is in the document, OMIT item_id.
15. unit_of_measure: Extract the unit if present (e.g., "each", "box", "kg", "hours", "months", "days", "per annum", "set"). Look for columns like "UOM", "Unit", "Measure". If not explicitly stated, OMIT — do not guess.
16. For EXCEL documents: The metadata section above the table contains header information (supplier, buyer, dates, quote/PO number). The LINE ITEMS TABLE section contains the actual products/services. Extract header fields from metadata and line items from the table.
```

- [ ] **Step 3: Commit**

```bash
git add src/services/direct_extraction_service.py
git commit -m "feat: strengthen extraction prompts for buyer/supplier accuracy and item_id"
```

---

### Task 4: Wire Product Catalog into Line Item Persistence

**Files:**
- Modify: `src/services/direct_extraction_service.py` — `_persist_line_items` method
- Modify: `src/services/agent_nick_orchestrator.py` — initialize ProductCatalogService

- [ ] **Step 1: Initialize ProductCatalogService in orchestrator**

In `src/services/agent_nick_orchestrator.py`, import and initialize the service during `__init__`:

```python
from services.product_catalog_service import ProductCatalogService
# In __init__:
self._product_catalog = ProductCatalogService(self._agent_nick.get_db_connection)
```

- [ ] **Step 2: Call product catalog before persistence**

In the extraction flow (within `_dispatch_extraction` or `process_document`), after line items are extracted and validated, before calling `_persist_line_items`:

```python
# Populate item_id via product catalog
for item in line_items:
    desc = item.get("item_description", "")
    doc_item_id = item.get("item_id")  # from document if extracted
    price = None
    try:
        price = float(item.get("unit_price", 0) or 0) or None
    except (ValueError, TypeError):
        pass
    currency = header.get("currency")
    uom = item.get("unit_of_measure")

    product_id = self._product_catalog.match_or_create(
        item_description=desc,
        item_id_from_doc=doc_item_id,
        unit_price=price,
        currency=currency,
        unit_of_measure=uom,
        doc_type=doc_type,
        doc_id=pk_value,
    )
    item["item_id"] = product_id
```

- [ ] **Step 3: Ensure bp_products table is created on startup**

In the service init or startup sequence, run the CREATE TABLE IF NOT EXISTS SQL from Task 1.

- [ ] **Step 4: Commit**

```bash
git add src/services/direct_extraction_service.py src/services/agent_nick_orchestrator.py src/services/product_catalog_service.py
git commit -m "feat: wire product catalog into extraction pipeline for item_id population"
```

---

### Task 5: End-to-End Testing with Sample Files

**Files:**
- Test manually against sample documents in `/home/muthu/Downloads/quotes/` and `/home/muthu/Downloads/new/`

- [ ] **Step 1: Restart the service**

```bash
kill -TERM $(pgrep -f "uvicorn api.main") 2>/dev/null
# Wait for systemd auto-restart
until journalctl -u procwise --no-pager --since "1 min ago" | grep -q "AgentNick is ready"; do sleep 5; done
```

- [ ] **Step 2: Verify bp_products table exists**

```sql
SELECT count(*) FROM proc.bp_products;
```

Should return 0 initially.

- [ ] **Step 3: Re-extract all quote Excel files**

Reset the process_monitor records for the quote_scenario xlsx files and the HR Quote:
```sql
UPDATE proc.process_monitor
SET status = 'Completed', start_ts = NULL, end_ts = NULL
WHERE file_path LIKE '%quote_scenario%' OR file_path LIKE '%HR Quote%';
```

Wait for extractions to complete. Check logs for:
- Correct supplier identification (e.g., "SupplyX Ltd" for quote_scenarios, "PeopleFirst HR Solutions Ltd" for HR Quote)
- All line items extracted (8 for quote_scenario_1, 10 for HR Quote)
- item_id populated on every line item
- bp_products table populated

- [ ] **Step 4: Verify data accuracy**

```sql
-- Check line items have item_id populated
SELECT quote_id, quote_line_id, item_id, item_description, quantity, unit_price, line_total
FROM proc.bp_quote_line_items
WHERE quote_id LIKE 'QTE%'
ORDER BY quote_id, line_number;

-- Check products were created
SELECT product_id, item_description, current_unit_price, currency, occurrence_count
FROM proc.bp_products
ORDER BY product_id;

-- Check no zero-line-item headers
SELECT q.quote_id, q.total_amount, count(li.quote_line_id) as lines
FROM proc.bp_quote q
LEFT JOIN proc.bp_quote_line_items li ON li.quote_id = q.quote_id
GROUP BY q.quote_id, q.total_amount
HAVING count(li.quote_line_id) = 0;
```

- [ ] **Step 5: Re-extract PO files and verify**

Reset PO process_monitor records and re-extract. Verify:
- Correct buyer/supplier (buyer = company placing order, supplier = vendor)
- All line items extracted with item_id
- Products that appeared in quotes now match in POs (same product_id)

- [ ] **Step 6: Commit any fixes needed**

```bash
git add -A
git commit -m "fix: extraction accuracy improvements from end-to-end testing"
```

---

### Task 6: Discrepancy Logging (Validation Only, No Auto-Correction)

**Files:**
- Verify: `src/services/extraction_validator.py` — confirm no auto-correction of source values

- [ ] **Step 1: Verify discrepancies are logged, not corrected**

Check that `_validate_amounts` and `_validate_line_item_totals` in `extraction_validator.py`:
- Log discrepancies when tax_amount ≠ total_amount × tax_percent / 100
- Log discrepancies when line_total ≠ quantity × unit_price
- Log discrepancies when sum(line_totals) ≠ header total_amount
- **Never mutate** `header[field]` or `item[field]` to "correct" values
- Source values from the document are preserved exactly as extracted

- [ ] **Step 2: Verify in bp_discrepancy_data table**

After re-extraction, check that discrepancies are recorded:
```sql
SELECT doc_type, field_name, rule_name, severity, extracted_value, expected_value, message
FROM proc.bp_discrepancy_data
WHERE doc_type = 'Purchase_Order'
ORDER BY created_date DESC
LIMIT 20;
```

- [ ] **Step 3: Commit if any adjustments needed**

---

### Summary

| Task | What it does | Key files |
|------|-------------|-----------|
| 1 | Create bp_products + ProductCatalogService | `product_catalog_service.py` |
| 2 | Smart Excel structure detection | `direct_extraction_service.py:_tabular_to_text` |
| 3 | Stronger extraction prompts | `direct_extraction_service.py:_build_extraction_prompt` |
| 4 | Wire product catalog into pipeline | `agent_nick_orchestrator.py`, `direct_extraction_service.py` |
| 5 | End-to-end testing with samples | Manual verification |
| 6 | Verify discrepancy logging only | `extraction_validator.py` |
