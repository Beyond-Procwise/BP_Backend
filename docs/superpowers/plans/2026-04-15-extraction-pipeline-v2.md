# Extraction Pipeline v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Redesign extraction pipeline for 100% accuracy with smart text windowing, vendor profile learning, supplier auto-creation, and schema fixes.

**Architecture:** Three-phase pipeline: Phase 1 (document intelligence — section detection, smart text builder, vendor lookup), Phase 2 (context-aware LLM call with layered prompts), Phase 3 (source verification, supplier resolution, smart computation, anomaly logging). Single LLM call per document, GPU-accelerated.

**Tech Stack:** Python 3.12, PostgreSQL (proc schema), Ollama (qwen3:30b MoE), FastAPI, pdfplumber, openpyxl

**Spec:** `docs/superpowers/specs/2026-04-15-extraction-pipeline-v2-design.md`

---

## File Structure

| File | Responsibility |
|------|---------------|
| `scripts/schema_migration_v2.sql` (CREATE) | SQL migration for PO columns + vendor_profile table + typo fix |
| `src/services/document_intelligence.py` (CREATE) | Section detector + smart text builder |
| `src/services/direct_extraction_service.py` (MODIFY) | Updated TABLE_SCHEMAS, smart text integration |
| `src/services/agent_nick_orchestrator.py` (MODIFY) | Phase 1/3 integration, supplier auto-creation, vendor profile injection |
| `src/services/extraction_validator.py` (MODIFY) | Value anchoring, line item completeness check |
| `src/services/process_monitor_watcher.py` (MODIFY) | Already done — pre-KG gate, training collection, dedup |

---

### Task 1: Schema Migration

**Files:**
- Create: `scripts/schema_migration_v2.sql`

- [ ] **Step 1: Write migration SQL**

```sql
-- PO header: add tax/total columns to match Invoice/Quote pattern
ALTER TABLE proc.bp_purchase_order ADD COLUMN IF NOT EXISTS tax_percent numeric;
ALTER TABLE proc.bp_purchase_order ADD COLUMN IF NOT EXISTS tax_amount numeric;
ALTER TABLE proc.bp_purchase_order ADD COLUMN IF NOT EXISTS total_amount_incl_tax numeric;
ALTER TABLE proc.bp_purchase_order ADD COLUMN IF NOT EXISTS supplier_id text;

-- Fix typo in PO line items
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.columns
               WHERE table_schema='proc' AND table_name='bp_po_line_items'
               AND column_name='unit_of_measue') THEN
        ALTER TABLE proc.bp_po_line_items RENAME COLUMN unit_of_measue TO unit_of_measure;
    END IF;
END $$;

-- Vendor profile table for extraction pattern learning
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

- [ ] **Step 2: Run migration**

Run: `PYTHONPATH=src:. .venv/bin/python -c "import psycopg2; from config.settings import settings; conn = psycopg2.connect(host=settings.db_host, port=settings.db_port, dbname=settings.db_name, user=settings.db_user, password=settings.db_password); conn.autocommit=True; cur=conn.cursor(); cur.execute(open('scripts/schema_migration_v2.sql').read()); print('Migration complete'); conn.close()"`

- [ ] **Step 3: Update TABLE_SCHEMAS in direct_extraction_service.py**

Add new PO columns to the `Purchase_Order` schema dict and fix the typo:
- Add `tax_percent`, `tax_amount`, `total_amount_incl_tax`, `supplier_id` to `header_columns`
- Change `unit_of_measue` → `unit_of_measure` in `line_columns`

- [ ] **Step 4: Verify columns exist**

Run: `PYTHONPATH=src:. .venv/bin/python -c "import psycopg2; from config.settings import settings; conn=psycopg2.connect(host=settings.db_host, port=settings.db_port, dbname=settings.db_name, user=settings.db_user, password=settings.db_password); cur=conn.cursor(); cur.execute(\"SELECT column_name FROM information_schema.columns WHERE table_schema='proc' AND table_name='bp_purchase_order' AND column_name IN ('tax_percent','tax_amount','total_amount_incl_tax','supplier_id')\"); print([r[0] for r in cur.fetchall()]); conn.close()"`

Expected: `['tax_percent', 'tax_amount', 'total_amount_incl_tax', 'supplier_id']`

---

### Task 2: Document Intelligence Module

**Files:**
- Create: `src/services/document_intelligence.py`

- [ ] **Step 1: Create the section detector**

```python
"""Document intelligence — section detection and smart text building.

Analyzes raw document text to identify structural sections (header,
line items, summary, payment) and builds optimized text for LLM
extraction with priority ordering that never truncates line items.
"""
from __future__ import annotations
import re
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# Section detection markers
_HEADER_MARKERS = re.compile(
    r"(invoice\s*(no|number|#|id)|po\s*(no|number|#)|quote\s*(no|number|#|id)|"
    r"quotation|purchase\s*order|bill\s*to|invoice\s*to|billed\s*to|from:|vendor:|supplier:)",
    re.IGNORECASE,
)
_LINE_ITEM_HEADERS = re.compile(
    r"(description|item|qty|quantity|unit\s*price|price|amount|total|cost|rate|"
    r"unit\s*cost|line\s*total|ext\.?\s*price|net\s*amount)",
    re.IGNORECASE,
)
_SUMMARY_MARKERS = re.compile(
    r"^[\s]*(sub[\-\s]?total|subtotal|total\s*(before|excl)|net\s*total|"
    r"tax|vat|gst|grand\s*total|total\s*(incl|due|payable|amount)|amount\s*due|balance)",
    re.IGNORECASE | re.MULTILINE,
)
_PAYMENT_MARKERS = re.compile(
    r"(bank\s*(name|account|code)|sort\s*code|iban|swift|bic|"
    r"payment\s*(info|details|method|terms)|remittance|account\s*(no|number))",
    re.IGNORECASE,
)


@dataclass
class DocumentSection:
    name: str
    start_line: int
    end_line: int
    priority: int  # lower = higher priority for text building


@dataclass
class DocumentStructure:
    sections: List[DocumentSection] = field(default_factory=list)
    total_lines: int = 0

    def get_section(self, name: str) -> Optional[DocumentSection]:
        for s in self.sections:
            if s.name == name:
                return s
        return None


def detect_sections(text: str) -> DocumentStructure:
    """Detect structural sections in document text."""
    lines = text.split("\n")
    total = len(lines)
    structure = DocumentStructure(total_lines=total)

    if total == 0:
        return structure

    # Find line item table boundaries
    item_start = None
    item_end = None
    summary_start = None
    payment_start = None

    for i, line in enumerate(lines):
        # Detect line item table header (2+ header keywords on same line)
        if item_start is None:
            matches = _LINE_ITEM_HEADERS.findall(line)
            if len(matches) >= 2:
                item_start = i
                continue

        # After line items started, detect summary section
        if item_start is not None and item_end is None:
            if _SUMMARY_MARKERS.search(line):
                item_end = i
                summary_start = i
                continue

        # Detect payment section
        if _PAYMENT_MARKERS.search(line) and payment_start is None:
            payment_start = i

    # Build sections
    header_end = (item_start or summary_start or payment_start or min(20, total)) 
    structure.sections.append(DocumentSection("header", 0, header_end, priority=2))

    if item_start is not None:
        end = item_end or summary_start or payment_start or total
        structure.sections.append(DocumentSection("line_items", item_start, end, priority=1))

    if summary_start is not None:
        end = payment_start or total
        structure.sections.append(DocumentSection("summary", summary_start, end, priority=2))

    if payment_start is not None:
        structure.sections.append(DocumentSection("payment", payment_start, total, priority=4))

    return structure


def build_smart_text(text: str, max_chars: int = 6000) -> str:
    """Build optimized text for LLM extraction.

    Priority ordering ensures line items are NEVER truncated:
    1. All line items (priority 1)
    2. Header + summary (priority 2)
    3. Remaining content (priority 3+)
    4. Boilerplate dropped last
    """
    lines = text.split("\n")
    structure = detect_sections(text)

    if not structure.sections:
        # No structure detected — return as much as fits
        return text[:max_chars]

    # Sort sections by priority
    sorted_sections = sorted(structure.sections, key=lambda s: s.priority)

    parts = []
    total_chars = 0

    for section in sorted_sections:
        section_lines = lines[section.start_line:section.end_line]
        section_text = "\n".join(section_lines)

        if total_chars + len(section_text) <= max_chars:
            tag = section.name.upper().replace("_", " ")
            parts.append(f"[{tag}]\n{section_text}")
            total_chars += len(section_text) + len(tag) + 4
        elif section.priority <= 2:
            # High priority — include even if over limit
            tag = section.name.upper().replace("_", " ")
            parts.append(f"[{tag}]\n{section_text}")
            total_chars += len(section_text)
        # else: drop low-priority sections

    result = "\n\n".join(parts)

    # If no sections captured enough, fall back to raw text
    if len(result) < 200:
        return text[:max_chars]

    return result


def count_source_line_items(text: str) -> int:
    """Count the number of line item rows in the source text.

    Used for completeness verification after extraction.
    """
    structure = detect_sections(text)
    items_section = structure.get_section("line_items")
    if not items_section:
        return 0

    lines = text.split("\n")
    item_lines = lines[items_section.start_line + 1:items_section.end_line]

    # Count non-empty lines that look like data rows (have numbers)
    count = 0
    number_pattern = re.compile(r"\d")
    for line in item_lines:
        stripped = line.strip()
        if stripped and number_pattern.search(stripped) and len(stripped) > 10:
            count += 1

    return count
```

- [ ] **Step 2: Verify syntax**

Run: `PYTHONPATH=src:. .venv/bin/python -c "import ast; ast.parse(open('src/services/document_intelligence.py').read()); print('OK')"`

---

### Task 3: Integrate Smart Text Builder into Extraction

**Files:**
- Modify: `src/services/agent_nick_orchestrator.py`
- Modify: `src/services/direct_extraction_service.py`

- [ ] **Step 1: Replace static text[:6000] with smart text builder in orchestrator**

In `_build_enhanced_text()`, use the document intelligence module:

```python
@staticmethod
def _build_enhanced_text(text: str, filename: str, doc_type: str) -> str:
    from services.document_intelligence import build_smart_text
    hint = (
        f"FILENAME: {filename}\n"
        f"(The filename may contain the document ID, supplier name, "
        f"and related document references — use as context.)\n\n"
    )
    smart_text = build_smart_text(text, max_chars=6000)
    return hint + smart_text
```

- [ ] **Step 2: Remove the static text[:6000] in direct_extraction_service.py prompt builder**

In `_build_extraction_prompt()`, the document text is already pre-processed by `_build_enhanced_text()` in the orchestrator. Change the prompt to use `{text}` instead of `{text[:6000]}` since the text is already optimized.

- [ ] **Step 3: Verify syntax**

Run: `PYTHONPATH=src:. .venv/bin/python -c "import ast; ast.parse(open('src/services/agent_nick_orchestrator.py').read()); ast.parse(open('src/services/direct_extraction_service.py').read()); print('OK')"`

---

### Task 4: Vendor Profile Learning + Injection

**Files:**
- Modify: `src/services/agent_nick_orchestrator.py`

- [ ] **Step 1: Add vendor profile lookup and injection**

Add method `_get_vendor_context()` to AgentNickOrchestrator:

```python
def _get_vendor_context(self, supplier_hint: str) -> str:
    """Look up vendor profile and return context string for LLM prompt."""
    if not supplier_hint or len(supplier_hint) < 3:
        return ""
    try:
        conn = self._agent_nick.get_db_connection()
        try:
            with conn.cursor() as cur:
                # Check vendor_profile table
                cur.execute(
                    "SELECT profile_data, extraction_count, avg_confidence "
                    "FROM proc.vendor_profile WHERE supplier_name ILIKE %s LIMIT 1",
                    (f"%{supplier_hint}%",),
                )
                row = cur.fetchone()
                if not row:
                    return ""
                profile = row[0] if isinstance(row[0], dict) else {}
                count = row[1] or 0
                if count < 2:
                    return ""  # Not enough history
                parts = [f"VENDOR CONTEXT (from {count} previous extractions):"]
                if profile.get("default_currency"):
                    parts.append(f"- Currency: {profile['default_currency']}")
                if profile.get("typical_tax_rate"):
                    parts.append(f"- Typical tax rate: {profile['typical_tax_rate']}%")
                if profile.get("date_format_hint"):
                    parts.append(f"- Date format: {profile['date_format_hint']}")
                if profile.get("id_pattern"):
                    parts.append(f"- Document ID pattern: {profile['id_pattern']}")
                return "\n".join(parts)
        finally:
            conn.close()
    except Exception:
        logger.debug("Vendor profile lookup failed", exc_info=True)
        return ""
```

- [ ] **Step 2: Inject vendor context into extraction**

In `_dispatch_extraction()`, before the LLM call:

```python
# Extract supplier hint from filename
supplier_hint = self._extract_supplier_from_filename(file_path)
vendor_context = self._get_vendor_context(supplier_hint or "")
if vendor_context:
    enhanced_text = vendor_context + "\n\n" + enhanced_text
```

- [ ] **Step 3: Update vendor profile after successful extraction**

Add method `_update_vendor_profile()`:

```python
def _update_vendor_profile(self, header: dict, doc_type: str) -> None:
    """Update vendor profile with patterns from this extraction."""
    supplier = header.get("supplier_id") or header.get("supplier_name") or ""
    if not supplier or len(supplier) < 3:
        return
    try:
        import json
        conn = self._agent_nick.get_db_connection()
        try:
            with conn.cursor() as cur:
                # Fetch existing profile
                cur.execute(
                    "SELECT profile_data, extraction_count, avg_confidence "
                    "FROM proc.vendor_profile WHERE supplier_name = %s",
                    (supplier,),
                )
                row = cur.fetchone()
                profile = row[0] if row and isinstance(row[0], dict) else {}
                count = (row[1] or 0) if row else 0
                old_conf = (row[2] or 0) if row else 0

                # Update profile with current extraction patterns
                profile["default_currency"] = header.get("currency") or profile.get("default_currency")
                if header.get("tax_percent"):
                    profile["typical_tax_rate"] = float(header["tax_percent"])
                profile["last_doc_type"] = doc_type

                new_count = count + 1
                conf = header.get("confidence_score", 0) or 0
                new_avg = ((old_conf * count) + conf) / new_count if new_count > 0 else 0

                cur.execute("""
                    INSERT INTO proc.vendor_profile
                        (supplier_name, profile_data, extraction_count, avg_confidence, last_extraction, last_modified_date)
                    VALUES (%s, %s, %s, %s, NOW(), NOW())
                    ON CONFLICT (supplier_name) DO UPDATE SET
                        profile_data = %s,
                        extraction_count = %s,
                        avg_confidence = %s,
                        last_extraction = NOW(),
                        last_modified_date = NOW()
                """, (supplier, json.dumps(profile), new_count, round(new_avg, 3),
                      json.dumps(profile), new_count, round(new_avg, 3)))
        finally:
            conn.close()
    except Exception:
        logger.debug("Vendor profile update failed", exc_info=True)
```

Call it in `process_document()` after successful extraction (replace existing `_learn_vendor_profile` call).

---

### Task 5: Supplier Auto-Creation

**Files:**
- Modify: `src/services/agent_nick_orchestrator.py`

- [ ] **Step 1: Add supplier auto-creation method**

```python
def _auto_create_supplier(self, header: dict, doc_type: str) -> Optional[str]:
    """Auto-create a new supplier in bp_supplier if not found.

    Only creates from explicitly extracted document values — never guesses.
    Returns the new supplier_id or None.
    """
    supplier_name = header.get("supplier_id") or header.get("supplier_name") or ""
    if not supplier_name or len(supplier_name) < 3:
        return None

    try:
        conn = self._agent_nick.get_db_connection()
        try:
            with conn.cursor() as cur:
                # Check if already exists
                cur.execute(
                    "SELECT supplier_id FROM proc.bp_supplier "
                    "WHERE LOWER(supplier_name) = LOWER(%s) OR LOWER(trading_name) = LOWER(%s) "
                    "LIMIT 1",
                    (supplier_name, supplier_name),
                )
                if cur.fetchone():
                    return None  # Already exists

                # Generate supplier_id
                clean = re.sub(r"[^a-zA-Z0-9]", "", supplier_name)
                supplier_id = f"SUP-{clean[:20]}"

                # Insert with only document-provided values
                cur.execute("""
                    INSERT INTO proc.bp_supplier
                        (supplier_id, supplier_name, trading_name, default_currency,
                         country, address_line1, city, postal_code,
                         contact_email_1, contact_phone_1,
                         created_date, created_by)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), %s)
                    ON CONFLICT DO NOTHING
                """, (
                    supplier_id, supplier_name, supplier_name,
                    header.get("currency"),
                    header.get("country"),
                    header.get("supplier_address", "").split(",")[0] if header.get("supplier_address") else None,
                    None, None,
                    header.get("contact_email"), header.get("contact_phone"),
                    "AgentNick-AutoDiscovery",
                ))

                # Also insert into proc.supplier (keep both tables in sync)
                cur.execute("""
                    INSERT INTO proc.supplier
                        (supplier_id, supplier_name, trading_name, default_currency,
                         created_date, created_by)
                    VALUES (%s, %s, %s, %s, NOW(), %s)
                    ON CONFLICT DO NOTHING
                """, (supplier_id, supplier_name, supplier_name,
                      header.get("currency"), "AgentNick-AutoDiscovery"))

                logger.info(
                    "[AgentNick] Auto-created supplier: %s → %s",
                    supplier_name, supplier_id,
                )
                return supplier_id
        finally:
            conn.close()
    except Exception:
        logger.debug("Supplier auto-creation failed", exc_info=True)
        return None
```

- [ ] **Step 2: Integrate into process_document flow**

In `process_document()`, after `_resolve_supplier()`, if supplier still not resolved:

```python
# Auto-create supplier if not found
if not header.get("supplier_id") or header.get("supplier_id") == header.get("supplier_name"):
    new_id = self._auto_create_supplier(header, doc_type)
    if new_id:
        header["supplier_id"] = new_id
```

---

### Task 6: Value Anchoring + Line Item Completeness

**Files:**
- Modify: `src/services/extraction_validator.py`

- [ ] **Step 1: Add value anchoring to Pass 3**

In `_pass3_confidence_scoring()`, the source-text verification already exists. Enhance it to log unanchored values:

```python
# After the existing source-text check, add:
if source_text and str(value).strip():
    val_str = str(value).strip()
    found_in_source = (
        val_str in source_text
        or val_str.replace("-", "").replace(" ", "") in source_text.replace("-", "").replace(" ", "")
    )
    if not found_in_source and field not in ("currency", "payment_terms", "country", "region"):
        discrepancies.append(Discrepancy(
            field_name=field,
            rule_name="value_not_anchored",
            severity="info",
            extracted_value=val_str,
            pass_number=3,
            source="value_anchoring",
            message=f"Extracted value '{val_str}' not found in source text — verify accuracy",
        ))
```

- [ ] **Step 2: Add line item completeness check**

In `validate_and_correct()`, after Pass 3, add:

```python
# Pass 4: Line item completeness check
from services.document_intelligence import count_source_line_items
source_item_count = count_source_line_items(source_text)
extracted_item_count = len(line_items)
if source_item_count > 0 and extracted_item_count < source_item_count:
    all_discrepancies.append(Discrepancy(
        field_name="line_items",
        rule_name="line_item_incomplete",
        severity="warning",
        extracted_value=str(extracted_item_count),
        expected_value=str(source_item_count),
        pass_number=4,
        source="completeness_check",
        message=f"Extracted {extracted_item_count} line items but source has ~{source_item_count} rows",
    ))
    logger.warning(
        "[Validator] Line item gap: extracted %d but source has ~%d rows",
        extracted_item_count, source_item_count,
    )
```

---

### Task 7: Restart + End-to-End Test

- [ ] **Step 1: Verify all syntax**

Run: `PYTHONPATH=src:. .venv/bin/python -c "import ast; [ast.parse(open(f).read()) for f in ['src/services/document_intelligence.py','src/services/direct_extraction_service.py','src/services/agent_nick_orchestrator.py','src/services/extraction_validator.py','src/services/process_monitor_watcher.py']]; print('All OK')"`

- [ ] **Step 2: Run schema migration**

- [ ] **Step 3: Restart server**

Kill uvicorn, let systemd restart with new code.

- [ ] **Step 4: Reset test records and trigger extraction**

Reset 3-5 records covering different doc types (Invoice, PO, Quote) and monitor for:
- Smart text windowing working (section tags in logs)
- Vendor profile being created/updated
- Supplier auto-creation for new suppliers
- PO extracting tax_amount and total_amount_incl_tax separately
- Line item completeness check logging
- Value anchoring logging unanchored values
- Confidence >= 0.92
- No data modification (source values preserved)

- [ ] **Step 5: Verify database**

Check:
- `proc.vendor_profile` has entries
- `proc.bp_purchase_order` has `tax_amount` and `total_amount_incl_tax` populated
- `proc.bp_supplier` has auto-created entries
- `proc.bp_discrepancy_data` has value_anchoring and completeness entries
