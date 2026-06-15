# CSV/Excel Extraction — Design Spec

## Problem

CSV and Excel files uploaded via the UI are tracked in `proc.process_monitor` but the `DataExtractionAgent` only handles PDF/DOCX/images. Tabular files (csv, xlsx) are silently dropped. Additionally, the `category` field from `process_monitor` is not threaded into the extraction pipeline, losing valuable routing context.

## Solution

Add a tabular document processing path to `DataExtractionAgent` that:
1. Reads CSV/Excel files with pandas
2. Uses `process_monitor.category` to select the target DB schema
3. Fuzzy-matches CSV columns to schema columns using the existing synonym system
4. Persists matched data via the existing staging/merge pattern
5. For unrecognized categories, ingests data into Neo4j as a supplier-centric knowledge graph

## Category-to-Schema Mapping

| `process_monitor.category` | Target Table | Line Items Table |
|---|---|---|
| `invoice` | `proc.invoice_agent` | `proc.invoice_line_items_agent` |
| `po` | `proc.purchase_order_agent` | `proc.po_line_items_agent` |
| `quote` / `quotes` | `proc.quote_agent` | `proc.quote_line_items_agent` |
| `contract` | `proc.contracts` | — |
| Any other (e.g., `spend`, `item`) | Neo4j Knowledge Graph | — |

## Architecture

```
process_monitor record (status=Completed, category=po, document_type=csv)
    |
    v
ProcessMonitorWatcher._process_record()
    |  passes: file_path, category, document_type, user_id
    v
orchestrator.execute_extraction_flow(
    s3_object_key=file_path,
    category=category,
    document_type=document_type,
    user_id=user_id,
)
    |
    v
DataExtractionAgent._process_single_document()
    |
    +-- .csv/.xlsx --> _process_tabular_document()
    |       |
    |       +-- Read file with pandas (read_csv / read_excel)
    |       +-- Resolve target schema from category
    |       |     known category --> DOC_TYPE_TO_TABLE --> DB insert
    |       |     unknown category --> Neo4j KG ingestion
    |       +-- Fuzzy-match columns using existing synonym system
    |       +-- Persist via _merge_from_staging() (existing pattern)
    |       +-- Vectorize into Qdrant for RAG
    |
    +-- .pdf/.docx/.png --> Existing pipeline (unchanged)
```

## Column Mapping Engine

For known categories:

1. Read file into pandas DataFrame
2. Get target schema from `CATEGORY_TO_DOC_TYPE[category]` then `DOC_TYPE_TO_TABLE[doc_type]`
3. Get schema definition from `PROCUREMENT_SCHEMAS[table_name]`
4. For each CSV column header:
   - Normalize: lowercase, strip whitespace, replace spaces/hyphens with underscore
   - Exact match against schema columns: use it
   - Synonym match against schema field synonyms (threshold 0.55): use best match
   - No match: skip column, log as unmapped
5. Validate required fields present (from `schema.required_fields`)
   - All present: proceed
   - Missing: log warning, proceed with available fields
6. For each row:
   - Apply type coercion (numeric, date, text)
   - Insert via existing `_merge_from_staging()` pattern
   - ON CONFLICT update (same as existing document extraction)

Multi-sheet Excel handling:
- Read all sheets
- If sheet name matches a known doc type (e.g., "invoices", "line_items"), route to that schema
- Otherwise treat each sheet as independent data with same category

Duplicate detection:
- Use primary key from schema (e.g., `invoice_id`, `po_id`) for conflict resolution
- If CSV has the PK column: upsert (ON CONFLICT DO UPDATE)
- If CSV lacks PK: generate composite key from first 3 non-null columns

## Neo4j Knowledge Graph Ingestion (Unrecognized Categories)

Supplier-centric graph with rich properties:

```
For each CSV row:
    +-- Extract supplier identifiers (supplier_id, supplier_name, vendor, manufacturer)
    |   MERGE (s:Supplier {supplier_id: ...})
    |   SET s.name, s.contact, s.country, s.region, ...
    |
    +-- Create data node with category-derived label
    |   MERGE (n:SpendRecord {row_id: ...})
    |   SET n.category_id, n.amount, n.currency, ...
    |
    +-- Create relationship to Supplier
    |   MERGE (s)-[:HAS_SPEND]->(n)
    |
    +-- Detect and link FK relationships:
    |   contract_id --> MERGE (c:Contract) --> (n)-[:UNDER_CONTRACT]->(c)
    |   po_id --> MERGE (p:PurchaseOrder) --> (n)-[:LINKED_TO_PO]->(p)
    |   category_id --> MERGE (cat:Category) --> (n)-[:IN_CATEGORY]->(cat)
    |   item_id --> MERGE (i:Item) --> (n)-[:FOR_ITEM]->(i)
    |
    +-- Vectorize text representation into Qdrant
        "{supplier_name} | {category} | {amount} | {description} | ..."
```

Supplier parameter capture:

| Field Pattern | Captured As | Node Property |
|---|---|---|
| `supplier_id`, `vendor_id` | Supplier node PK | `supplier_id` |
| `supplier_name`, `vendor`, `manufacturer` | Supplier identity | `name` |
| `preferred_supplier_id` | Supplier preference flag | `is_preferred: true` |
| `country`, `region` | Supplier geography | `country`, `region` |
| `currency` | Supplier trading currency | `trading_currency` |
| `standard_price`, `unit_price` | Pricing on relationship | relationship property |
| `brand`, `manufacturer` | Product-supplier link | `(Supplier)-[:MANUFACTURES]->(Item)` |

No supplier data is lost. If a column contains supplier-identifiable information, it becomes a property on the Supplier node or a relationship.

## Threading Category Through the Pipeline

Currently `execute_extraction_flow()` only accepts `s3_prefix` and `s3_object_key`. Changes:

- `ProcessMonitorWatcher._process_record()`: pass `category`, `document_type`, `user_id` to orchestrator
- `orchestrator.execute_extraction_flow()`: add `category`, `document_type`, `user_id` kwargs (optional, backward compatible)
- `DataExtractionAgent._process_single_document()`: read `category` from `context.input_data["category"]`, branch on file extension

## Error Handling

| Scenario | Behavior |
|---|---|
| Empty file (0 rows) | Log warning, mark `Extracted` with `total_count=0` |
| No columns match schema (known category) | Log warning, push all data to Neo4j as fallback |
| Missing required fields | Proceed with available fields, log which required fields are missing |
| Corrupt Excel file (bad ZIP) | Mark `Extraction_Failed`, log error |
| Mixed data types in column | Coerce what's possible, set failed cells to NULL |
| File too large (>50MB) | Process in chunks (10,000 rows per batch) |
| Encoding issues (non-UTF8 CSV) | Try UTF-8, Latin-1, CP1252 fallback chain |

## Files Changed

| File | Change | Description |
|---|---|---|
| `src/agents/data_extraction_agent.py` | Modify | Add `_process_tabular_document()`, branch on file extension in `_process_single_document()` |
| `src/services/process_monitor_watcher.py` | Modify | Pass `category`, `document_type`, `user_id` to orchestrator |
| `src/orchestration/orchestrator.py` | Modify | Add `category`, `document_type`, `user_id` kwargs to `execute_extraction_flow()` |
| `src/services/kg_ingestion_service.py` | New | Neo4j ingestion for unrecognized categories |
| `utils/procurement_schema.py` | Modify | Add `CATEGORY_TO_DOC_TYPE` mapping and category alias normalization |

## Non-Goals

- No changes to existing PDF/DOCX/image extraction pipeline
- No LLM-assisted column mapping (strict fuzzy match only)
- No auto-creation of new DB columns for unmapped CSV columns
- No retry mechanism for failed extractions
