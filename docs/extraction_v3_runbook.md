# Extraction V3 Ops Runbook

## 1. What It Is

Extraction V3 is the new schema-driven pipeline that replaces AgentNick's LLM extraction calls for invoice, purchase order, quote, and contract documents. It comprises three layers: a Universal Parser that extracts text and layout coordinates, a Multi-Model Candidate Generation layer that runs document-category-specific extractors (LayoutLMv3, table transformers, QA models, vendor templates, semantic search, NER) in parallel, and a Schema-Bound Judge that resolves conflicts and enforces database invariants. All computation runs local-on-GPU; no external APIs. The Judge uses LLM as a layered last resort for grounded disambiguation when signal-based methods (evidence proximity, schema coherence, inventory checks) cannot decide. No regex.

## 2. Per-Category Feature Flag

Enable extraction_v3 on a per-document-category basis via environment variables:

```bash
# Default: agentnick (legacy)
EXTRACTION_PIPELINE_INVOICE=agentnick
EXTRACTION_PIPELINE_PURCHASE_ORDER=agentnick
EXTRACTION_PIPELINE_QUOTE=agentnick
EXTRACTION_PIPELINE_CONTRACT=agentnick
```

To enable v3 for invoices:

```bash
# In .env or systemd Environment= directive
EXTRACTION_PIPELINE_INVOICE=v3
```

After setting the flag, either:
- Restart procwise service, OR
- (Preferred for live systems) Wait for the next document ingestion — the pipeline flag is checked at lifespan initialization.

**DO NOT restart when a loop monitor (e.g., `process_monitor_watcher.py`) is actively observing extraction queues.** Coordinate with ops to pause the watcher first.

## 3. Audit: proc.bp_extraction_provenance_v3

One row per committed field. Schema:

| Column | Purpose |
|--------|---------|
| `doc_type` | invoice, purchase_order, quote, contract |
| `doc_pk` | Primary key (e.g., `INV-005-41`) |
| `field_path` | Dot-notation field (e.g., `vendor.name`, `line_items.0.amount`) |
| `value` | The final value committed to `proc.bp_<doc_type>.<column>` |
| `evidence_text` | Verbatim substring of source document |
| `model` | Extractor that produced the candidate (`layoutlmv3`, `table_transformer`, `qa_roberta`, `vendor_template`, `sbert_anchor`, `spacy_ner`) |
| `judge_actions` | JSONB array of actions taken: `["tiebreaker"]`, `["grounded_last_resort"]`, `["schema_coherence"]`, `["inventory_check"]`, etc. |
| `final_confidence` | Score after judge demotions and invariant enforcement |
| `pipeline_version` | Iteration identifier (e.g., `v3.0.1`) |

**Audit example:**

```sql
SELECT field_path, value, evidence_text, model, final_confidence, judge_actions
FROM proc.bp_extraction_provenance_v3
WHERE doc_pk = 'INV-005-41'
ORDER BY field_path;
```

This shows which model extracted each field, what evidence grounded it, and any judge interventions.

## 4. Adding a New Field to a YAML Schema

1. **Create DB migration** in `scripts/migrations/`:
   ```sql
   ALTER TABLE proc.bp_invoice ADD COLUMN my_new_field TEXT;
   ```

2. **Update YAML schema** at `extraction_schemas/invoice.yaml`:
   ```yaml
   - name: my_new_field
     type: string
     required: false
     db_column: my_new_field
     canonical_labels: ["My Label", "MyLabel"]
     extractors: [layoutlmv3, qa_roberta, sbert_anchor]
     judge:
       grounded_last_resort: true
   ```

3. **Restart procwise** (or wait for next process). The lifespan safety check verifies YAML ↔ DB alignment and fails loud if they diverge.

## 5. Live Smoke Test

```bash
.venv/bin/python scripts/extraction_v3_live_smoke.py --n 20
```

**Success criteria:**

- `DOCS_WITH_HALLUCINATIONS` **must be 0** (P0 if not).
- `avg_judge_calls` ≤ 3 per document (per spec §13).
- `p95_latency_s` < 25 seconds (per spec §13).

**Note:** If documents are S3-only and not on local disk, the script reports `file not found` — this is expected and does not indicate failure.

## 6. Known Limitations (Plan 1)

- **LayoutLMv3 fine-tuning** not included; uses pre-trained `microsoft/layoutlmv3-base`. Accuracy on novel layouts is limited; canonical-label proximity baseline handles common patterns. Fine-tuning lands in **Plan 2**.
- **PO, Quote, Contract schemas** are stubs; full schemas in **Plan 3**.
- **Live-shadow comparison and cutover decision** deferred to **Plan 4**.
- **Email-body-only invoices** still routed through AgentNick.
- **Legacy orchestrator** (`agent_nick_orchestrator.py`) is preserved, not deleted, allowing rapid flag-back if needed.

## 7. Hallucination Guarantee (Architectural)

Every committed value's `evidence_text` **must** be a verbatim substring of `ParsedDocument.full_text`. This is enforced at three layers:

1. **Layer 2 (candidate generation):** Each model emits evidence verified against full text.
2. **Judge (grounded-last-resort):** Safety check 2 verifies substring before committing.
3. **Persist:** Provenance writes reject rows with evidence mismatches.

Verified by integration test: `tests/extraction_v3/test_judge_grounded_last_resort.py::test_PROPERTY_no_random_fabrication_committed` (100/100 fabrications rejected in latest run).

**Any hallucination found in production is a P0 bug, not a tunable parameter.** Report immediately to the extraction team.
