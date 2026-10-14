-- Fresh-extraction reset: truncate the document pipeline + deal artifacts.
--
-- FK-safe: bp_*_raw.process_monitor_id -> proc.process_monitor(id) has no
-- ON DELETE CASCADE, and bp_*_line_items_raw -> bp_*_raw do. Listing every
-- pipeline table in ONE TRUNCATE ... CASCADE lets Postgres resolve the order.
-- RESTART IDENTITY resets the raw_id / process_monitor_id sequences.
--
-- PRESERVED (not truncated): governance (bp_prompt, bp_policy), learned models
-- (bp_extraction_patterns/template/type_priors), supplier master (bp_supplier),
-- workflow/email tables, routing. Add them below if you want a deeper reset.
--
-- proc.bp_agent_actions was in the list below and has been REMOVED: it is the audit
-- spine, and 2026-09-16_bp_agent_actions_immutable.sql makes it append-only, TRUNCATE
-- included. Reloading a corpus is no reason to erase the record that the previous run
-- happened, and "the reset script did it" is exactly how an audit log goes missing
-- without anyone deciding to lose it. Adding it back here will now fail loudly rather
-- than quietly emptying the table.
--
-- Run inside a transaction; verify counts, then COMMIT (or ROLLBACK to abort).
BEGIN;

TRUNCATE TABLE
    proc.process_monitor,
    proc.bp_invoice_raw,  proc.bp_invoice_stg,  proc.bp_invoice_trgt,
    proc.bp_invoice_line_items_raw, proc.bp_invoice_line_items_stg, proc.bp_invoice_line_items_trgt,
    proc.bp_quote_raw,    proc.bp_quote_stg,    proc.bp_quote_trgt,
    proc.bp_quote_line_items_raw,   proc.bp_quote_line_items_stg,   proc.bp_quote_line_items_trgt,
    proc.bp_purchase_order_raw, proc.bp_purchase_order_stg, proc.bp_purchase_order_trgt,
    proc.bp_po_line_items_raw, proc.bp_po_line_items_stg, proc.bp_po_line_items_trgt,
    proc.bp_contract_raw, proc.bp_contracts,
    proc.bp_deal_document_map,
    proc.extraction_review_queue,
    proc.bp_extraction_discrepancy,
    proc.bp_extraction_provenance, proc.bp_extraction_provenance_v3,
    proc.bp_extraction_hallucination_audit,
    proc.bp_extraction_retry_log,
    proc.bp_extraction_observation,
    proc.bp_summary, proc.bp_analysis_summary
RESTART IDENTITY CASCADE;

-- Sanity check before committing:
--   SELECT count(*) FROM proc.process_monitor;        -- expect 0
--   SELECT count(*) FROM proc.bp_invoice_trgt;        -- expect 0

COMMIT;
