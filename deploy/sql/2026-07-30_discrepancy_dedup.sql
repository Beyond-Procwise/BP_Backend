-- Duplicate-finding guard for proc.bp_extraction_discrepancy.
--
-- Re-processing a document created a fresh raw_id and re-inserted the same
-- findings; the Test Data_300726 audit found identical open rows stacked up
-- to 7 deep (94 duplicated keys corpus-wide). Identity of an OPEN finding is
-- (doc_type, doc_pk_candidate, issue_type, field_name); resolved rows are
-- history and stay outside the constraint.

-- 1. Collapse existing open duplicates, keeping the first-raised row.
DELETE FROM proc.bp_extraction_discrepancy d
USING proc.bp_extraction_discrepancy k
WHERE coalesce(d.status, 'open') <> 'resolved'
  AND coalesce(k.status, 'open') <> 'resolved'
  AND d.doc_type = k.doc_type
  AND coalesce(d.doc_pk_candidate, '') = coalesce(k.doc_pk_candidate, '')
  AND d.issue_type = k.issue_type
  AND coalesce(d.field_name, '') = coalesce(k.field_name, '')
  AND k.discrepancy_id < d.discrepancy_id;

-- 2. Close stale missing_required rows whose document has since been
--    re-extracted and promoted with the required amount present (the 20
--    formula-blind xlsx docs from ses-20260730-UJF3 left these behind).
UPDATE proc.bp_extraction_discrepancy d
   SET status = 'resolved',
       resolved_at = now(),
       resolution_action = 'dismiss',
       resolved_by = 'dedup-migration',
       notes = coalesce(d.notes, '')
               || ' [auto-resolved: the document has since been re-extracted and promoted with the value present]'
 WHERE coalesce(d.status, 'open') <> 'resolved'
   AND d.issue_type = 'missing_required'
   AND (
        EXISTS (SELECT 1 FROM proc.bp_quote_trgt q
                 WHERE q.quote_id = d.doc_pk_candidate
                   AND q.total_amount_incl_tax IS NOT NULL)
     OR EXISTS (SELECT 1 FROM proc.bp_invoice_trgt i
                 WHERE i.invoice_id = d.doc_pk_candidate
                   AND i.invoice_amount IS NOT NULL)
     OR EXISTS (SELECT 1 FROM proc.bp_purchase_order_trgt p
                 WHERE p.po_id = d.doc_pk_candidate
                   AND p.total_amount IS NOT NULL)
   );

-- 3. Withdraw "charged but not on the PO" claims made against rows with no
--    money (computed_value NULL = no amount was on the line). The check now
--    requires a charge before it calls something charged; these rows were
--    document furniture ("Commercial terms", footers) raised before that fix.
UPDATE proc.bp_extraction_discrepancy d
   SET status = 'resolved',
       resolved_at = now(),
       resolution_action = 'dismiss',
       resolved_by = 'dedup-migration',
       notes = coalesce(d.notes, '')
               || ' [auto-resolved: the line carries no amount, so it is not a charge]'
 WHERE coalesce(d.status, 'open') <> 'resolved'
   AND d.issue_type = 'line_not_on_po'
   AND d.computed_value IS NULL;

-- 4. Enforce the open-findings key; write_discrepancies upserts against it.
CREATE UNIQUE INDEX IF NOT EXISTS ix_bp_extraction_discrepancy_open_key
    ON proc.bp_extraction_discrepancy
       (doc_type, coalesce(doc_pk_candidate, ''),
        issue_type, coalesce(field_name, ''))
 WHERE coalesce(status, 'open') <> 'resolved';
