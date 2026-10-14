-- Removes exactly what 2026-10-02_document_type_order_form_sales_order.sql adds.
--
-- After this, 'order form' and 'sales order' are unknown upload categories again
-- and routing REFUSES them, which is the pre-2026-10-02 behaviour.
--
-- bp_document_type first: its concept_code is a foreign key into bp_concept.
BEGIN;

DELETE FROM proc.bp_document_type
 WHERE concept_code IN ('doctype.order_form', 'doctype.sales_order');

UPDATE proc.bp_concept
   SET not_to_be_confused_with = array_remove(not_to_be_confused_with, 'doctype.order_form')
 WHERE concept_code = 'doctype.call_off_contract';

UPDATE proc.bp_concept
   SET not_to_be_confused_with = array_remove(not_to_be_confused_with, 'doctype.sales_order')
 WHERE concept_code = 'doctype.order';

DELETE FROM proc.bp_concept
 WHERE concept_code IN ('doctype.order_form', 'doctype.sales_order');

COMMIT;
