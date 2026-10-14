-- Quote templates: the shapes a seller can start a customer quote from.
--
-- A template is STRUCTURE, not content: which lines this kind of offer usually has,
-- what each is measured in, and which terms need an answer. It holds no price, no
-- cost, no discount and no legal wording -- those are the seller's to give, on the
-- quote. The quote_drafting agent reads the template to know what a finished quote of
-- this kind needs, fills in what the seller tells it, and says what is still missing.
--
--   body = {"category": text|null,
--           "lines": [{"key", "tag": product|service|one-off, "description", "unit"}],
--           "termFields": [payment|start|lengthMonths|renewal|sla ...],
--           "assumptionsHtml": text|null}
--
-- bp_sales_quote.template_key records which template a quote was started from, so the
-- agent can still read it on a later turn. It is a note of origin, not a constraint:
-- the seller may add, remove and reword lines freely.
--
-- The four rows below are starters, there so the list is not empty on day one. They
-- are ordinary rows: an Admin can reword, retire or add to them
-- (PUT /spendiq/quote-templates/{key}), and re-running this file never overwrites an
-- edit. Additive and idempotent. Apply to BOTH bp_testdb and bp_sqldb, before the code
-- that reads it.

BEGIN;

CREATE TABLE IF NOT EXISTS proc.bp_sales_quote_template (
    template_id        bigserial PRIMARY KEY,
    template_key       text        NOT NULL,
    template_name      text        NOT NULL,
    description        text,
    body               jsonb       NOT NULL,
    status             text        NOT NULL DEFAULT 'active',   -- active | retired
    created_by         text,
    created_date       timestamptz NOT NULL DEFAULT now(),
    last_modified_by   text,
    last_modified_date timestamptz NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS ix_bp_sales_quote_template_key
    ON proc.bp_sales_quote_template (template_key);

ALTER TABLE proc.bp_sales_quote
    ADD COLUMN IF NOT EXISTS template_key text;

INSERT INTO proc.bp_sales_quote_template (template_key, template_name, description, body, created_by, last_modified_by)
VALUES
('software_subscription', 'Software subscription',
 'Licences for a period, with the work to get the customer live.',
 '{"category": null,
   "lines": [
     {"key": "licences",   "tag": "product", "description": "Subscription licences",  "unit": "seat"},
     {"key": "onboarding", "tag": "service", "description": "Onboarding and set-up",  "unit": "engagement"},
     {"key": "training",   "tag": "service", "description": "Training",               "unit": "day"}],
   "termFields": ["payment", "start", "lengthMonths", "renewal", "sla"],
   "assumptionsHtml": "<p><b>Assumes:</b> </p><p><b>Excludes:</b> </p>"}'::jsonb,
 'starter', 'starter'),
('professional_services', 'Professional services',
 'Time-based work delivered by people: consulting, a project, an implementation.',
 '{"category": null,
   "lines": [
     {"key": "delivery",   "tag": "service", "description": "Project delivery",   "unit": "day"},
     {"key": "management", "tag": "service", "description": "Project management", "unit": "day"}],
   "termFields": ["payment", "start"],
   "assumptionsHtml": "<p><b>Assumes:</b> </p><p><b>Excludes:</b> </p>"}'::jsonb,
 'starter', 'starter'),
('product_supply', 'Product supply',
 'Goods supplied, with delivery and installation where they apply.',
 '{"category": null,
   "lines": [
     {"key": "goods",        "tag": "product", "description": "Goods",        "unit": "unit"},
     {"key": "delivery",     "tag": "one-off", "description": "Delivery",     "unit": "delivery"},
     {"key": "installation", "tag": "service", "description": "Installation", "unit": "job"}],
   "termFields": ["payment", "start"],
   "assumptionsHtml": "<p><b>Assumes:</b> </p><p><b>Excludes:</b> </p>"}'::jsonb,
 'starter', 'starter'),
('managed_service', 'Managed service',
 'An ongoing service charged by the month, with a one-off set-up.',
 '{"category": null,
   "lines": [
     {"key": "monthly_fee", "tag": "service", "description": "Monthly service fee", "unit": "month"},
     {"key": "setup",       "tag": "one-off", "description": "Set-up",              "unit": "engagement"}],
   "termFields": ["payment", "start", "lengthMonths", "renewal", "sla"],
   "assumptionsHtml": "<p><b>Assumes:</b> </p><p><b>Excludes:</b> </p>"}'::jsonb,
 'starter', 'starter')
ON CONFLICT (template_key) DO NOTHING;

-- Who may change the templates every seller starts from.
INSERT INTO proc.bp_policy (
    policy_name, policy_type, policy_desc, policy_details,
    policy_linked_agents, policy_status, version,
    created_date, created_by, last_modified_date, last_modified_by
)
SELECT 'QuoteTemplateWriteAuthorityPolicy', 'authority',
       'Who may add, reword or retire the templates a customer quote is started from. '
       'No agent changes them.',
       jsonb_build_object(
         'policy_identifier', 'quote_template_write_authority',
         'applies_to', jsonb_build_array('quote_template.write'),
         'required_role', 'Admin',
         'rules', jsonb_build_object('effect', 'allow')),
       '', 1, 1, now(), 'sales_quote_template', now(), 'sales_quote_template'
 WHERE NOT EXISTS (
    SELECT 1 FROM proc.bp_policy p
     WHERE p.policy_details->>'policy_identifier' = 'quote_template_write_authority'
       AND p.policy_status = 1);

-- The governed instruction for a template-filling turn. prompt_type 'extraction', like
-- sales_quote_drafting; a missing row fails the turn and no text in code stands in.
INSERT INTO proc.bp_prompt (prompt_name, prompt_type, prompt_linked_agents, prompts_desc,
                            created_by, last_modified_by)
SELECT
    'sales_quote_template_fill',
    'extraction',
    'quote_drafting',
    jsonb_build_object('prompt_template',
'You help a seller fill in a customer quote that was started from a template. You do not edit the quote yourself. You read what the seller tells you and say which values go where.

Today is {today}.
The template: {template}
The quote so far (line ids and what each line is; you are not shown its prices): {quote}
What the seller says: {message}

The changes you may name, as "op":
- set_line: fill in one line. Give "line_id", and any of "qty", "unit_price", "cost" and "percent" (a discount) that the seller stated for that line. "unit_price" is what the customer is charged for one; a figure given for a line is its unit_price unless the seller says it is what THEY pay. "cost" is only what the seller says it costs them ("costs us 125"); never give a cost the seller did not call a cost, and never copy the price into it.
- set_customer: who the quote is for. Give "name" and/or "contact", copied exactly as the seller wrote them.
- set_term: one term. Give "field" (payment, start, lengthMonths, renewal or sla) and its value: "text" for payment, renewal and sla, copied exactly as the seller wrote it; "date" as YYYY-MM-DD for start; "number" for lengthMonths.
- valid_until: only if the seller says how long the quote is valid. Give "date" as YYYY-MM-DD; turn "valid for 30 days" into a date counted from today.
- add_line: something the seller mentions that no line covers. Give "description", and "qty", "unit" and "tag" (product, service or one-off) only if stated.
- markup_all / markup_line / discount_all / discount_line: a markup or discount given as a percentage. Give "percent", and "line_id" for one line.

Rules:
- Use only values the seller states. Never estimate, never fill a gap, never repeat a number from anywhere else. A line the seller says nothing about gets no change.
- Match what the seller says to the line it is about by meaning (for example "seats" or "licences" to the subscription line). Use only a line_id from the quote above. If you cannot tell which line a figure belongs to, leave it out.
- Money is a plain number with no currency sign. "18k" is 18000.
- A percentage "off" or a "discount" is a discount on the line it was said about: give it as "percent" on that line''s set_line. It is never a payment, renewal or service-level term.
- If the seller states nothing that fits the quote, return an empty "intents" list and say why in "refused". Otherwise set "refused" to an empty string.

Reply with only JSON: {"refused": "", "intents": [{"op": "...", ...}]}'),
    'deploy/sql/2026-10-13_bp_sales_quote_template.sql',
    'deploy/sql/2026-10-13_bp_sales_quote_template.sql'
WHERE NOT EXISTS (
    SELECT 1 FROM proc.bp_prompt WHERE prompt_name = 'sales_quote_template_fill'
);

COMMIT;
