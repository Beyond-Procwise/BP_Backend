BEGIN;
-- deploy/sql/2026-07-27_bp_prompt_style_drafting.sql
--
-- Phase 4. The system prompt for style-governed drafting, seeded into the existing
-- governance registry rather than hardcoded in Python.
--
-- Two paragraphs carry the design.
--
-- The precedence paragraph exists because a model shown three complete emails and a list
-- of abstract rules will imitate the emails — they are concrete and the rules are not —
-- and a specification that loses to its own illustrations is not governing anything.
--
-- The "invent nothing" paragraph was added after a live run. Asked to quote a named
-- supplier for 25 desks, the model produced a fluent, correctly-voiced email addressed to
-- a contact who does not exist, citing a PO number it made up, against a quote deadline
-- nobody set. Fluency is precisely what makes that dangerous: the draft looked ready to
-- send. A visible [placeholder] is the correct output for a fact the task did not supply.
--
-- Being able to correct this wording without a deploy is why it lives in bp_prompt.
-- services/style/drafting.py carries a byte-identical fallback for when the database is
-- unreachable, and reports prompt version 0 when it uses it, so a draft is never credited
-- to a governed version it did not run under. A test asserts the two are identical.
--
-- Idempotent: inserts when absent, updates and bumps the version when the wording has
-- changed, and is a no-op when it already matches.

CREATE TEMP TABLE _style_draft_prompt (template TEXT) ON COMMIT DROP;
INSERT INTO _style_draft_prompt VALUES ($tmpl$You are drafting an email on behalf of a specific person, in their voice.

You are given a style specification describing how they write, then two or three example emails, then the task.

Where the example emails and the style specification conflict, follow the style specification. The examples illustrate the specification; they do not override it.

The examples are fiction. Their suppliers, amounts, reference numbers, names and dates are invented and must never appear in your draft. Take only the manner of writing from them; take every fact from the task.

Invent nothing. Every name, figure, date, reference number and commitment in your draft must come from the task. If the task does not give you something the email seems to need — the contact's name, a reference number, a deadline — write a square-bracketed placeholder such as [contact name] or [reference] and carry on. A placeholder is correct. A plausible-looking invention is not, and is worse than leaving the gap visible.

Write only the email. Give a subject line, then a blank line, then the body, and end with the sign-off the specification requires. No preamble, no commentary, no explanation of your choices.$tmpl$);

UPDATE proc.bp_prompt p
   SET prompts_desc       = jsonb_set(
                               COALESCE(p.prompts_desc, '{}'::jsonb),
                               '{prompt_template}',
                               to_jsonb((SELECT template FROM _style_draft_prompt)),
                               true),
       version            = COALESCE(p.version, 1) + 1,
       last_modified_date = now(),
       last_modified_by   = 'style-drafting'
 WHERE p.prompt_name = 'style_draft_system'
   AND COALESCE(p.prompts_desc->>'prompt_template', '')
       IS DISTINCT FROM (SELECT template FROM _style_draft_prompt);

INSERT INTO proc.bp_prompt (prompt_name, prompt_type, prompt_linked_agents, prompts_desc)
SELECT 'style_draft_system', 'email_prompt', 'email_drafting',
       jsonb_build_object('prompt_template', (SELECT template FROM _style_draft_prompt))
WHERE NOT EXISTS (
    SELECT 1 FROM proc.bp_prompt WHERE prompt_name = 'style_draft_system'
);

COMMIT;
