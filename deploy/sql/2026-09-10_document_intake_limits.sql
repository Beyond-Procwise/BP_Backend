-- P4: how much may enter the product in one upload, stated in policy.
--
-- POST /document/embed-document accepted List[UploadFile] with an extension
-- allow-list and nothing else -- no count cap, no size cap. The limits belong
-- here rather than in a constant or an environment variable for the same reason
-- the rest of the guardrail does: they change what the product accepts, and a
-- change to that should be versioned, attributable and visible on a governance
-- screen rather than a redeploy or a shell export nobody reviews.
--
-- #807 DocumentIntakeAuthorityPolicy already states WHO may put documents in
-- (required_role Buyer, applies_to document.upload/document.promote). HOW MUCH
-- belongs on the same row: it is the same authority, quantified. The row keeps
-- its applies_to and its effect, so P7's gate-visibility invariants still hold.
--
-- A MISSING OR UNUSABLE VALUE REFUSES (api/routers/documents._intake_limits).
-- An unconfigured cap is not an unlimited one, so a deployment that has not run
-- this migration finds out by being unable to upload rather than by being
-- unable to stop an upload.
--
-- The numbers: 25 MiB per file comfortably clears the scanned invoices, POs and
-- contracts this corpus actually carries, and 20 files is a generous
-- multi-select. Both are starting points a customer is meant to change -- which
-- is the point of them being a policy row.
BEGIN;

UPDATE proc.bp_policy
   SET policy_details = jsonb_set(
           jsonb_set(policy_details,
                     '{rules,max_files_per_request}', '20'::jsonb, true),
           '{rules,max_bytes_per_file}', '26214400'::jsonb, true),
       version = version + 1,
       last_modified_by = 'document_intake_limits',
       last_modified_date = now()
 WHERE policy_status = 1
   AND policy_details->>'policy_identifier' = 'document_intake_authority'
   AND (policy_details->'rules'->>'max_files_per_request' IS DISTINCT FROM '20'
     OR policy_details->'rules'->>'max_bytes_per_file' IS DISTINCT FROM '26214400');

COMMIT;
