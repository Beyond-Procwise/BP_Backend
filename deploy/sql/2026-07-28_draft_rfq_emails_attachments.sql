-- deploy/sql/2026-07-28_draft_rfq_emails_attachments.sql
BEGIN;
-- Attachments a human added to a draft before sending. The SMTP layer has always
-- been able to attach files (EmailService.send_email -> MIMEBase per file) and
-- EmailDispatchService.send_draft has always accepted them; there was simply
-- nowhere to record one, so no UI could offer it.
--
-- Bytes live in S3 under an email-attachments/ prefix -- deliberately NOT the
-- data-integration document path, which would ingest the file into the extraction
-- pipeline and raise findings against a supplier's own signed PDF.
--
-- Shape: [{"filename","content_type","bytes","s3_key","added_by","added_at"}]
ALTER TABLE proc.draft_rfq_emails
  ADD COLUMN IF NOT EXISTS attachments JSONB;
COMMIT;
