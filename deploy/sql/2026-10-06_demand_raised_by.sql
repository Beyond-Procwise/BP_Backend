-- 2026-10-06  Who raised a demand, recorded by the server.
--
-- proc.bp_demand.requester is whatever name the browser sent, so it cannot decide who may close or
-- edit a demand: anyone can type anyone's name. These two columns are written by the gateway from
-- the verified sign-in token and never from the request, and they are not part of the payload the
-- browser sends back. A demand with no value here predates this (or was raised by a caller whose
-- token could not be verified): it has no recorded requester, so only an approver or admin can
-- close it.
--
-- Additive and idempotent. Safe to re-run.
ALTER TABLE proc.bp_demand
    ADD COLUMN IF NOT EXISTS raised_by_sub   TEXT,
    ADD COLUMN IF NOT EXISTS raised_by_email TEXT;

COMMENT ON COLUMN proc.bp_demand.raised_by_sub IS
    'Cognito subject of whoever raised this demand, from the verified token. NULL = not recorded.';
