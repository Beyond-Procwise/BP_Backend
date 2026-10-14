-- The public path (GET /i18n/public/{lang}) must be rolled back first.
BEGIN;
DROP TABLE IF EXISTS proc.bp_i18n_public_key;
COMMIT;
