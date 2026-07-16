-- SpendIQ Admin configuration store.
-- Org-policy config (authority bands, procurement-cycle phases) has no source in the
-- extraction pipeline, so it lives in its own key/JSONB table. Seeded with the SpendIQ
-- prototype defaults; the Admin screen reads these back and PUTs edits here.
CREATE TABLE IF NOT EXISTS proc.bp_admin_config (
    config_key         text PRIMARY KEY,
    config_value       jsonb       NOT NULL,
    last_modified_by   text,
    last_modified_date timestamptz NOT NULL DEFAULT now()
);

INSERT INTO proc.bp_admin_config (config_key, config_value, last_modified_by)
VALUES
  ('authority_bands', '[
     {"band":"Officer / Requester","limit":0,"scope":"Own cost centre","adds":"Raise only — no approval"},
     {"band":"Team Manager","limit":10000,"scope":"Own cost centre","adds":"Budget check"},
     {"band":"Senior Manager","limit":50000,"scope":"Department cost centres","adds":"+ Finance notify"},
     {"band":"Director","limit":250000,"scope":"Function cost centres","adds":"+ Finance gate"},
     {"band":"VP / C-suite","limit":1000000,"scope":"All cost centres","adds":"+ CFO sign-off"},
     {"band":"Board","limit":-1,"scope":"All cost centres","adds":"Two-person sign-off"}
   ]'::jsonb, 'seed'),
  ('cycles', '{
     "demand":{"name":"Demand lifecycle","phases":["Request","Analysis","Requirement","Sourcing","Purchase order","Realisation"]},
     "sourcing":{"name":"Sourcing cycle","phases":["RFx issued","Responses received","Evaluated","Awarded"]},
     "approval":{"name":"Approval cycle","phases":["Submitted","In review","Approved","Conditions attached"]},
     "review":{"name":"Review stages","phases":["Accepted","Review","Completion"]}
   }'::jsonb, 'seed')
ON CONFLICT (config_key) DO NOTHING;
