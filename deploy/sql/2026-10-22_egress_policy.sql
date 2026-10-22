-- 1.20 (b) / D-62: which outside destinations each purpose and agent may call is an administrator's
-- policy row. FOR REVIEW. NOT APPLIED to any database. Idempotent: writes the starting row only if there
-- is none. The same row is written by POST /admin/egress-policy/seed (services.egress_policy.seed).
-- SHADOW: the egress gate records what this row would decide on every outbound call and blocks nothing;
-- enforcement per purpose is a later step. Local destinations never consult this row.
BEGIN;

INSERT INTO proc.bp_policy (
    policy_name, policy_type, policy_desc, policy_details,
    policy_linked_agents, policy_status, version, created_date, created_by
)
SELECT 'EgressPolicy', 'security',
       'Which outside destinations each purpose and agent may call. Local destinations are always allowed. A deny rule beats an allow rule; anything unmatched is denied. In shadow mode the decision is recorded on every egress event and nothing is blocked.',
       $json${
  "policy_identifier": "egress",
  "history": [
    {
      "version": 1,
      "changed_by": "migration:2026-10-22_egress_policy",
      "field": "rules"
    }
  ],
  "rules": {
    "allow": [
      {
        "purpose": "fx_rates",
        "hosts": [
          "open.er-api.com"
        ],
        "agents": [
          "*"
        ],
        "note": "currency reference rates; no company text in the request"
      },
      {
        "purpose": "mailbox",
        "hosts": [
          "graph.microsoft.com",
          "login.microsoftonline.com"
        ],
        "agents": [
          "*"
        ],
        "max_classification": "confidential",
        "note": "the company's own Microsoft 365 mailbox: addresses and message text belong there"
      },
      {
        "purpose": "supplier_research",
        "hosts": [
          "*"
        ],
        "agents": [
          "*"
        ],
        "max_classification": "public",
        "note": "public web for supplier research: supplier names and generic words only; the address guard still refuses private hosts"
      }
    ],
    "deny": [
      {
        "purpose": "model_inference",
        "hosts": [
          "api.ollama.com"
        ],
        "agents": [
          "*"
        ],
        "note": "D-62: company text never goes to Ollama cloud"
      }
    ],
    "enforce": []
  }
}$json$::jsonb,
       '', 1, 1, now(), 'migration:2026-10-22_egress_policy'
WHERE NOT EXISTS (
    SELECT 1 FROM proc.bp_policy WHERE policy_name = 'EgressPolicy' AND policy_status = 1
)
ON CONFLICT (policy_type, policy_name) WHERE policy_status = 1 DO NOTHING;

COMMIT;
