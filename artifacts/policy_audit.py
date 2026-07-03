"""Read-only audit of proc.bp_policy in the live bp_sqldb.

Proves the governance gap: how many policies are actually enforced (hard gate)
vs advisory/prompt-only, and whether the `version` column / approval lifecycle
is used at all. Does NOT modify any data.
"""
import json
import os

import psycopg2

# Load .env without extra deps
ENV = {}
with open(os.path.join(os.path.dirname(__file__), "..", ".env")) as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            ENV[k.strip()] = v.strip().strip('"').strip("'")

conn = psycopg2.connect(
    host=ENV["DB_HOST"], dbname=ENV["DB_NAME"], user=ENV["DB_USER"],
    password=ENV["DB_PASSWORD"], port=ENV["DB_PORT"],
)
cur = conn.cursor()

print("=" * 78)
print("1. COLUMNS ON proc.bp_policy (does an approval lifecycle exist?)")
print("=" * 78)
cur.execute(
    """
    SELECT column_name, data_type, column_default
    FROM information_schema.columns
    WHERE table_schema='proc' AND table_name='bp_policy'
    ORDER BY ordinal_position
    """
)
cols = cur.fetchall()
for name, dtype, default in cols:
    print(f"  {name:24} {dtype:14} default={default}")
colnames = {c[0] for c in cols}
lifecycle = {"approved_by", "approval_date", "effective_from", "effective_to",
             "supersedes", "approval_status", "state"}
print("\n  Governance-lifecycle columns present:",
      sorted(lifecycle & colnames) or "NONE")

print("\n" + "=" * 78)
print("2. EVERY POLICY: version / status / how many versions per name")
print("=" * 78)
cur.execute(
    """
    SELECT policy_id, policy_name, policy_type, version, policy_status,
           policy_linked_agents, last_modified_by
    FROM proc.bp_policy ORDER BY policy_name, version
    """
)
rows = cur.fetchall()
print(f"  Total rows: {len(rows)}\n")
print(f"  {'id':>3} {'name':32} {'type':16} {'ver':>3} {'st':>2} linked_agents")
distinct_versions = {}
for pid, name, ptype, ver, st, linked, lmb in rows:
    distinct_versions.setdefault(name, set()).add(ver)
    print(f"  {pid:>3} {str(name)[:32]:32} {str(ptype)[:16]:16} {ver:>3} {st:>2} {linked}")

multi = {n: v for n, v in distinct_versions.items() if len(v) > 1}
print(f"\n  Policy names with >1 version on record: {multi or 'NONE'}")
print(f"  All versions == 1? {all(v == {1} for v in distinct_versions.values())}")
print(f"  Any non-active status (status != 1)? "
      f"{any(r[4] != 1 for r in rows)}")
print(f"  Distinct last_modified_by values: "
      f"{sorted({r[6] for r in rows})}")

print("\n" + "=" * 78)
print("3. policy_details: is there machine-enforceable rule structure?")
print("=" * 78)
cur.execute("SELECT policy_name, policy_details FROM proc.bp_policy ORDER BY policy_name")
for name, details in cur.fetchall():
    if isinstance(details, str):
        try:
            details = json.loads(details)
        except Exception:
            pass
    keys = list(details.keys()) if isinstance(details, dict) else type(details).__name__
    print(f"  {str(name)[:32]:32} detail keys: {keys}")

cur.close()
conn.close()
print("\nDone (read-only).")
