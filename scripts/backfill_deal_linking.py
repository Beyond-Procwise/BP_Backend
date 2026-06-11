#!/usr/bin/env python3
"""One-time deal-linking backfill: apply DDL, run assignment over existing rows,
create views, print a reconciliation report. Idempotent; no deletes."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

def _env():
    env = {}
    for line in open(os.path.join(ROOT, ".env")):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1); env[k.strip()] = v.strip().strip('"').strip("'")
    for k_pg, k_db in [("PGHOST","DB_HOST"),("PGDATABASE","DB_NAME"),("PGUSER","DB_USER"),
                       ("PGPASSWORD","DB_PASSWORD"),("PGPORT","DB_PORT")]:
        if env.get(k_db): os.environ.setdefault(k_pg, env[k_db])
    os.environ.setdefault("PGSSLMODE", "require")

def main():
    _env()
    import psycopg2
    conn = psycopg2.connect(host=os.environ["PGHOST"], dbname=os.environ["PGDATABASE"],
        user=os.environ["PGUSER"], password=os.environ["PGPASSWORD"],
        port=os.environ.get("PGPORT","5432"), sslmode="require")
    conn.autocommit = False
    cur = conn.cursor()
    cur.execute(open(os.path.join(ROOT, "deploy/sql/2026-06-11_deal_linking.sql")).read())
    conn.commit()
    from src.services.deal_assignment_service import assign_deals
    result = assign_deals(conn=conn); conn.commit()
    cur.execute(open(os.path.join(ROOT, "deploy/sql/2026-06-11_deal_views.sql")).read()); conn.commit()
    print("ASSIGN:", result)
    for v in ("bp_deal_overview", "bp_deal_kpis"):
        cur.execute(f"select count(*) from proc.{v}"); print(v, cur.fetchone()[0])
    cur.execute("select deal_id, deal_name, quote_count, po_count, invoice_count, deal_date "
                "from proc.bp_deal_overview order by deal_id")
    for r in cur.fetchall(): print("  DEAL", r)
    conn.close()

if __name__ == "__main__":
    main()
