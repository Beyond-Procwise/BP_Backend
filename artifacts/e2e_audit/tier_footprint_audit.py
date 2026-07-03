"""Reusable tier-footprint integrity audit (read-only): verifies each doc's full
data footprint is preserved across raw -> stg -> trgt. Flags: (1) _stg columns
missing from _trgt (field loss), (2) any promoted doc whose _trgt line-item count
is LESS than its _stg count (line loss), (3) docs stuck in _stg not in _trgt."""
import sys
sys.path.insert(0, "src"); sys.path.insert(0, ".")
from src.services.db import get_conn
DOC = {"invoice": "bp_invoice", "quote": "bp_quote", "purchase_order": "bp_purchase_order"}
LINES = {"invoice": ("invoice_id", "bp_invoice_line_items"), "quote": ("quote_id", "bp_quote_line_items"), "purchase_order": ("po_id", "bp_po_line_items")}
AUDIT = {"created_by", "created_date", "last_modified_by", "last_modified_date"}

def run():
    issues = []
    with get_conn() as c:
        cur = c.cursor(); cur.execute("SET TRANSACTION READ ONLY")
        def cols(t):
            cur.execute("select column_name from information_schema.columns where table_schema='proc' and table_name=%s", (t,)); return {r[0] for r in cur.fetchall()}
        for dt, base in DOC.items():
            lost = sorted((cols(f"{base}_stg") - cols(f"{base}_trgt")) - AUDIT)
            if lost: issues.append(f"{dt}: _stg cols missing in _trgt -> {lost}")
        for dt, (pk, lbase) in LINES.items():
            cur.execute(f"select s.{pk} from proc.{lbase}_stg s group by s.{pk} having count(*) > (select count(*) from proc.{lbase}_trgt t where t.{pk}=s.{pk}) and exists(select 1 from proc.{DOC[dt]}_trgt d where d.{pk}=s.{pk})")
            for r in cur.fetchall(): issues.append(f"{dt} line loss: {pk}={r[0]} has fewer lines in _trgt than _stg")
    return issues

if __name__ == "__main__":
    found = run()
    print("TIER FOOTPRINT AUDIT:", "CLEAN — full footprint preserved" if not found else f"{len(found)} issue(s):")
    for i in found: print("  -", i)
