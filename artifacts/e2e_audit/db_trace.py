"""Read-only trace of the live data tiers: raw -> stg -> trgt -> deal views.
NO writes. Validates promotion, deal linking, reconciliation, dashboards."""
from __future__ import annotations
import json, os, sys
sys.path.insert(0, "/home/muthu/PycharmProjects/BP_Backend/src")
sys.path.insert(0, "/home/muthu/PycharmProjects/BP_Backend")
os.chdir("/home/muthu/PycharmProjects/BP_Backend")
from src.services.db import get_conn

def q(cur, sql, args=None):
    cur.execute(sql, args or ())
    cols = [c[0] for c in cur.description] if cur.description else []
    return [dict(zip(cols, r)) for r in cur.fetchall()]

def show(title, rows, limit=20):
    print(f"\n{'='*70}\n{title}\n{'='*70}")
    if not rows:
        print("  (no rows)"); return
    for r in rows[:limit]:
        print("  " + json.dumps(r, default=str))
    if len(rows) > limit:
        print(f"  ... +{len(rows)-limit} more")

with get_conn() as conn:
    cur = conn.cursor()
    cur.execute("SET TRANSACTION READ ONLY")

    # tier counts
    print("="*70 + "\nTIER ROW COUNTS\n" + "="*70)
    for t in ["bp_quote_raw","bp_quote_stg","bp_quote_trgt",
              "bp_purchase_order_raw","bp_purchase_order_stg","bp_purchase_order_trgt",
              "bp_invoice_raw","bp_invoice_stg","bp_invoice_trgt",
              "bp_invoice_line_items_stg","bp_invoice_line_items_trgt",
              "bp_po_line_items_trgt","bp_quote_line_items_trgt"]:
        try:
            cur.execute(f"SELECT count(*) FROM proc.{t}")
            print(f"  proc.{t:32}: {cur.fetchone()[0]}")
        except Exception as e:
            conn.rollback(); cur.execute("SET TRANSACTION READ ONLY")
            print(f"  proc.{t:32}: ERR {str(e)[:50]}")

    # raw promotion status
    for t in ["bp_invoice_raw","bp_purchase_order_raw","bp_quote_raw"]:
        try:
            show(f"{t}: promotion_status + doc_pk", q(cur,
                f"SELECT raw_id, doc_pk_candidate, promotion_status, pipeline_version, extracted_at "
                f"FROM proc.{t} ORDER BY raw_id"))
        except Exception as e:
            conn.rollback(); cur.execute("SET TRANSACTION READ ONLY"); print(f"  {t}: ERR {str(e)[:80]}")

    # deal overview
    show("bp_deal_overview (one row per deal)", q(cur, "SELECT * FROM proc.bp_deal_overview"))
    # deal documents
    try:
        show("bp_deal_documents", q(cur, "SELECT deal_id, doc_type, doc_pk, doc_number, amount, currency, converted_amount_usd, confidence_score, status FROM proc.bp_deal_documents ORDER BY deal_id, doc_type"))
    except Exception as e:
        conn.rollback(); cur.execute("SET TRANSACTION READ ONLY"); print("deal_documents ERR", str(e)[:100])
    # deal kpis
    try:
        show("bp_deal_kpis (executive)", q(cur, "SELECT * FROM proc.bp_deal_kpis"))
    except Exception as e:
        conn.rollback(); cur.execute("SET TRANSACTION READ ONLY"); print("deal_kpis ERR", str(e)[:100])

    # trgt rows w/ deal columns -> verify SQL trigger populated deal_id
    for t in ["bp_invoice_trgt","bp_purchase_order_trgt","bp_quote_trgt"]:
        try:
            show(f"{t}: deal linkage", q(cur,
                f"SELECT deal_id, deal_name, document_id, deal_date, confidence_score FROM proc.{t}"))
        except Exception as e:
            conn.rollback(); cur.execute("SET TRANSACTION READ ONLY"); print(f"{t} ERR", str(e)[:80])

    # reconciliation actions
    try:
        show("bp_agent_actions: recent reconciliation", q(cur,
            "SELECT deal_id, action_type, field_name, status, summary, created_at "
            "FROM proc.bp_agent_actions WHERE action_type LIKE 'reconcile%' OR phase='reconcile' "
            "ORDER BY created_at DESC LIMIT 25"))
    except Exception as e:
        conn.rollback(); cur.execute("SET TRANSACTION READ ONLY"); print("recon ERR", str(e)[:100])

    # action type distribution
    try:
        show("bp_agent_actions: action_type distribution", q(cur,
            "SELECT action_type, count(*) n FROM proc.bp_agent_actions GROUP BY action_type ORDER BY n DESC"), limit=40)
    except Exception as e:
        conn.rollback(); cur.execute("SET TRANSACTION READ ONLY"); print("dist ERR", str(e)[:100])

    # opportunities
    try:
        show("bp_opportunity", q(cur,
            "SELECT opportunity_id, detector_type, supplier_name, item_description, financial_impact_gbp, realised_savings_gbp, stage, deal_id, quote_id FROM proc.bp_opportunity ORDER BY detected_on DESC"))
    except Exception as e:
        conn.rollback(); cur.execute("SET TRANSACTION READ ONLY"); print("opp ERR", str(e)[:100])

    # governance
    try:
        show("bp_prompt", q(cur, "SELECT prompt_id, prompt_name, prompt_type, prompts_status FROM proc.bp_prompt ORDER BY prompt_id"))
        show("bp_policy", q(cur, "SELECT policy_id, policy_name, policy_type, policy_status FROM proc.bp_policy ORDER BY policy_id"))
    except Exception as e:
        conn.rollback(); cur.execute("SET TRANSACTION READ ONLY"); print("gov ERR", str(e)[:100])

print("\n[db_trace] DONE")
