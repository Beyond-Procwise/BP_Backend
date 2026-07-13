import sys; sys.path.insert(0,'/home/muthu/PycharmProjects/BP_Backend')
from dotenv import load_dotenv; load_dotenv('/home/muthu/PycharmProjects/BP_Backend/.env')
from src.services.db import get_conn

def q(cur, label, sql, args=None):
    print(f"\n### {label}")
    try:
        cur.execute(sql, args or ())
        rows = cur.fetchall()
        cols = [d.name if hasattr(d,'name') else d[0] for d in (cur.description or [])]
        print("  cols:", cols)
        for r in rows[:40]: print("  ", r)
        if len(rows) > 40: print(f"  ...(+{len(rows)-40})")
    except Exception as e:
        print("  ERR:", e)

with get_conn() as c:
    cur=c.cursor()
    # latest telemetry row per document (dedupe multi-run)
    latest = """with r as (
        select *, row_number() over (partition by doc_type, coalesce(doc_pk,file_path)
                  order by captured_at desc) rn
        from proc.bp_extraction_telemetry)
    select {sel} from r where rn=1"""
    q(cur,"DISTINCT docs: completeness (latest per doc)",
      latest.format(sel="completeness_status, count(*)")+" group by 1 order by 2 desc")
    q(cur,"DISTINCT docs: status (latest per doc)",
      latest.format(sel="status, count(*)")+" group by 1 order by 2 desc")
    q(cur,"DISTINCT docs: null confidence (latest)",
      latest.format(sel="(confidence is null) conf_null, count(*)")+" group by 1")
    q(cur,"DISTINCT docs: zero header/line (latest)",
      latest.format(sel="doc_type, count(*) filter (where coalesce(header_fields,0)=0) zh, count(*) filter (where coalesce(line_items,0)=0) zl, count(*) tot")+" group by 1 order by 4 desc")
    q(cur,"DISTINCT docs total","select count(*) from ("+latest.format(sel="1 as x")+") z")
    # non-determinism: distinct docs with confidence spread
    q(cur,"Non-determinism: docs w/ >0.05 confidence spread across runs",
      "select count(*) from (select doc_type,doc_pk from proc.bp_extraction_telemetry where doc_pk is not null group by 1,2 having count(*)>1 and (max(confidence)-min(confidence))>5) z")
    # health cron staleness
    q(cur,"Health cron: days since last record","select now()::date - max(recorded_at)::date as days_stale, max(recorded_at) from proc.bp_extraction_health_metrics")
    q(cur,"Telemetry: days since last capture","select now()::date - max(captured_at)::date as days_stale, max(captured_at) from proc.bp_extraction_telemetry")
    q(cur,"Provenance v3: days since last extract","select now()::date - max(extracted_at)::date as days_stale, max(extracted_at) from proc.bp_extraction_provenance_v3")
    q(cur,"PM: days since last upload","select now()::date - max(created_date)::date as days_stale, max(created_date) from proc.process_monitor")
    # raw->stg->trgt funnel per doc type
    for dt,raw,stg,trg in [("quote","bp_quote_raw","bp_quote_stg","bp_quote_trgt"),
                            ("invoice","bp_invoice_raw","bp_invoice_stg","bp_invoice_trgt"),
                            ("po","bp_purchase_order_raw","bp_purchase_order_stg","bp_purchase_order_trgt")]:
        try:
            cur.execute(f"select (select count(*) from proc.{raw}),(select count(*) from proc.{stg}),(select count(*) from proc.{trg})")
            print(f"\n### FUNNEL {dt}: raw/stg/trgt =", cur.fetchone())
        except Exception as e:
            print(f"\n### FUNNEL {dt} ERR:", e)
    # Deal_Linked but failed contradiction detail
    q(cur,"PM Deal_Linked+failed rows","select id, document_type, deal_name, action_status, doc_action, content_hash is null hnull from proc.process_monitor where status='Deal_Linked' and action_status='failed' order by id")
    # doc_type label inconsistency
    q(cur,"Telemetry doc_type labels","select doc_type, count(*) from proc.bp_extraction_telemetry group by 1 order by 2 desc")
    # provenance: hallucinated fields still present in trgt? cross-check final_confidence of hallucinated
    q(cur,"Hallucination: are they high-confidence? join prov","select h.doc_type,h.field_path,round(p.final_confidence::numeric,2) fconf, p.model from proc.bp_extraction_hallucination_audit h join proc.bp_extraction_provenance_v3 p on p.provenance_id=h.provenance_id order by 3 desc limit 25")
