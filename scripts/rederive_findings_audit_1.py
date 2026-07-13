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
        for r in rows[:40]:
            print("  ", r)
        if len(rows) > 40: print(f"  ...(+{len(rows)-40} more)")
    except Exception as e:
        print("  ERR:", e)

with get_conn() as c:
    cur = c.cursor()
    # ---- process_monitor integrity ----
    q(cur,"PM status x action_status","select status, action_status, doc_action, count(*) from proc.process_monitor group by 1,2,3 order by 4 desc")
    q(cur,"PM content_hash NULL (dedup/F28)","select (content_hash is null) as hash_null, count(*) from proc.process_monitor group by 1")
    q(cur,"PM latency start->end minutes by type","select type, count(*), round(avg(extract(epoch from (end_ts-start_ts))/60.0)::numeric,1) avg_min, round(max(extract(epoch from (end_ts-start_ts))/60.0)::numeric,1) max_min from proc.process_monitor where end_ts is not null group by 1 order by 3 desc nulls last")
    q(cur,"PM rows still running (end_ts null)","select status, action_status, count(*) from proc.process_monitor where end_ts is null group by 1,2")
    # ---- session outcomes / orphans ----
    q(cur,"SDO outcome dist","select outcome, count(*) from proc.session_document_outcome group by 1 order by 2 desc")
    q(cur,"SDO orphans (session not in PM)","select count(*) from proc.session_document_outcome s where not exists (select 1 from proc.process_monitor p where p.session_id=s.session_id)")
    q(cur,"SDO dup (same session+file multiple outcomes)","select session_id,file_path,count(*) n, array_agg(distinct outcome) from proc.session_document_outcome group by 1,2 having count(*)>1 order by 3 desc")
    # ---- telemetry: extraction quality ----
    q(cur,"TEL status dist","select status, completeness_status, count(*) from proc.bp_extraction_telemetry group by 1,2 order by 3 desc")
    q(cur,"TEL confidence buckets","select width_bucket(coalesce(confidence,-1),0,1,10) b, count(*), round(min(confidence)::numeric,3), round(max(confidence)::numeric,3) from proc.bp_extraction_telemetry group by 1 order by 1")
    q(cur,"TEL zero header/line","select doc_type, count(*) filter (where coalesce(header_fields,0)=0) zero_hdr, count(*) filter (where coalesce(line_items,0)=0) zero_lines, count(*) tot from proc.bp_extraction_telemetry group by 1")
    q(cur,"TEL missing_required top","select missing_required, count(*) from proc.bp_extraction_telemetry where missing_required is not null and missing_required<>'' group by 1 order by 2 desc limit 15")
    q(cur,"TEL errors","select error_detail, count(*) from proc.bp_extraction_telemetry where error_detail is not null and error_detail<>'' group by 1 order by 2 desc limit 15")
    q(cur,"TEL parser backends","select parser_backend, count(*) from proc.bp_extraction_telemetry group by 1 order by 2 desc")
    q(cur,"TEL pipeline versions","select pipeline_version, count(*) from proc.bp_extraction_telemetry group by 1 order by 2 desc")
    # non-determinism: same doc_pk+doc_type multiple telemetry rows with differing confidence
    q(cur,"TEL non-determinism (same doc varying confidence)","select doc_type, doc_pk, count(*) n, round(min(confidence)::numeric,3) lo, round(max(confidence)::numeric,3) hi, round((max(confidence)-min(confidence))::numeric,3) spread from proc.bp_extraction_telemetry where doc_pk is not null group by 1,2 having count(*)>1 and (max(confidence)-min(confidence))>0.05 order by spread desc limit 20")
    # ---- discrepancy ----
    q(cur,"DISC issue x severity x status","select issue_type, severity, status, count(*) from proc.bp_extraction_discrepancy group by 1,2,3 order by 4 desc limit 30")
    q(cur,"DISC blocks_promotion open","select blocks_promotion, status, count(*) from proc.bp_extraction_discrepancy group by 1,2 order by 3 desc")
    q(cur,"DISC top fields","select field_name, count(*) from proc.bp_extraction_discrepancy group by 1 order by 2 desc limit 15")
    # ---- hallucination ----
    q(cur,"HALLUC by field/reason","select doc_type, field_path, reason, count(*) from proc.bp_extraction_hallucination_audit group by 1,2,3 order by 4 desc limit 25")
    # ---- provenance confidence ----
    q(cur,"PROV low final_confidence share","select doc_type, count(*) tot, count(*) filter (where final_confidence<0.5) lo50, count(*) filter (where final_confidence is null) nullc from proc.bp_extraction_provenance_v3 group by 1 order by 2 desc")
    q(cur,"PROV models used","select model, count(*) from proc.bp_extraction_provenance_v3 group by 1 order by 2 desc limit 10")
    # ---- health metrics latest ----
    q(cur,"HEALTH latest 3","select recorded_at, stuck_rows_reset, stuck_rows_failed, audit_sample, audit_violations, failed_reaped, invoice_active_hitl, po_active_hitl, quote_active_hitl from proc.bp_extraction_health_metrics order by recorded_at desc limit 3")
    q(cur,"HEALTH total audit violations","select sum(audit_violations) viol, sum(stuck_rows_failed) stuck_failed, sum(failed_reaped) reaped, max(recorded_at) latest, min(recorded_at) earliest from proc.bp_extraction_health_metrics")
