import json

from src.services.structural_extractor.types import ExtractionResult


def park_in_review_queue(conn, result: ExtractionResult, file_path: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO proc.extraction_review_queue "
            "(process_monitor_id, file_path, doc_type, partial_header, partial_line_items, "
            " failed_fields, parsed_text, attempt_count, last_attempt_at, signals_json) "
            "VALUES (%s, %s, %s, %s::jsonb, %s::jsonb, %s, %s, %s, NOW(), %s::jsonb)",
            (
                result.process_monitor_id, file_path, result.doc_type,
                json.dumps({k: v.value for k, v in result.header.items()}, default=str),
                json.dumps([{k: ev.value for k, ev in item.items()} for item in result.line_items], default=str),
                result.unresolved_fields, result.parsed_text[:50000],
                result.attempts,
                json.dumps({"layout_signature": result.layout_signature}),
            ),
        )
        if result.process_monitor_id:
            cur.execute(
                "UPDATE proc.process_monitor SET status='Extraction_InReview' WHERE id=%s",
                (result.process_monitor_id,),
            )
        conn.commit()
