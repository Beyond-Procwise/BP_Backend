import json
import logging
from typing import Callable, Optional

log = logging.getLogger(__name__)


def get_trust_level(success_count: int) -> str:
    if success_count == 0:
        return "none"
    if success_count < 3:
        return "learning"
    return "trusted"


class PatternStore:
    def __init__(self, get_conn: Callable):
        self._get_conn = get_conn

    def get_pattern_anchors(self, file_type: str, doc_type: str, supplier_name: str,
                            layout_signature: str) -> Optional[dict]:
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT anchor_patterns FROM proc.bp_extraction_patterns "
                    "WHERE file_type=%s AND doc_type=%s AND supplier_name=%s AND layout_signature=%s",
                    (file_type, doc_type, supplier_name, layout_signature),
                )
                row = cur.fetchone()
                if row and row[0]:
                    return row[0] if isinstance(row[0], dict) else json.loads(row[0])
        except Exception:
            log.debug("get_pattern_anchors failed", exc_info=True)
        return None

    def save_pattern_anchors(self, file_type: str, doc_type: str, supplier_name: str,
                             layout_signature: str, anchors: dict) -> None:
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT pattern_id, success_count FROM proc.bp_extraction_patterns "
                    "WHERE file_type=%s AND doc_type=%s AND supplier_name=%s AND layout_signature=%s",
                    (file_type, doc_type, supplier_name, layout_signature),
                )
                row = cur.fetchone()
                if row:
                    pid, sc = row
                    cur.execute(
                        "UPDATE proc.bp_extraction_patterns "
                        "SET anchor_patterns=%s::jsonb, success_count=%s, last_used=NOW() "
                        "WHERE pattern_id=%s",
                        (json.dumps(anchors), sc + 1, pid),
                    )
                else:
                    cur.execute(
                        "INSERT INTO proc.bp_extraction_patterns "
                        "(file_type, doc_type, supplier_name, layout_signature, anchor_patterns, success_count) "
                        "VALUES (%s, %s, %s, %s, %s::jsonb, 1)",
                        (file_type, doc_type, supplier_name, layout_signature, json.dumps(anchors)),
                    )
                conn.commit()
        except Exception:
            log.debug("save_pattern_anchors failed", exc_info=True)

    def get_trust(self, file_type: str, doc_type: str, supplier_name: str,
                  layout_signature: str) -> str:
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT success_count FROM proc.bp_extraction_patterns "
                    "WHERE file_type=%s AND doc_type=%s AND supplier_name=%s AND layout_signature=%s",
                    (file_type, doc_type, supplier_name, layout_signature),
                )
                row = cur.fetchone()
                if row:
                    return get_trust_level(int(row[0] or 0))
        except Exception:
            log.debug("get_trust failed", exc_info=True)
        return "none"
