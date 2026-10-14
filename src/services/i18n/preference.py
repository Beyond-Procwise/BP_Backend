"""A signed-in user's chosen language, so it follows them to another device."""
from __future__ import annotations

import json
from typing import Optional

from src.services.db import get_conn


def get_language(subject: str) -> Optional[dict]:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT pref_value FROM proc.bp_user_preference WHERE user_subject = %s AND pref_key = 'language'",
                    (subject,))
        row = cur.fetchone()
    if not row:
        return None
    return row[0] if isinstance(row[0], dict) else json.loads(row[0])


def set_language(subject: str, value: dict) -> None:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO proc.bp_user_preference (user_subject, pref_key, pref_value)
            VALUES (%s, 'language', %s::jsonb)
            ON CONFLICT (user_subject, pref_key)
            DO UPDATE SET pref_value = EXCLUDED.pref_value, updated_at = now()
            """,
            (subject, json.dumps(value, ensure_ascii=False)),
        )
