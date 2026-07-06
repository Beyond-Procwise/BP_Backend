
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from services.db import get_conn

log = logging.getLogger(__name__)

router = APIRouter(prefix="/session", tags=["Session"])


@router.get("/extraction-status")
def get_extraction_status():

    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT COUNT(*)
                    FROM proc.process_monitor
                    WHERE action_status IS NULL
                    AND session_id IS NOT NULL
                    """
                )
                count = cur.fetchone()[0]

        return {"resolved": count == 0}

    except Exception as exc:
        log.exception("Failed to fetch extraction status")
        raise HTTPException(status_code=500, detail=str(exc))
