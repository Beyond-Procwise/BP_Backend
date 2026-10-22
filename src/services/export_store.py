"""The exports table: a file an agent built for a person, kept until they collect or delete it.

One row per export in ``proc.bp_export`` (deploy/sql/2026-10-22_bp_export.sql). The agent tools save
here and hand back only the export id and the filename; a route serves the bytes to the person the
file was built for. The store is a thin layer: it neither builds nor inspects content. Every call is a
single statement on an autocommit connection (see reference_get_conn_is_autocommit). An unreadable
store is reported as unavailable, never as "no exports".
"""
from __future__ import annotations

import logging
import uuid
from typing import Any, Callable, Dict, List, Optional, Sequence

from src.services.contract_tools import store

log = logging.getLogger(__name__)

KINDS = ("docx", "xlsx", "pptx", "svg")
MEDIA = {
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "svg": "image/svg+xml",
}
EXPORT_MAX_BYTES = 5 * 1024 * 1024
LIST_LIMIT = 50

Rows = Callable[[str, Sequence[Any]], List[tuple]]

_INSERT = ("INSERT INTO proc.bp_export (export_id, kind, filename, media_type, title, content, size_bytes, source, created_by) "
           "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING export_id")
_GET = ("SELECT export_id, kind, filename, media_type, title, content, size_bytes, source, created_by, created_date "
        "FROM proc.bp_export WHERE export_id = %s")
_RECENT = ("SELECT export_id, kind, filename, media_type, title, NULL, size_bytes, source, created_by, created_date "
           "FROM proc.bp_export WHERE created_by = %s ORDER BY created_date DESC LIMIT %s")
_DELETE = "DELETE FROM proc.bp_export WHERE export_id = %s AND created_by = %s RETURNING export_id"

UNAVAILABLE = {"found": False, "available": False,
               "note": "The export store could not be reached, so no file was saved. The content is still in this answer."}


class ExportTooLarge(ValueError):
    pass


def _record(row: tuple, *, with_content: bool) -> Dict[str, Any]:
    export_id, kind, filename, media_type, title, content, size, source, created_by, created = row
    out = {"export_id": export_id, "kind": kind, "filename": filename, "media_type": media_type, "title": title,
           "size_bytes": int(size), "source": source, "created_by": created_by,
           "created_date": created.isoformat() if hasattr(created, "isoformat") else created}
    if with_content:
        out["content"] = bytes(content) if content is not None else b""
    return out


def save(kind: str, *, title: str, content: bytes, created_by: str, source: str = "agent",
         filename: Optional[str] = None, rows: Optional[Rows] = None) -> Dict[str, Any]:
    """Store one file and return its record without the bytes. Raises ExportTooLarge above the cap and
    store.StoreUnavailable when the table cannot be written."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {', '.join(KINDS)}")
    if len(content) > EXPORT_MAX_BYTES:
        raise ExportTooLarge(f"the file is {len(content)} bytes; the cap is {EXPORT_MAX_BYTES}")
    export_id = uuid.uuid4().hex[:16]
    name = filename or f"{_slug(title)}.{kind}"
    read = rows or store.rows
    read(_INSERT, [export_id, kind, name, MEDIA[kind], title, content, len(content), source, created_by])
    return {"export_id": export_id, "kind": kind, "filename": name, "media_type": MEDIA[kind], "title": title,
            "size_bytes": len(content), "source": source, "created_by": created_by}


def get(export_id: str, *, rows: Optional[Rows] = None) -> Optional[Dict[str, Any]]:
    found = (rows or store.rows)(_GET, [export_id])
    return _record(found[0], with_content=True) if found else None


def recent(created_by: str, *, limit: int = LIST_LIMIT, rows: Optional[Rows] = None) -> List[Dict[str, Any]]:
    return [_record(r, with_content=False) for r in (rows or store.rows)(_RECENT, [created_by, max(1, min(int(limit), LIST_LIMIT))])]


def delete(export_id: str, *, created_by: str, rows: Optional[Rows] = None) -> bool:
    return bool((rows or store.rows)(_DELETE, [export_id, created_by]))


def _slug(title: str) -> str:
    keep = "".join(c.lower() if c.isalnum() else "-" for c in (title or "export").strip())
    while "--" in keep:
        keep = keep.replace("--", "-")
    return keep.strip("-")[:60] or "export"
