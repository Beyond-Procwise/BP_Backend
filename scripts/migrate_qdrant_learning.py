"""One-off: copy the Qdrant 'learning' collection's records into Postgres source rows (proc.bp_vector_source, collection 'learning').

WHY THIS EXISTS. Every other vector can be rebuilt from a Postgres source (documents from their _raw rows, policy from the S3 files). Learning
records and model plans were only ever stored in Qdrant, so the vector store itself held the only copy. Removing Qdrant without this loses them.
It copies the PAYLOADS (not the vectors, which are re-derived by the worker from the text), idempotently by point id.

    python scripts/migrate_qdrant_learning.py              # reads QDRANT_URL / QDRANT_API_KEY, writes via the app's database settings
    then let the vector worker embed them (the scheduler does, or: python -m src.services.vector_store.worker)

Uses Qdrant's REST API directly (no qdrant_client), so it keeps working after the dependency is removed. Delete it once every environment is migrated.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
from typing import Any, Dict, Iterator, Optional, Tuple

COLLECTION = os.environ.get("LEARNING_COLLECTION", "learning")
TENANT = "default"


def scroll_learning(url: str, api_key: Optional[str]) -> Iterator[Dict[str, Any]]:
    offset = None
    while True:
        body: Dict[str, Any] = {"limit": 100, "with_payload": True, "with_vector": False}
        if offset is not None:
            body["offset"] = offset
        headers = {"content-type": "application/json", **({"api-key": api_key} if api_key else {})}
        req = urllib.request.Request(f"{url.rstrip('/')}/collections/{COLLECTION}/points/scroll", data=json.dumps(body).encode(), headers=headers)
        result = json.load(urllib.request.urlopen(req, timeout=60))["result"]
        yield from result.get("points", [])
        offset = result.get("next_page_offset")
        if offset is None:
            return


def record_to_source(point: Dict[str, Any]) -> Tuple[str, str, Dict[str, Any]]:
    """(source_id, text, metadata): the text is what the vector was computed from (summary / plan_text); the metadata is the whole payload."""
    payload = dict(point.get("payload") or {})
    text = payload.get("summary") if payload.get("document_type") != "model_plan" else payload.get("plan_text")
    text = text or payload.get("summary") or payload.get("plan_text")
    if not isinstance(text, str) or not text.strip():
        text = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)     # nothing is lost for a record with no text
    return str(point["id"]), text, payload


def put_source(tenant: str, collection: str, source_id: str, content: str, metadata: Dict[str, Any]) -> None:
    from src.services.vector_store import sources
    sources.put_source(tenant, collection, source_id, content, metadata)


def export(url: str, api_key: Optional[str]) -> Dict[str, int]:
    report = {"read": 0, "written": 0, "failed": 0}
    for point in scroll_learning(url, api_key):
        report["read"] += 1
        try:
            source_id, content, meta = record_to_source(point)
            put_source(TENANT, "learning", source_id, content, meta)
            report["written"] += 1
        except Exception as exc:  # noqa: BLE001 - one bad record must not stop the export
            report["failed"] += 1
            print(f"failed {point.get('id')}: {type(exc).__name__}: {exc}", file=sys.stderr)
    return report


def main() -> int:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    url = os.environ.get("QDRANT_URL")
    if not url:
        print("QDRANT_URL is not set", file=sys.stderr)
        return 2
    report = export(url, (os.environ.get("QDRANT_API_KEY") or "").strip() or None)
    print(report)
    return 1 if report["failed"] else 0


if __name__ == "__main__":  # pragma: no cover
    # An operator tool acts for the platform across every company; with row-level security on, a tool
    # that names no company would otherwise see no rows (D-47).
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    from src.services.tenant_context import enter_system as _enter_system
    _enter_system()
    raise SystemExit(main())
