"""Read + cache of active per-vendor extraction hints.

Active hints live in proc.bp_prompt (prompt_type='extraction_vendor_hint') so they
reuse the existing governance table + versioning. This store mirrors PromptEngine:
it reads the active rows, caches them by (doc_type, vendor_key), and exposes a
hot-reload (refresh) so an approval is live in-process without a deploy.

context_layer consults hints_for() when building the extraction prompt.
"""
from __future__ import annotations

import json
import logging
import threading

from src.services.db import get_conn

log = logging.getLogger(__name__)

HINT_PROMPT_TYPE = "extraction_vendor_hint"


class ExtractionHintStore:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._cache: dict[tuple[str, str], list[str]] = {}
        self._loaded = False

    def refresh(self) -> int:
        """Reload active hints from bp_prompt. Returns the number of hints loaded.

        On DB error the existing cache is preserved (fail-open: extraction keeps
        running with whatever hints were last known, or none).
        """
        cache: dict[tuple[str, str], list[str]] = {}
        try:
            with get_conn() as c, c.cursor() as cur:
                cur.execute(
                    "SELECT prompts_desc FROM proc.bp_prompt "
                    "WHERE prompt_type = %s AND COALESCE(prompts_status, 1) = 1",
                    (HINT_PROMPT_TYPE,),
                )
                rows = cur.fetchall()
        except Exception as exc:  # noqa: BLE001
            log.warning("ExtractionHintStore.refresh failed (keeping cache): %s", exc)
            return sum(len(v) for v in self._cache.values())

        count = 0
        for row in rows:
            desc = row[0]
            if desc is None:
                continue
            data = desc if isinstance(desc, dict) else json.loads(desc)
            scope = data.get("scope") or {}
            doc_type = str(scope.get("doc_type") or "").strip().lower()
            vendor_key = str(scope.get("vendor_key") or "").strip().lower()
            hint = data.get("hint_text")
            if doc_type and vendor_key and hint:
                cache.setdefault((doc_type, vendor_key), []).append(str(hint))
                count += 1

        with self._lock:
            self._cache = cache
            self._loaded = True
        log.info("ExtractionHintStore: loaded %d active hints", count)
        return count

    def hints_for(self, doc_type: str | None, vendor_key: str | None) -> list[str]:
        """Active advisory hints for this (doc_type, vendor). Empty if none."""
        if not self._loaded:
            self.refresh()
        if not doc_type or not vendor_key:
            return []
        key = (str(doc_type).strip().lower(), str(vendor_key).strip().lower())
        with self._lock:
            return list(self._cache.get(key, []))


# Process-wide singleton (mirrors PromptEngine usage).
HINT_STORE = ExtractionHintStore()
