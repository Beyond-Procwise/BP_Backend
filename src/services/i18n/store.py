"""Where translations are kept: an in-process LRU in front of proc.bp_translation.

For audit, a machine translation is written once and never changed: the text a person was
shown can always be found again under (language, source hash, prompt version, model).
Reviewed (human) rows may be replaced by a later import; the import audits old -> new
(see audit.record_reviewed_import).

The database is a cache, not a dependency. If it is unreachable a lookup is a miss and
a save is logged and dropped; the UI still gets English and the model's fresh output.
get_conn() is AUTOCOMMIT, so each statement stands alone.
"""
from __future__ import annotations

import hashlib
import logging
import threading
import unicodedata
from collections import OrderedDict
from typing import Optional

from src.services.db import get_conn

logger = logging.getLogger(__name__)

REVIEWED_VERSION = "reviewed"
REVIEWED_MODEL = "human"


def source_hash(text: str) -> str:
    return hashlib.sha256(unicodedata.normalize("NFC", text).encode("utf-8")).hexdigest()


class MemoryLayer:
    def __init__(self, max_size: int = 20000):
        self._max = max(1, max_size)
        self._data: OrderedDict = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key) -> Optional[str]:
        with self._lock:
            if key not in self._data:
                return None
            self._data.move_to_end(key)
            return self._data[key]

    def put(self, key, value: str) -> None:
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self._max:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


class InMemoryTranslationStore:
    """The same contract as the database store, for tests and DB-less runs."""

    def __init__(self):
        self._rows: dict[tuple, str] = {}

    def lookup(self, lang, hashes, prompt_version, model) -> dict[str, str]:
        out = {}
        for h in hashes:
            v = self._rows.get((lang, h, REVIEWED_VERSION, REVIEWED_MODEL)) or \
                self._rows.get((lang, h, prompt_version, model))
            if v is not None:
                out[h] = v
        return out

    def __init_public(self):
        if not hasattr(self, "_public"):
            self._public: dict[str, str] = {}

    def save(self, lang, prompt_version, model, rows) -> None:
        for h, (_src, text) in rows.items():
            self._rows.setdefault((lang, h, prompt_version, model), text)

    def import_reviewed(self, lang, rows) -> int:
        for h, (_src, text) in rows.items():
            self._rows[(lang, h, REVIEWED_VERSION, REVIEWED_MODEL)] = text
        return len(rows)

    def reviewed_changes(self, lang, rows) -> list[tuple[str, str, str]]:
        out = []
        for h, (_src, text) in rows.items():
            old = self._rows.get((lang, h, REVIEWED_VERSION, REVIEWED_MODEL))
            if old is not None and old != text:
                out.append((h, old, text))
        return out

    def languages_with(self, hashes, prompt_version, model) -> dict[str, int]:
        wanted, found = set(hashes), {}
        for (lang, h, pv, m) in self._rows:
            if h in wanted and ((pv, m) == (REVIEWED_VERSION, REVIEWED_MODEL) or (pv, m) == (prompt_version, model)):
                found.setdefault(lang, set()).add(h)
        return {lang: len(hs) for lang, hs in found.items()}

    def set_public_keys(self, keys: dict[str, str]) -> None:
        self.__init_public()
        self._public = dict(keys)

    def public_keys(self) -> dict[str, str]:
        self.__init_public()
        return dict(self._public)


class PgTranslationStore:
    def lookup(self, lang, hashes, prompt_version, model) -> dict[str, str]:
        if not hashes:
            return {}
        try:
            with get_conn() as conn, conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT DISTINCT ON (source_hash) source_hash, translated_text
                      FROM proc.bp_translation
                     WHERE target_lang = %s AND source_hash = ANY(%s)
                       AND (origin = 'reviewed' OR (prompt_version = %s AND model = %s))
                     ORDER BY source_hash, (origin = 'reviewed') DESC
                    """,
                    (lang, list(hashes), prompt_version, model),
                )
                return {h: t for h, t in cur.fetchall()}
        except Exception as exc:  # the cache must never take the UI down
            logger.warning("i18n: translation lookup failed (treated as a miss): %s", exc)
            return {}

    def _write(self, lang, prompt_version, model, rows, origin) -> int:
        if not rows:
            return 0
        values = [(lang, h, prompt_version, model, src, text, origin) for h, (src, text) in rows.items()]
        # Machine rows are write-once (audit); a reviewed import replaces, and audits the change.
        on_conflict = ("DO NOTHING" if origin == "machine" else
                       "DO UPDATE SET translated_text = EXCLUDED.translated_text, created_at = now()")
        try:
            with get_conn() as conn, conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO proc.bp_translation
                        (target_lang, source_hash, prompt_version, model, source_text, translated_text, origin)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (target_lang, source_hash, prompt_version, model)
                    """ + on_conflict,
                    values,
                )
            return len(values)
        except Exception as exc:
            logger.warning("i18n: saving %d translations for %s failed: %s", len(values), lang, exc)
            return 0

    def save(self, lang, prompt_version, model, rows) -> None:
        self._write(lang, prompt_version, model, rows, "machine")

    def import_reviewed(self, lang, rows) -> int:
        return self._write(lang, REVIEWED_VERSION, REVIEWED_MODEL, rows, "reviewed")

    def reviewed_changes(self, lang, rows) -> list[tuple[str, str, str]]:
        """Reviewed rows an import would replace with different text: (hash, old, new)."""
        if not rows:
            return []
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT source_hash, translated_text FROM proc.bp_translation "
                "WHERE target_lang = %s AND origin = 'reviewed' AND source_hash = ANY(%s)",
                (lang, list(rows)),
            )
            current = dict(cur.fetchall())
        return [(h, current[h], text) for h, (_src, text) in rows.items()
                if h in current and current[h] != text]

    def languages_with(self, hashes, prompt_version, model) -> dict[str, int]:
        """Languages holding a servable translation of any of `hashes`, with how many."""
        if not hashes:
            return {}
        try:
            with get_conn() as conn, conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT target_lang, count(DISTINCT source_hash)
                      FROM proc.bp_translation
                     WHERE source_hash = ANY(%s)
                       AND (origin = 'reviewed' OR (prompt_version = %s AND model = %s))
                     GROUP BY target_lang
                    """,
                    (list(hashes), prompt_version, model),
                )
                return {lang: int(n) for lang, n in cur.fetchall()}
        except Exception as exc:
            logger.warning("i18n: listing languages failed (treated as none): %s", exc)
            return {}

    def set_public_keys(self, keys: dict[str, str]) -> None:
        """Replace the signed-out key list (one statement set; get_conn is AUTOCOMMIT, so
        the replace runs inside an explicit transaction)."""
        with get_conn() as conn:
            conn.autocommit = False
            try:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM proc.bp_i18n_public_key")
                    cur.executemany(
                        "INSERT INTO proc.bp_i18n_public_key (msg_key, source_text, source_hash) VALUES (%s, %s, %s)",
                        [(k, v, source_hash(v)) for k, v in keys.items()],
                    )
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def public_keys(self) -> dict[str, str]:
        try:
            with get_conn() as conn, conn.cursor() as cur:
                cur.execute("SELECT msg_key, source_text FROM proc.bp_i18n_public_key")
                return dict(cur.fetchall())
        except Exception as exc:
            logger.warning("i18n: reading the public key list failed (serving none): %s", exc)
            return {}
