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
import time
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
    """LRU with a time limit: a reviewed import made by another process (the CLI) replaces
    what this process serves within `ttl` seconds, without a restart."""

    def __init__(self, max_size: int = 20000, ttl: float = 600.0, clock=time.monotonic):
        self._max = max(1, max_size)
        self._ttl, self._clock = ttl, clock
        self._data: OrderedDict = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key) -> Optional[str]:
        with self._lock:
            hit = self._data.get(key)
            if hit is None:
                return None
            value, expires = hit
            if expires <= self._clock():
                del self._data[key]
                return None
            self._data.move_to_end(key)
            return value

    def put(self, key, value: str) -> None:
        with self._lock:
            self._data[key] = (value, self._clock() + self._ttl)
            self._data.move_to_end(key)
            while len(self._data) > self._max:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


_CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}


def _merge_status(old: Optional[dict], recognized: Optional[bool], confidence: Optional[str]) -> dict:
    """Sticky 'not recognised'; the lowest confidence seen."""
    old = old or {"recognized": None, "confidence": None}
    if old["recognized"] is False or recognized is False:
        rec = False
    else:
        rec = recognized if recognized is not None else old["recognized"]
    confs = [c for c in (old["confidence"], confidence) if c in _CONFIDENCE_RANK]
    return {"recognized": rec, "confidence": min(confs, key=_CONFIDENCE_RANK.get) if confs else None}


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

    def record_language_status(self, lang, prompt_version, model, recognized, confidence) -> None:
        if recognized is None and confidence is None:
            return
        if not hasattr(self, "_status"):
            self._status: dict[tuple, dict] = {}
        key = (lang, prompt_version, model)
        self._status[key] = _merge_status(self._status.get(key), recognized, confidence)

    def language_status(self, lang, prompt_version, model) -> Optional[dict]:
        return dict(getattr(self, "_status", {}).get((lang, prompt_version, model)) or {}) or None

    def language_statuses(self, prompt_version, model) -> dict[str, dict]:
        return {k[0]: dict(v) for k, v in getattr(self, "_status", {}).items()
                if k[1] == prompt_version and k[2] == model}

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

    def record_language_status(self, lang, prompt_version, model, recognized, confidence) -> None:
        """Merge one reply's flags into the language's verdict (see _merge_status)."""
        if recognized is None and confidence is None:
            return
        try:
            with get_conn() as conn, conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO proc.bp_translation_language_status
                        (target_lang, prompt_version, model, recognized, confidence)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (target_lang, prompt_version, model) DO UPDATE SET
                        recognized = CASE
                            WHEN bp_translation_language_status.recognized IS FALSE
                              OR EXCLUDED.recognized IS FALSE THEN FALSE
                            ELSE COALESCE(EXCLUDED.recognized, bp_translation_language_status.recognized) END,
                        confidence = CASE
                            WHEN EXCLUDED.confidence IS NULL THEN bp_translation_language_status.confidence
                            WHEN bp_translation_language_status.confidence IS NULL THEN EXCLUDED.confidence
                            WHEN array_position(ARRAY['low','medium','high'], EXCLUDED.confidence)
                               < array_position(ARRAY['low','medium','high'], bp_translation_language_status.confidence)
                              THEN EXCLUDED.confidence
                            ELSE bp_translation_language_status.confidence END,
                        updated_at = now()
                    """,
                    (lang, prompt_version, model, recognized, confidence),
                )
        except Exception as exc:
            logger.warning("i18n: recording the language verdict for %s failed: %s", lang, exc)

    def language_status(self, lang, prompt_version, model) -> Optional[dict]:
        return self.language_statuses(prompt_version, model, lang=lang).get(lang)

    def language_statuses(self, prompt_version, model, *, lang: Optional[str] = None) -> dict[str, dict]:
        try:
            with get_conn() as conn, conn.cursor() as cur:
                cur.execute(
                    "SELECT target_lang, recognized, confidence FROM proc.bp_translation_language_status "
                    "WHERE prompt_version = %s AND model = %s" + (" AND target_lang = %s" if lang else ""),
                    (prompt_version, model, lang) if lang else (prompt_version, model),
                )
                return {t: {"recognized": r, "confidence": c} for t, r, c in cur.fetchall()}
        except Exception as exc:
            logger.warning("i18n: reading language verdicts failed (treated as none): %s", exc)
            return {}
