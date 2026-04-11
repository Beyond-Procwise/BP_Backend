"""PatternService — pattern-based intelligence store for procurement.

Patterns are learned insights distilled from procurement outcomes, e.g.
"Cooperative strategy yields 10% discount with repeat suppliers".  They are
NOT raw data rows — they are generalised, reusable signals that agents can
query to guide decisions.

Supports two storage backends:
  storage="postgres"  — persists to proc.procurement_patterns via psycopg2
  storage="memory"    — in-process dict store used in unit tests
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# In-memory store (used for storage="memory")
# ---------------------------------------------------------------------------

class _MemoryStore:
    """Thread-unsafe in-process store for unit tests."""

    def __init__(self) -> None:
        # Keyed by (pattern_type, pattern_text)
        self._rows: Dict[tuple, Dict[str, Any]] = {}
        self._seq: int = 0

    def _next_id(self) -> int:
        self._seq += 1
        return self._seq

    def upsert(self, pattern_type: str, pattern_text: str, category: str,
               confidence: float) -> Dict[str, Any]:
        key = (pattern_type, pattern_text)
        if key in self._rows:
            row = self._rows[key]
            row["source_count"] += 1
            row["confidence"] = min(1.0, round(row["confidence"] + 0.05, 3))
            row["last_validated"] = datetime.now(timezone.utc)
        else:
            row = {
                "id": self._next_id(),
                "pattern_type": pattern_type,
                "pattern_text": pattern_text,
                "category": category,
                "confidence": round(min(1.0, max(0.0, confidence)), 3),
                "source_count": 1,
                "last_validated": datetime.now(timezone.utc),
                "deprecated": False,
            }
            self._rows[key] = row
        return dict(row)

    def get(self, pattern_type: Optional[str], category: Optional[str],
            min_confidence: float) -> List[Dict[str, Any]]:
        results = []
        for row in self._rows.values():
            if row["deprecated"]:
                continue
            if pattern_type and row["pattern_type"] != pattern_type:
                continue
            if category and row["category"] != category:
                continue
            if row["confidence"] < min_confidence:
                continue
            results.append(dict(row))
        results.sort(key=lambda r: r["confidence"], reverse=True)
        return results

    def reinforce(self, pattern_type: str, pattern_text: str,
                  delta: float) -> Optional[Dict[str, Any]]:
        key = (pattern_type, pattern_text)
        row = self._rows.get(key)
        if row is None:
            return None
        row["confidence"] = round(min(1.0, max(0.0, row["confidence"] + delta)), 3)
        row["source_count"] += 1
        row["last_validated"] = datetime.now(timezone.utc)
        return dict(row)

    def deprecate(self, pattern_type: str, pattern_text: str) -> bool:
        key = (pattern_type, pattern_text)
        row = self._rows.get(key)
        if row is None:
            return False
        row["deprecated"] = True
        return True


# ---------------------------------------------------------------------------
# PatternService
# ---------------------------------------------------------------------------

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS proc.procurement_patterns (
    id SERIAL PRIMARY KEY,
    pattern_type VARCHAR(50) NOT NULL,
    pattern_text TEXT NOT NULL,
    category VARCHAR(100) DEFAULT '',
    confidence NUMERIC(4,3) DEFAULT 0.5,
    source_count INTEGER DEFAULT 1,
    last_validated TIMESTAMP DEFAULT NOW(),
    deprecated BOOLEAN DEFAULT FALSE,
    UNIQUE (pattern_type, pattern_text)
);
"""


class PatternService:
    """Procurement pattern intelligence store.

    Parameters
    ----------
    storage:
        ``"postgres"`` (default) — use the project's psycopg2 connection
        helper from :mod:`services.db`.
        ``"memory"`` — use an in-process dict store (suitable for tests).
    """

    def __init__(self, storage: str = "postgres") -> None:
        self._storage = storage
        if storage == "memory":
            self._mem = _MemoryStore()
        else:
            self._mem = None  # type: ignore[assignment]
            self.ensure_table()

    # ------------------------------------------------------------------
    # Table bootstrap
    # ------------------------------------------------------------------

    def ensure_table(self) -> None:
        """Create the proc.procurement_patterns table if it does not exist."""
        if self._storage == "memory":
            return
        try:
            from services.db import get_conn  # local import keeps tests fast
            import psycopg2  # noqa: F401 — confirm it is available
        except ImportError as exc:
            logger.warning("PatternService.ensure_table skipped: %s", exc)
            return

        try:
            from services.db import get_conn
            with get_conn() as conn:
                cur = conn.cursor()
                cur.execute(_CREATE_TABLE_SQL)
                logger.debug("proc.procurement_patterns table ensured")
        except Exception:
            logger.exception("PatternService.ensure_table failed")

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def record_pattern(
        self,
        pattern_type: str,
        pattern_text: str,
        category: str = "",
        confidence: float = 0.5,
    ) -> Dict[str, Any]:
        """Insert or update a pattern (UPSERT).

        If the pattern already exists the ``source_count`` is incremented and
        ``confidence`` is nudged upward (+0.05, capped at 1.0).

        Returns the current state of the pattern row.
        """
        pattern_type = pattern_type.strip()
        pattern_text = pattern_text.strip()
        if not pattern_type or not pattern_text:
            raise ValueError("pattern_type and pattern_text must be non-empty strings")

        confidence = float(min(1.0, max(0.0, confidence)))

        if self._storage == "memory":
            return self._mem.upsert(pattern_type, pattern_text, category, confidence)

        return self._pg_upsert(pattern_type, pattern_text, category, confidence)

    def get_patterns(
        self,
        pattern_type: Optional[str] = None,
        category: Optional[str] = None,
        min_confidence: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Return non-deprecated patterns sorted by confidence descending.

        Parameters
        ----------
        pattern_type:
            Filter by pattern type when provided.
        category:
            Filter by category when provided.
        min_confidence:
            Only include patterns with ``confidence >= min_confidence``.
        """
        if self._storage == "memory":
            return self._mem.get(pattern_type, category, min_confidence)

        return self._pg_get(pattern_type, category, min_confidence)

    def reinforce_pattern(
        self,
        pattern_type: str,
        pattern_text: str,
        delta: float = 0.05,
    ) -> Optional[Dict[str, Any]]:
        """Increase the confidence of an existing pattern by *delta*.

        ``confidence`` is clamped to [0.0, 1.0].  Returns the updated row or
        ``None`` if no matching pattern exists.
        """
        if self._storage == "memory":
            return self._mem.reinforce(pattern_type, pattern_text, delta)

        return self._pg_reinforce(pattern_type, pattern_text, delta)

    def deprecate_pattern(
        self,
        pattern_type: str,
        pattern_text: str,
    ) -> bool:
        """Mark a pattern as deprecated so it no longer appears in queries.

        Returns ``True`` if the pattern was found and deprecated, ``False``
        if no matching pattern exists.
        """
        if self._storage == "memory":
            return self._mem.deprecate(pattern_type, pattern_text)

        return self._pg_deprecate(pattern_type, pattern_text)

    def learn_from_outcome(
        self,
        outcome: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Compare expected vs actual outcome and record new patterns.

        ``outcome`` should contain:
            - ``strategy``   (str)  — strategy that was applied
            - ``expected``   (dict) — expected metrics (e.g. ``{"discount": 0.10}``)
            - ``actual``     (dict) — actual measured metrics
            - ``category``   (str, optional) — procurement category
            - ``supplier_id`` (str, optional)

        Returns a list of newly recorded/updated pattern dicts.
        """
        strategy = str(outcome.get("strategy") or "unknown").strip()
        expected: Dict[str, Any] = outcome.get("expected") or {}
        actual: Dict[str, Any] = outcome.get("actual") or {}
        category = str(outcome.get("category") or "").strip()

        recorded: List[Dict[str, Any]] = []

        for metric, exp_val in expected.items():
            act_val = actual.get(metric)
            if act_val is None:
                continue

            try:
                exp_f = float(exp_val)
                act_f = float(act_val)
            except (TypeError, ValueError):
                continue

            if abs(exp_f) < 1e-9:
                continue  # avoid division by zero

            ratio = act_f / exp_f  # > 1 means over-performed

            if ratio >= 1.0:
                # Outcome met or exceeded expectation — positive pattern
                pattern_text = (
                    f"{strategy} strategy achieves {metric} "
                    f"of {act_f:.2f} (expected {exp_f:.2f})"
                )
                confidence = round(min(1.0, 0.5 + (ratio - 1.0) * 0.25), 3)
                row = self.record_pattern(
                    pattern_type="outcome_positive",
                    pattern_text=pattern_text,
                    category=category,
                    confidence=confidence,
                )
            else:
                # Under-performed — record a caution pattern
                shortfall = round((1.0 - ratio) * 100, 1)
                pattern_text = (
                    f"{strategy} strategy under-delivers on {metric} "
                    f"by {shortfall}% (expected {exp_f:.2f}, got {act_f:.2f})"
                )
                confidence = round(max(0.0, 0.5 - (1.0 - ratio) * 0.25), 3)
                row = self.record_pattern(
                    pattern_type="outcome_caution",
                    pattern_text=pattern_text,
                    category=category,
                    confidence=confidence,
                )

            recorded.append(row)

        return recorded

    # ------------------------------------------------------------------
    # PostgreSQL back-end helpers
    # ------------------------------------------------------------------

    def _pg_upsert(
        self,
        pattern_type: str,
        pattern_text: str,
        category: str,
        confidence: float,
    ) -> Dict[str, Any]:
        sql = """
            INSERT INTO proc.procurement_patterns
                (pattern_type, pattern_text, category, confidence, source_count, last_validated, deprecated)
            VALUES (%s, %s, %s, %s, 1, NOW(), FALSE)
            ON CONFLICT (pattern_type, pattern_text) DO UPDATE
                SET source_count    = proc.procurement_patterns.source_count + 1,
                    confidence      = LEAST(1.0,
                                           proc.procurement_patterns.confidence + 0.05),
                    last_validated  = NOW()
            RETURNING id, pattern_type, pattern_text, category,
                      confidence, source_count, last_validated, deprecated
        """
        from services.db import get_conn
        with get_conn() as conn:
            cur = conn.cursor()
            cur.execute(sql, (pattern_type, pattern_text, category, confidence))
            row = cur.fetchone()
            if row is None:
                raise RuntimeError("UPSERT returned no row")
            return self._row_to_dict(cur, row)

    def _pg_get(
        self,
        pattern_type: Optional[str],
        category: Optional[str],
        min_confidence: float,
    ) -> List[Dict[str, Any]]:
        conditions = ["deprecated = FALSE", "confidence >= %s"]
        params: list = [min_confidence]

        if pattern_type:
            conditions.append("pattern_type = %s")
            params.append(pattern_type)
        if category:
            conditions.append("category = %s")
            params.append(category)

        where = " AND ".join(conditions)
        sql = f"""
            SELECT id, pattern_type, pattern_text, category,
                   confidence, source_count, last_validated, deprecated
            FROM proc.procurement_patterns
            WHERE {where}
            ORDER BY confidence DESC
        """
        from services.db import get_conn
        with get_conn() as conn:
            cur = conn.cursor()
            cur.execute(sql, params)
            rows = cur.fetchall()
            return [self._row_to_dict(cur, r) for r in rows]

    def _pg_reinforce(
        self,
        pattern_type: str,
        pattern_text: str,
        delta: float,
    ) -> Optional[Dict[str, Any]]:
        sql = """
            UPDATE proc.procurement_patterns
            SET confidence     = LEAST(1.0, GREATEST(0.0, confidence + %s)),
                source_count   = source_count + 1,
                last_validated = NOW()
            WHERE pattern_type = %s AND pattern_text = %s
            RETURNING id, pattern_type, pattern_text, category,
                      confidence, source_count, last_validated, deprecated
        """
        from services.db import get_conn
        with get_conn() as conn:
            cur = conn.cursor()
            cur.execute(sql, (delta, pattern_type, pattern_text))
            row = cur.fetchone()
            if row is None:
                return None
            return self._row_to_dict(cur, row)

    def _pg_deprecate(self, pattern_type: str, pattern_text: str) -> bool:
        sql = """
            UPDATE proc.procurement_patterns
            SET deprecated = TRUE
            WHERE pattern_type = %s AND pattern_text = %s
            RETURNING id
        """
        from services.db import get_conn
        with get_conn() as conn:
            cur = conn.cursor()
            cur.execute(sql, (pattern_type, pattern_text))
            row = cur.fetchone()
            return row is not None

    @staticmethod
    def _row_to_dict(cur: Any, row: tuple) -> Dict[str, Any]:
        columns = [desc.name for desc in cur.description]
        return dict(zip(columns, row))
