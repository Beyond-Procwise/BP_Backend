"""Product catalog service — fuzzy-matched product master data.

Maintains a master product catalog in ``proc.bp_products``.  When a line
item is extracted from a PO, Quote, or Invoice the caller passes the
description (and optional SKU / item_id) through
:meth:`ProductCatalogService.match_or_create` which either links it to an
existing product or creates a new one.

Matching strategy
-----------------
1. If the source document supplies an explicit ``item_id`` / SKU that value
   is used directly as the ``product_id`` (exact look-up, insert if new).
2. Otherwise, fuzzy match the ``item_description`` against all known
   products using ``difflib.SequenceMatcher`` (threshold **0.85**).
3. If no match is found a sequential ID ``PROD-NNNNN`` is generated.

Thread safety is handled via ``INSERT … ON CONFLICT … DO UPDATE`` so
concurrent workers never produce duplicate rows.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS proc.bp_products (
    product_id          TEXT PRIMARY KEY,
    item_description    TEXT NOT NULL,
    current_unit_price  NUMERIC,
    currency            TEXT,
    unit_of_measure     TEXT,
    first_seen_date     TIMESTAMP DEFAULT NOW(),
    last_seen_date      TIMESTAMP DEFAULT NOW(),
    source_doc_type     TEXT,
    source_doc_id       TEXT,
    occurrence_count    INTEGER DEFAULT 1,
    created_date        TIMESTAMP DEFAULT NOW(),
    last_modified_date  TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_bp_products_description
    ON proc.bp_products USING btree (lower(item_description));
"""

_UPSERT_SQL = """
INSERT INTO proc.bp_products (
    product_id, item_description, current_unit_price, currency,
    unit_of_measure, first_seen_date, last_seen_date,
    source_doc_type, source_doc_id, occurrence_count,
    created_date, last_modified_date
) VALUES (
    %s, %s, %s, %s,
    %s, %s, %s,
    %s, %s, 1,
    %s, %s
)
ON CONFLICT (product_id) DO UPDATE SET
    item_description   = CASE
                            WHEN length(%s) > length(proc.bp_products.item_description)
                            THEN %s
                            ELSE proc.bp_products.item_description
                         END,
    current_unit_price = COALESCE(EXCLUDED.current_unit_price, proc.bp_products.current_unit_price),
    currency           = COALESCE(EXCLUDED.currency, proc.bp_products.currency),
    unit_of_measure    = COALESCE(EXCLUDED.unit_of_measure, proc.bp_products.unit_of_measure),
    last_seen_date     = EXCLUDED.last_seen_date,
    source_doc_type    = EXCLUDED.source_doc_type,
    source_doc_id      = EXCLUDED.source_doc_id,
    occurrence_count   = proc.bp_products.occurrence_count + 1,
    last_modified_date = EXCLUDED.last_modified_date;
"""

_FUZZY_THRESHOLD = 0.85


class ProductCatalogService:
    """Master product catalog with fuzzy description matching.

    Parameters
    ----------
    get_db_connection:
        A callable that returns a ``psycopg2`` connection object.  The
        service calls it each time it needs a connection so connection
        pooling or refresh logic stays with the caller.
    """

    def __init__(self, get_db_connection: Callable[[], Any]) -> None:
        self._get_conn = get_db_connection
        self._lock = threading.Lock()

        # Cache: product_id -> (lower(item_description), item_description)
        self._cache: Dict[str, Tuple[str, str]] = {}
        self._max_seq: int = 0  # tracks the highest PROD-NNNNN number seen

        self._ensure_table()
        self._load_cache()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def match_or_create(
        self,
        item_description: str,
        item_id_from_doc: Optional[str] = None,
        unit_price: Optional[float] = None,
        currency: Optional[str] = None,
        unit_of_measure: Optional[str] = None,
        doc_type: Optional[str] = None,
        doc_id: Optional[str] = None,
    ) -> str:
        """Return a ``product_id`` for the given item, creating one if needed.

        Matching priority:
        1. Explicit ``item_id_from_doc`` (SKU from the document) — used as-is.
        2. Fuzzy description match (≥ 0.85 similarity ratio).
        3. New sequential ``PROD-NNNNN`` ID.

        The product row is upserted so price / occurrence metadata stays
        current regardless of whether the product already existed.
        """
        if not item_description or not item_description.strip():
            logger.warning("match_or_create called with empty description")
            item_description = "UNKNOWN ITEM"

        item_description = item_description.strip()
        now = datetime.now(timezone.utc)

        # --- 1. Explicit ID from document ---
        if item_id_from_doc and item_id_from_doc.strip():
            product_id = item_id_from_doc.strip()
            self._upsert(product_id, item_description, unit_price, currency,
                         unit_of_measure, doc_type, doc_id, now)
            return product_id

        # --- 2. Fuzzy match against cache ---
        matched_id = self._fuzzy_match(item_description)
        if matched_id:
            self._upsert(matched_id, item_description, unit_price, currency,
                         unit_of_measure, doc_type, doc_id, now)
            return matched_id

        # --- 3. Generate new sequential ID ---
        product_id = self._next_seq_id()
        self._upsert(product_id, item_description, unit_price, currency,
                     unit_of_measure, doc_type, doc_id, now)
        return product_id

    def refresh_cache(self) -> None:
        """Reload the in-memory cache from the database."""
        self._load_cache()

    @property
    def catalog_size(self) -> int:
        """Number of products currently in the cache."""
        return len(self._cache)

    def backfill_item_ids(self) -> int:
        """Populate item_id on existing line items that have NULL item_id.

        Matches item_description against the product catalog and updates
        the item_id column. Returns count of rows updated.
        """
        total_updated = 0
        tables = [
            ("proc.bp_po_line_items", "po_line_id", "po_id"),
            ("proc.bp_quote_line_items", "quote_line_id", "quote_id"),
            ("proc.bp_invoice_line_items", "invoice_line_id", "invoice_id"),
        ]
        for table, pk, fk in tables:
            try:
                conn = self._get_conn()
                conn.autocommit = True
                with conn.cursor() as cur:
                    cur.execute(
                        f"SELECT {pk}, item_description, unit_price, {fk} "
                        f"FROM {table} WHERE item_id IS NULL AND item_description IS NOT NULL"
                    )
                    rows = cur.fetchall()
                    for line_pk, desc, price, doc_id in rows:
                        price_f = float(price) if price else None
                        product_id = self.match_or_create(
                            item_description=desc,
                            item_id_from_doc=None,
                            unit_price=price_f,
                            currency=None,
                            unit_of_measure=None,
                            doc_type=table.split("_")[1] if "_" in table else "",
                            doc_id=str(doc_id or ""),
                        )
                        cur.execute(
                            f"UPDATE {table} SET item_id = %s WHERE {pk} = %s",
                            (product_id, line_pk),
                        )
                        total_updated += 1
                conn.close()
            except Exception:
                logger.warning(
                    "item_id backfill failed for %s", table, exc_info=True
                )

        if total_updated:
            logger.info("Backfilled item_id on %d line items", total_updated)
        return total_updated

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_table(self) -> None:
        """Create the ``proc.bp_products`` table if it does not exist."""
        try:
            conn = self._get_conn()
            with conn.cursor() as cur:
                cur.execute(_CREATE_TABLE_SQL)
            conn.commit()
            logger.info("proc.bp_products table ensured")
        except Exception:
            logger.exception("Failed to create proc.bp_products table")
            raise

    def _load_cache(self) -> None:
        """Populate the in-memory product cache from the database."""
        try:
            conn = self._get_conn()
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT product_id, item_description FROM proc.bp_products"
                )
                rows = cur.fetchall()

            with self._lock:
                self._cache.clear()
                self._max_seq = 0
                for pid, desc in rows:
                    self._cache[pid] = (desc.lower(), desc)
                    self._track_seq(pid)

            logger.info("Product catalog cache loaded: %d products", len(self._cache))
        except Exception:
            logger.exception("Failed to load product catalog cache")
            raise

    def _track_seq(self, product_id: str) -> None:
        """Keep ``_max_seq`` updated for PROD-NNNNN IDs."""
        if product_id.startswith("PROD-"):
            try:
                seq = int(product_id.split("-", 1)[1])
                if seq > self._max_seq:
                    self._max_seq = seq
            except ValueError:
                pass

    def _next_seq_id(self) -> str:
        """Generate the next ``PROD-NNNNN`` ID (thread-safe)."""
        with self._lock:
            self._max_seq += 1
            return f"PROD-{self._max_seq:05d}"

    def _fuzzy_match(self, description: str) -> Optional[str]:
        """Return the ``product_id`` of the best fuzzy match, or *None*.

        Uses ``difflib.SequenceMatcher`` with a ratio threshold of 0.85.
        """
        desc_lower = description.lower()
        best_id: Optional[str] = None
        best_ratio: float = 0.0

        with self._lock:
            for pid, (cached_lower, _) in self._cache.items():
                ratio = SequenceMatcher(None, desc_lower, cached_lower).ratio()
                if ratio >= _FUZZY_THRESHOLD and ratio > best_ratio:
                    best_ratio = ratio
                    best_id = pid

        if best_id:
            logger.debug(
                "Fuzzy matched '%.60s' → %s (ratio=%.3f)",
                description, best_id, best_ratio,
            )
        return best_id

    def _upsert(
        self,
        product_id: str,
        item_description: str,
        unit_price: Optional[float],
        currency: Optional[str],
        unit_of_measure: Optional[str],
        doc_type: Optional[str],
        doc_id: Optional[str],
        now: datetime,
    ) -> None:
        """Insert or update a product row and refresh the local cache entry."""
        try:
            conn = self._get_conn()
            with conn.cursor() as cur:
                cur.execute(_UPSERT_SQL, (
                    # INSERT values
                    product_id, item_description, unit_price, currency,
                    unit_of_measure, now, now,
                    doc_type, doc_id,
                    now, now,
                    # ON CONFLICT — description length comparison
                    item_description, item_description,
                ))
            conn.commit()
        except Exception:
            logger.exception("Failed to upsert product %s", product_id)
            raise

        # Update local cache
        with self._lock:
            existing = self._cache.get(product_id)
            if existing and len(existing[1]) >= len(item_description):
                # Keep the longer description already in cache
                pass
            else:
                self._cache[product_id] = (item_description.lower(), item_description)
            self._track_seq(product_id)
