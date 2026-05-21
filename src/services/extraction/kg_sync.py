"""Refresh the Knowledge Graph with a single promoted stg row.

Called from the process_monitor_watcher *after* dispatch_document returns
``status='promoted'``. The watcher has the live ``agent_nick`` object, so
this thin adapter borrows it to construct the existing
``KGIngestionService`` and re-ingests just the one stg row that landed.

Why per-row rather than batch:
  - The renovation pipeline processes documents one at a time, so a
    per-row sync keeps the KG in lock-step with stg.
  - Re-ingesting the same row idempotently (Neo4j MERGE) ensures
    re-extractions update the graph rather than duplicating nodes.

Failures are NEVER raised back to the caller — KG sync is a downstream
side effect of a successful promotion. If Neo4j is down or the row can't
be re-read for any reason, we log and move on. The stg row is the source
of truth; KG can be rebuilt from stg at any time.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_STG_TABLE = {
    "invoice": "proc.bp_invoice_stg",
    "purchase_order": "proc.bp_purchase_order_stg",
    "quote": "proc.bp_quote_stg",
}
_PK_COL = {
    "invoice": "invoice_id",
    "purchase_order": "po_id",
    "quote": "quote_id",
}


def sync_row_to_kg(
    agent_nick: Any,
    doc_type: str,
    doc_pk: str | None,
) -> int:
    """Refresh the KG node for one (doc_type, doc_pk) row.

    Returns the number of rows ingested (0 or 1). Never raises.
    """
    if not doc_pk:
        return 0
    if agent_nick is None:
        logger.debug("kg_sync: no agent_nick — skipping KG refresh")
        return 0
    stg_t = _STG_TABLE.get(doc_type)
    pk_col = _PK_COL.get(doc_type)
    if not stg_t or not pk_col:
        logger.debug("kg_sync: unknown doc_type=%r — skipping", doc_type)
        return 0

    try:
        import pandas as pd  # local import — keep cold start small
        from src.services.db import get_conn
        from src.services.kg_ingestion_service import KGIngestionService

        with get_conn() as conn:
            df = pd.read_sql(
                f"SELECT * FROM {stg_t} WHERE {pk_col} = %s",
                conn,
                params=(doc_pk,),
            )
        if df.empty:
            logger.warning(
                "kg_sync: no stg row found for %s pk=%s — skipping", doc_type, doc_pk,
            )
            return 0

        kg = KGIngestionService(agent_nick)
        n = kg.ingest_dataframe(df, doc_type, source=f"{stg_t}:{doc_pk}")
        logger.info(
            "AgentNick: KG refreshed %s pk=%s rows=%d", doc_type, doc_pk, n,
        )
        return n
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "kg_sync failed for %s pk=%s: %s", doc_type, doc_pk, exc,
        )
        return 0
