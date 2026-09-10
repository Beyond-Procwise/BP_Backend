"""Refresh the Knowledge Graph with a single promoted _trgt row.

Called from ``BackendScheduler._run_trgt_promotion`` after a row reaches
_trgt. It builds the existing ``KGIngestionService`` and re-ingests that one
row.

WHY _trgt AND NOT _stg. This used to read _stg and fire from the
process_monitor_watcher the moment dispatch returned ``promoted`` — which is
_stg promotion, not _trgt. The graph mirrors _trgt: _trgt is the final,
accepted state of a document, and what sits there is what is approved and
transacted against. Syncing from _stg put nodes in the graph for documents that
had not reached that state, and the reconciling rebuild then swept them as
absent from _trgt — so the graph oscillated for exactly the documents still in
flight. Observed on invoice 0526: present in bp_invoice_stg, absent from
bp_invoice_trgt, synced by this function and removed by the next rebuild.

Why per-row rather than batch:
  - The renovation pipeline processes documents one at a time, so a
    per-row sync keeps the KG in lock-step with stg.
  - Re-ingesting the same row idempotently (Neo4j MERGE) ensures
    re-extractions update the graph rather than duplicating nodes.

Failures are NEVER raised back to the caller — KG sync is a downstream
side effect of a successful promotion. If Neo4j is down or the row can't
be re-read for any reason, we log and move on. The _trgt row is the source
of truth; the KG can be rebuilt from _trgt at any time.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_TRGT_TABLE = {
    "invoice": "proc.bp_invoice_trgt",
    "purchase_order": "proc.bp_purchase_order_trgt",
    "quote": "proc.bp_quote_trgt",
    # Contracts do not go through _stg/_trgt: contract.yaml writes straight to
    # proc.bp_contracts. Without this entry a contract could be extracted and
    # still never reach the graph.
    "contract": "proc.bp_contracts",
}
_PK_COL = {
    "invoice": "invoice_id",
    "purchase_order": "po_id",
    "quote": "quote_id",
    "contract": "contract_id",
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
    trgt_t = _TRGT_TABLE.get(doc_type)
    pk_col = _PK_COL.get(doc_type)
    if not trgt_t or not pk_col:
        logger.debug("kg_sync: unknown doc_type=%r — skipping", doc_type)
        return 0

    try:
        import pandas as pd  # local import — keep cold start small
        from src.services.db import get_conn
        from src.services.kg_ingestion_service import KGIngestionService

        with get_conn() as conn:
            df = pd.read_sql(
                f"SELECT * FROM {trgt_t} WHERE {pk_col} = %s",
                conn,
                params=(doc_pk,),
            )
        if df.empty:
            logger.warning(
                "kg_sync: no _trgt row for %s pk=%s — not synced (it has not "
                "reached final state)", doc_type, doc_pk,
            )
            return 0

        kg = KGIngestionService(agent_nick)
        n = kg.ingest_dataframe(df, doc_type, source=f"{trgt_t}:{doc_pk}")
        logger.info(
            "AgentNick: KG refreshed %s pk=%s rows=%d", doc_type, doc_pk, n,
        )
        return n
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "kg_sync failed for %s pk=%s: %s", doc_type, doc_pk, exc,
        )
        return 0
