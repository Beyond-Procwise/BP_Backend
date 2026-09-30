"""Walk the open findings, select, propose. Nothing else.

Batched on the key, not on OFFSET. Deal assignment already proved that a scan
re-reading from the start does not survive this corpus, and there are 4,838
open detection findings today.

The count is logged on EVERY run, including zero. A sweep that logged only
when it found something would be indistinguishable from one that had silently
stopped running -- and an empty playbook table, which is the honest state until
an expert authors a strategy, produces exactly that zero.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterator, Optional

from src.services.db import get_conn

from . import proposer
from .finding_source import OPEN_SQL, SOURCES, normalise
from .selector import select, tied_candidates
from .store import PlaybookStore, load_playbook_store

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 500


@dataclass
class SweepReport:
    scanned: int = 0
    proposed: int = 0
    already_queued: int = 0
    ambiguous: int = 0
    unmatched: int = 0

    def render(self) -> str:
        return (
            f"playbook sweep: {self.scanned} scanned, {self.proposed} proposed, "
            f"{self.already_queued} already queued, {self.ambiguous} ambiguous, "
            f"{self.unmatched} unmatched"
        )


def _pages(conn: Any, source: str, batch_size: int) -> Iterator[list]:
    """Rows of ``source``, keyset-paginated. Yields dicts, a page at a time."""

    # Start below every key the source holds. finding_id is a BIGINT and its
    # identity sequence starts at 1; opportunity_id is a VARCHAR and the empty
    # string sorts below every non-empty one.
    cursor_key: Any = 0 if source == "detection_finding" else ""
    while True:
        cur = conn.cursor()
        try:
            cur.execute(OPEN_SQL[source], (cursor_key, batch_size))
            columns = [c[0] for c in cur.description]
            rows = [dict(zip(columns, r)) for r in cur.fetchall()]
        finally:
            cur.close()
        if not rows:
            return
        yield rows
        last = rows[-1]
        cursor_key = last["finding_id" if source == "detection_finding" else "opportunity_id"]


def sweep(
    *,
    store: Optional[PlaybookStore] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    conn: Optional[Any] = None,
) -> SweepReport:
    """Propose a playbook for every open finding that one governs.

    Proposes only. Nothing here starts a workflow; the proposal waits for a
    person to accept it through the endpoint.
    """

    active = store if store is not None else load_playbook_store()
    if active is None:
        logger.error("playbook sweep skipped: the store could not be read")
        return SweepReport()

    report = SweepReport()
    if conn is not None:
        _run(conn, active, batch_size, report)
    else:
        with get_conn() as own:
            _run(own, active, batch_size, report)

    logger.info("%s", report.render())
    return report


def _run(conn: Any, store: PlaybookStore, batch_size: int, report: SweepReport) -> None:
    for source in SOURCES:
        playbooks = store.for_source(source)
        for page in _pages(conn, source, batch_size):
            for row in page:
                report.scanned += 1
                try:
                    finding = normalise(source, row)
                except ValueError:
                    logger.exception("skipping unreadable %s row", source)
                    continue
                if not playbooks:
                    report.unmatched += 1
                    continue
                selection = select(finding, playbooks)
                if selection is None:
                    tied = tied_candidates(finding, playbooks)
                    if tied:
                        proposer.record_ambiguous(finding, tied)
                        report.ambiguous += 1
                    else:
                        report.unmatched += 1
                    continue
                if proposer.propose(finding, selection, conn=conn) is None:
                    report.already_queued += 1
                else:
                    report.proposed += 1
