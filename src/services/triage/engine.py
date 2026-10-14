"""Discrepancy triage engine.

Spec: docs/superpowers/specs/2026-09-24-discrepancy-triage-p2p-design.md

triage_set is the whole pure pipeline for one deal. run_triage batches it over many
deals: loads each batch in a fixed number of queries, isolates a failing deal, and
writes each batch in one transaction. load_config() runs first, so a missing
tolerance aborts the run before anything is written.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Callable, Iterable, Optional

from src.services.db import get_conn

from . import loader, writer
from .checks import run_checks
from .fingerprint import deal_content_hash
from .group import group
from .link import link
from .model import DocumentSet, Finding, Result, Severity, Verdict
from .report import RunReport
from .score import score_result
from .text import describe
from .tolerance import TriageConfig, load_config
from .verdict import verdict

log = logging.getLogger(__name__)

# Only a completed full backfill is a baseline: a --deals or --limit run ('single') has
# not triaged the corpus, and the scheduler must not start from one.
_BASELINE_SQL = """
SELECT 1 FROM proc.bp_triage_run
 WHERE mode = 'backfill' AND finished_at IS NOT NULL AND rolled_back_at IS NULL
 LIMIT 1
"""


@dataclass
class DealOutput:
    deal_id: str
    results: list[Result]
    findings: list[Finding]
    verdict: Optional[Verdict]
    content_hash: str = ""       # "" = the deal's documents have all gone (see vanished())

    @classmethod
    def vanished(cls, deal_id: str) -> "DealOutput":
        """A deal that no longer has documents: no results, so the writer closes its
        untouched open findings and drops its state row."""
        return cls(deal_id, [], [], None, "")


def triage_set(ds: DocumentSet, cfg: TriageConfig) -> DealOutput:
    links = link(ds, cfg)
    results = run_checks(ds, links, cfg)
    for r in results:
        score_result(r, cfg)
    findings = [describe(f) for f in group(results, cfg)]
    return DealOutput(ds.deal_id, results, findings, verdict(ds, links, findings, results),
                      deal_content_hash(ds))


def _chunks(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def run_triage(deal_ids: Iterable[str], mode: str, *, dry_run: bool = False,
               cfg: Optional[TriageConfig] = None, connect: Callable = get_conn,
               on_batch: Optional[Callable[[RunReport], None]] = None,
               known_gaps: Iterable[str] = (), vanished: Iterable[str] = ()) -> RunReport:
    """`vanished`: requested deals that have a bp_triage_deal_state row. One of them
    that loads no documents is written as an empty DealOutput (its findings close)
    instead of being counted as a deal without documents."""
    cfg = cfg or load_config()
    ids = list(dict.fromkeys(deal_ids))
    had_state = set(vanished)
    report = RunReport(mode=mode, dry_run=dry_run, deals_requested=len(ids))
    report.known_gaps.extend(known_gaps)
    with connect() as conn:
        if not dry_run:
            report.run_id = writer.start_run(conn, mode, cfg)
        for batch in _chunks(ids, int(cfg["batch_size"])):
            try:
                sets = loader.load_deal_sets(conn.cursor(), batch)
            except Exception as exc:  # noqa: BLE001 - a batch that cannot load is reported
                log.exception("triage: loading batch failed")
                for d in batch:
                    report.fail(d, f"load failed: {exc}")
                continue
            outputs = []
            for d in batch:
                ds = sets.get(d)
                if ds is None:
                    if d in had_state:
                        outputs.append(DealOutput.vanished(d))
                    else:
                        report.deals_without_documents += 1
                    continue
                try:
                    outputs.append(triage_set(ds, cfg))
                except Exception as exc:  # noqa: BLE001 - one bad deal must not stop the run
                    log.exception("triage: deal %s failed", d)
                    report.fail(d, repr(exc))
            if not dry_run and outputs:
                try:
                    report.write_counts.update(writer.write_batch(conn, report.run_id, outputs))
                except Exception as exc:  # noqa: BLE001 - the batch rolled back; say so
                    log.exception("triage: writing batch failed")
                    for o in outputs:
                        report.fail(o.deal_id, f"write failed: {exc}")
                    continue
            for o in outputs:
                if o.content_hash:
                    report.add(o)
                else:
                    report.deals_vanished += 1
            if on_batch:
                on_batch(report)
        report.finish()
        if not dry_run:
            writer.finish_run(conn, report.run_id, report)
    return report


def view_dict(out: DealOutput, finding_ids: Optional[dict] = None) -> dict:
    ids = finding_ids or {}
    v = out.verdict
    shown = sorted((f for f in out.findings if f.severity >= Severity.S2),
                   key=lambda f: (-int(f.severity), -(f.exposure_gbp or Decimal("0"))))
    return {
        "deal_id": v.deal_id, "verdict": v.verdict, "summary": v.summary,
        "counts": {"s1": v.s1, "s2": v.s2, "notes": v.notes},
        "exposure_gbp": str(v.exposure_gbp),
        "findings": [{"rule_id": f.rule_id, "category": f.category,
                      "severity": f.severity.name, "headline": f.headline, "text": f.text,
                      "exposure_gbp": None if f.exposure_gbp is None else str(f.exposure_gbp),
                      "finding_id": ids.get(f.fingerprint)} for f in shown],
    }


def triage_deal_view(deal_id: str, *, cfg: Optional[TriageConfig] = None,
                     connect: Callable = get_conn) -> Optional[dict]:
    """The deal's verdict now (a dry run), with Action Centre ids where they exist."""
    cfg = cfg or load_config()
    with connect() as conn:
        cur = conn.cursor()
        ds = loader.load_deal_sets(cur, [deal_id]).get(deal_id)
        if ds is None:
            return None
        out = triage_set(ds, cfg)
        ids = writer.finding_ids(cur, [f.fingerprint for f in out.findings
                                       if f.severity >= Severity.S2])
    return view_dict(out, ids)


def _has_baseline(cur) -> bool:
    cur.execute(_BASELINE_SQL)
    return cur.fetchone() is not None


def _select_changed(cur, cfg: TriageConfig) -> tuple[list[str], set[str]]:
    """Deals whose content or tolerances differ from their last successful triage.

    Returns (deals to run, those among them that have a state row). A deal is selected
    when it has no state row (new, or it failed last time), its content hash differs, or
    the tolerance fingerprint differs; and a state-row deal that no longer loads any
    documents is selected so its findings close. A batch that cannot load is skipped:
    its deals keep their state and are looked at again next pass.
    """
    state = writer.deal_state(cur)
    ids = list(dict.fromkeys([*loader.list_deal_ids(cur), *sorted(state)]))
    selected: list[str] = []
    for batch in _chunks(ids, int(cfg["batch_size"])):
        try:
            sets = loader.load_deal_sets(cur, batch)
        except Exception:  # noqa: BLE001 - retried next pass
            log.exception("triage: loading a batch to fingerprint failed")
            continue
        for d in batch:
            ds, prior = sets.get(d), state.get(d)
            if ds is None:
                if prior is not None:
                    selected.append(d)
            elif (prior is None or prior[0] != deal_content_hash(ds)
                  or prior[1] != cfg.fingerprint):
                selected.append(d)
    return selected, set(selected) & set(state)


def run_changed(*, cfg: Optional[TriageConfig] = None,
                connect: Callable = get_conn) -> Optional[RunReport]:
    """Re-triage deals whose documents (or the tolerances) changed since their last
    successful triage, by comparing each deal's content hash with bp_triage_deal_state.

    Does nothing until a full backfill has completed: the first pass over the whole
    corpus is a deliberate, reported act, not a side effect of the scheduler starting.
    Writes nothing -- not even a run row -- when nothing changed.
    """
    with connect() as conn:
        cur = conn.cursor()
        if not _has_baseline(cur):
            log.info("triage: no completed full backfill yet; nothing scheduled")
            return None
        cfg = cfg or load_config()
        ids, had_state = _select_changed(cur, cfg)
    if not ids:
        return None
    return run_triage(ids, "scheduled", cfg=cfg, connect=connect, vanished=had_state)
