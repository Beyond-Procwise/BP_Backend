"""The report worker: one report at a time, and every job leaves 'running'.

ONE AT A TIME

Composing a report holds the GPU for minutes. Two at once would each take twice
as long and crowd out extraction, which shares the card, so jobs queue behind a
single worker thread. A queued job simply waits its turn in the executor.

WHY THE JOB'S OWN AS-OF

The as-of is fixed when the job is filed, not when the worker reaches it. A job
queued at 23:59 and run at 00:03 is still the report that was asked for, and its
Fact Pack id -- derived from scope and as-of -- still matches the request.
"""
from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from src.services.rga import audit, signoff
from src.services.rga import job_store as _store
from src.services.rga.pipeline import generate_report as _generate

logger = logging.getLogger(__name__)

_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rga-report")

# The heartbeat is its own thread, not a step of the worker: the worker spends
# minutes inside one model call, and a job queued behind it must stay vouched
# for all that time. See job_store's docstring for what a missed beat means.
_HEARTBEAT_SECONDS = 30
_heartbeat = None
_heartbeat_lock = threading.Lock()


def _beat_forever() -> None:
    while True:
        try:
            _store.beat()
        except Exception:  # noqa: BLE001 - one missed beat is not a reason to stop
            logger.exception("rga: report job heartbeat failed")
        time.sleep(_HEARTBEAT_SECONDS)


def _ensure_heartbeat() -> None:
    global _heartbeat
    with _heartbeat_lock:
        if _heartbeat is None or not _heartbeat.is_alive():
            _heartbeat = threading.Thread(target=_beat_forever, daemon=True,
                                          name="rga-report-heartbeat")
            _heartbeat.start()


def submit(job_id: str) -> None:
    """Queue a filed job. Returns at once."""
    _ensure_heartbeat()
    _EXECUTOR.submit(run_job, job_id)


def run_job(job_id: str, *, store: Any = _store,
            generate: Callable[..., Any] = _generate) -> None:
    """Run one job to a terminal state. Never raises."""
    if not store.claim(job_id):
        # Already claimed, finished, or healed after a restart: not ours to run.
        return
    job = None
    try:
        job = store.get(job_id)
        # Every event this run writes, from any module, carries the job id and
        # who asked; SCOPE also records the entitlement the job was filed under.
        with audit.run_context(job_id=job_id, requested_by=job.get("requested_by"),
                               entitlement=job.get("entitlement")):
            run = generate(job["report_type"], scope=job["scope"], as_of=job["as_of"])
        if run.released and run.artefact is not None:
            store.finish_released(
                job_id, run_id=run.run_id, stage_reached=run.stage_reached,
                deck=run.artefact.content, media_type=run.artefact.media_type,
                filename=f"{run.report_type_id}_{run.run_id}.pptx",
                page=run.page.content if run.page is not None else None,
                page_media_type=run.page.media_type if run.page is not None else None)
            # Released is not yet allowed to leave: the policy decides whether a person
            # must sign it off first, and the trail records that one was asked for.
            if signoff.required(run.report_type_id):
                with audit.run_context(job_id=job_id, requested_by=job.get("requested_by")):
                    audit.emit(audit.APPROVAL_REQUESTED, run_id=run.run_id,
                               agent="rga_job_runner", summary="awaiting sign-off",
                               details={"policy": "ReportSignoffPolicy"})
        else:
            store.finish_blocked(
                job_id, run_id=run.run_id, stage_reached=run.stage_reached,
                blocking=[{"finding_id": f.finding_id, "code": f.code.value,
                           "severity": f.severity.value, "detail": f.detail}
                          for f in run.findings if f.blocks_release])
    except Exception as exc:  # noqa: BLE001 - a dead worker would strand the job
        logger.exception("rga: report job %s failed", job_id)
        error = f"The report could not be built: {exc}"
        try:
            store.finish_failed(job_id, error)
        except Exception:  # noqa: BLE001
            logger.exception("rga: could not mark report job %s failed", job_id)
        # Close the trail: a crash otherwise leaves events that just stop.
        try:
            with audit.run_context(job_id=job_id,
                                   requested_by=(job or {}).get("requested_by")):
                audit.emit(audit.RUN_FAILED, run_id=(job or {}).get("run_id") or job_id,
                           agent="rga_job_runner", status="failed", summary=error,
                           details={"reason": "crashed", "error": type(exc).__name__})
        except Exception:  # noqa: BLE001
            logger.exception("rga: could not audit the failure of report job %s", job_id)
