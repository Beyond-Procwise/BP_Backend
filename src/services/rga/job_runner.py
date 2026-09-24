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
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from src.services.rga import job_store as _store
from src.services.rga.pipeline import generate_report as _generate

logger = logging.getLogger(__name__)

_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rga-report")


def submit(job_id: str) -> None:
    """Queue a filed job. Returns at once."""
    _EXECUTOR.submit(run_job, job_id)


def run_job(job_id: str, *, store: Any = _store,
            generate: Callable[..., Any] = _generate) -> None:
    """Run one job to a terminal state. Never raises."""
    if not store.claim(job_id):
        # Already claimed, finished, or healed after a restart: not ours to run.
        return
    try:
        job = store.get(job_id)
        run = generate(job["report_type"], scope=job["scope"], as_of=job["as_of"])
        if run.released and run.artefact is not None:
            store.finish_released(
                job_id, run_id=run.run_id, stage_reached=run.stage_reached,
                deck=run.artefact.content, media_type=run.artefact.media_type,
                filename=f"{run.report_type_id}_{run.run_id}.pptx")
        else:
            store.finish_blocked(
                job_id, run_id=run.run_id, stage_reached=run.stage_reached,
                blocking=[{"finding_id": f.finding_id, "code": f.code.value,
                           "severity": f.severity.value, "detail": f.detail}
                          for f in run.findings if f.blocks_release])
    except Exception as exc:  # noqa: BLE001 - a dead worker would strand the job
        logger.exception("rga: report job %s failed", job_id)
        try:
            store.finish_failed(job_id, f"The report could not be built: {exc}")
        except Exception:  # noqa: BLE001
            logger.exception("rga: could not mark report job %s failed", job_id)
