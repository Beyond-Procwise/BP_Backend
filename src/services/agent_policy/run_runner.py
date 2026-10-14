"""The extraction worker: one run at a time, and every run leaves 'running'.

A single worker thread, because a run holds the GPU for minutes and two at once
would each take twice as long. A heartbeat thread beats every 30 s for the
duration of ``work`` (stopped when it ends) so a long model call stays vouched
for, and the runs queued behind it with it. ``submit`` and ``run`` never raise.

``work(conn, run, emit)`` does the run: ``emit(kind, payload, **fields)`` appends
an item and returns its seq; ``work`` returns the counts dict.
"""
from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Optional

from services.agent_policy import run_store as _store
from services.db import get_conn as _get_conn

logger = logging.getLogger(__name__)

_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="policy-extraction")
HEARTBEAT_SECONDS = 30


def _one_line(exc: BaseException) -> str:
    text = " ".join(str(exc).split()) or type(exc).__name__
    return text[:300]


def _heartbeat(run_id: int, stop: threading.Event, store: Any, conn_factory: Callable,
               interval: float) -> None:
    while not stop.wait(interval):
        try:
            with conn_factory() as conn:
                store.beat(conn, run_id, store.OWNER)
                store.beat_queued(conn, store.OWNER)   # the runs waiting behind this one
        except Exception:  # noqa: BLE001 - one missed beat is not a reason to stop
            logger.exception("agent_policy: extraction heartbeat failed for run %s", run_id)


def run(run_id: int, work: Callable[..., Optional[Dict[str, Any]]], *, store: Any = _store,
        conn_factory: Callable = _get_conn, interval: float = HEARTBEAT_SECONDS) -> None:
    """Run one run to a terminal state. Never raises."""
    stop = threading.Event()
    beater: Optional[threading.Thread] = None
    try:
        with conn_factory() as conn:
            if not store.claim(conn, run_id, store.OWNER):
                return  # claimed, finished, or healed already: not ours to run
            beater = threading.Thread(target=_heartbeat, name="policy-extraction-heartbeat",
                                      args=(run_id, stop, store, conn_factory, interval),
                                      daemon=True)
            beater.start()
            try:
                current = store.get(conn, run_id)

                def emit(kind: str, payload: Dict[str, Any], **fields: Any) -> int:
                    return store.append_item(conn, run_id, kind=kind, payload=payload, **fields)

                counts = work(conn, current, emit)
                if not store.finish(conn, run_id, "done", counts=counts or {}):
                    logger.warning("agent_policy: extraction run %s finished but was no longer "
                                   "active (healed or closed meanwhile); its result was not "
                                   "recorded as done", run_id)
            except Exception as exc:  # noqa: BLE001 - a dead worker would strand the run
                logger.exception("agent_policy: extraction run %s failed", run_id)
                try:
                    store.finish(conn, run_id, "failed", counts={}, error=_one_line(exc))
                except Exception:  # noqa: BLE001
                    logger.exception("agent_policy: could not mark run %s failed", run_id)
    except Exception:  # noqa: BLE001
        logger.exception("agent_policy: extraction run %s could not start", run_id)
    finally:
        stop.set()
        if beater is not None:
            beater.join(timeout=5)


def submit(run_id: int, work: Callable[..., Optional[Dict[str, Any]]], *,
           executor: Any = None, store: Any = _store,
           conn_factory: Callable = _get_conn) -> None:
    """Queue a filed run. Returns at once; never raises.

    If it cannot be queued, the run is marked failed so it is not left queued forever.
    Queuing stamps this process's queued runs, so a run waiting its turn is not healed.
    """
    try:
        with conn_factory() as conn:
            store.beat_queued(conn, store.OWNER)
    except Exception:  # noqa: BLE001 - an unstamped run is judged by its age instead
        logger.exception("agent_policy: could not stamp queued run %s", run_id)
    try:
        (executor or _EXECUTOR).submit(run, run_id, work)
    except Exception as exc:  # noqa: BLE001
        logger.exception("agent_policy: could not queue extraction run %s", run_id)
        try:
            with conn_factory() as conn:
                store.finish(conn, run_id, "failed", counts={},
                             error=f"The run could not be queued: {_one_line(exc)}")
        except Exception:  # noqa: BLE001
            logger.exception("agent_policy: could not mark run %s failed", run_id)
