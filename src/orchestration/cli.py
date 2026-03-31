"""CLI entry point for running agent workers.

Usage:
    python -m src.orchestration.cli --all
    python -m src.orchestration.cli --agents supplier_ranking,negotiation
    python -m src.orchestration.cli --agents data_extraction --concurrency 4

Spec reference: Section 4 of orchestration-rearchitecture-design.md
"""
from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
from typing import List

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ProcWise Agent Worker")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="Run workers for all agent types")
    group.add_argument("--agents", type=str, help="Comma-separated agent types to serve")
    parser.add_argument("--concurrency", type=int, default=1, help="Workers per agent type")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    return parser.parse_args()


ALL_AGENT_TYPES = [
    "data_extraction", "supplier_ranking", "quote_comparison",
    "quote_evaluation", "opportunity_miner", "email_drafting",
    "email_dispatch", "email_watcher", "negotiation",
    "supplier_interaction", "approvals", "discrepancy_detection", "rag",
]


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))

    if args.all:
        agent_types = ALL_AGENT_TYPES
    else:
        agent_types = [a.strip() for a in args.agents.split(",")]

    logger.info(
        "Starting workers for %d agent type(s): %s (concurrency=%d)",
        len(agent_types), agent_types, args.concurrency,
    )

    # Import here to avoid loading heavy deps at parse time
    import redis
    from config.settings import Settings
    from orchestration.worker import AgentWorker
    from orchestration.worker_context import WorkerContext

    settings = Settings()
    redis_url = settings.redis_streams_url or settings.redis_url
    redis_client = redis.from_url(redis_url)

    worker_ctx = WorkerContext(settings=settings, agent_types=agent_types)

    shutdown = threading.Event()

    def _signal_handler(signum, frame):
        logger.info("Shutdown signal received, finishing current tasks...")
        shutdown.set()

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    threads: List[threading.Thread] = []
    for agent_type in agent_types:
        for i in range(args.concurrency):
            worker = AgentWorker(
                agent_type=agent_type,
                worker_context=worker_ctx,
                redis_client=redis_client,
                consumer_name=f"worker-{agent_type}-{i}",
            )
            t = threading.Thread(
                target=_run_worker_loop,
                args=(worker, redis_client, shutdown),
                name=f"worker-{agent_type}-{i}",
                daemon=True,
            )
            threads.append(t)
            t.start()

    logger.info("All %d worker threads started", len(threads))
    shutdown.wait()
    logger.info("Shutdown complete")


def _run_worker_loop(
    worker: "AgentWorker", redis_client, shutdown: threading.Event
) -> None:
    from orchestration.message_protocol import TaskMessage

    stream = f"agent:tasks:{worker.agent_type}"
    group = "workers"

    # Ensure consumer group exists
    try:
        redis_client.xgroup_create(stream, group, id="0", mkstream=True)
    except Exception:
        pass  # Group already exists

    while not shutdown.is_set():
        try:
            messages = redis_client.xreadgroup(
                group, worker._consumer, {stream: ">"}, count=1, block=5000,
            )
            if not messages:
                continue
            for _stream, entries in messages:
                for msg_id, fields in entries:
                    task = TaskMessage.from_redis(fields)
                    worker.execute_task(task)
                    redis_client.xack(stream, group, msg_id)
        except Exception:
            if not shutdown.is_set():
                logger.exception("Worker %s error", worker._consumer)


if __name__ == "__main__":
    main()
