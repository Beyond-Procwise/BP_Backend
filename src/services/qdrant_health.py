"""Qdrant health monitoring and auto-recovery service.

Design principle: Qdrant must **never stay down**. When a connection failure
is detected, the service automatically restarts the Qdrant Docker container
and reconnects the client. All operations should retry through this layer.
"""

from __future__ import annotations

import logging
import os
import subprocess
import threading
import time
from typing import Optional

import requests
from qdrant_client import QdrantClient

logger = logging.getLogger(__name__)

_COMPOSE_FILE = os.getenv(
    "PROCWISE_COMPOSE_FILE",
    "/home/muthu/PycharmProjects/BP_Backend/procurement_knowledge_graph/docker-compose.yml",
)
_ENV_FILE = os.getenv(
    "PROCWISE_ENV_FILE",
    "/home/muthu/PycharmProjects/BP_Backend/.env",
)

# Recovery coordination
_recovery_lock = threading.Lock()
_last_recovery_attempt: float = 0.0
_RECOVERY_COOLDOWN = 30.0  # seconds between restart attempts


def is_qdrant_healthy(url: str = "http://localhost:6333", api_key: Optional[str] = None) -> bool:
    """Check if Qdrant is responding to health checks.

    Works for both local (HTTP) and cloud (HTTPS + API key) instances.
    """
    try:
        headers = {}
        if api_key:
            headers["api-key"] = api_key
        resp = requests.get(f"{url}/healthz", timeout=5, headers=headers)
        return resp.status_code == 200
    except Exception:
        return False


def _restart_qdrant_container() -> bool:
    """Restart the Qdrant Docker container via docker compose."""
    global _last_recovery_attempt

    if not _recovery_lock.acquire(blocking=False):
        logger.info("Qdrant recovery already in progress by another thread")
        # Wait for the other thread's recovery to complete
        with _recovery_lock:
            pass
        return True

    try:
        now = time.monotonic()
        if now - _last_recovery_attempt < _RECOVERY_COOLDOWN:
            logger.info(
                "Qdrant recovery cooldown active (%.0fs remaining)",
                _RECOVERY_COOLDOWN - (now - _last_recovery_attempt),
            )
            return False
        _last_recovery_attempt = now

        logger.warning("Restarting Qdrant via docker compose...")

        # First try restart
        try:
            result = subprocess.run(
                [
                    "docker", "compose",
                    "--env-file", _ENV_FILE,
                    "-f", _COMPOSE_FILE,
                    "restart", "qdrant",
                ],
                capture_output=True,
                text=True,
                timeout=90,
            )
            if result.returncode != 0:
                logger.warning(
                    "Qdrant restart returned code %d: %s",
                    result.returncode,
                    result.stderr.strip(),
                )
                # Try 'up -d' instead (container may have been removed)
                subprocess.run(
                    [
                        "docker", "compose",
                        "--env-file", _ENV_FILE,
                        "-f", _COMPOSE_FILE,
                        "up", "-d", "qdrant",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=90,
                )
        except subprocess.TimeoutExpired:
            logger.error("Docker compose command timed out for Qdrant restart")
            return False
        except Exception as exc:
            logger.error("Docker compose restart failed for Qdrant: %s", exc)
            return False

        # Wait for Qdrant to become healthy
        for check in range(30):
            if is_qdrant_healthy():
                logger.info("Qdrant recovered after restart (check %d)", check + 1)
                return True
            time.sleep(2)

        logger.error("Qdrant did not recover within 60s after restart")
        return False
    finally:
        _recovery_lock.release()


def ensure_qdrant_available(
    url: str = "http://localhost:6333",
    max_wait: int = 120,
    api_key: Optional[str] = None,
) -> bool:
    """Ensure Qdrant is up, restarting the container if needed.

    Blocks until Qdrant is healthy or ``max_wait`` seconds elapse.
    For cloud-hosted Qdrant, only health checks are performed (no
    Docker restart since the container is managed externally).
    """
    if is_qdrant_healthy(url, api_key=api_key):
        return True

    is_cloud = url.startswith("https://") or "cloud.qdrant.io" in url
    logger.warning("Qdrant is not responding at %s — initiating recovery", url)

    if is_cloud:
        # Cloud Qdrant: we can't restart it, just wait and hope it recovers
        logger.warning(
            "Cloud Qdrant detected — cannot restart container, waiting for recovery"
        )
    else:
        _restart_qdrant_container()

    deadline = time.monotonic() + max_wait
    while time.monotonic() < deadline:
        if is_qdrant_healthy(url, api_key=api_key):
            return True
        time.sleep(3)

    return is_qdrant_healthy(url, api_key=api_key)


def reconnect_qdrant_client(
    agent_nick,
    *,
    url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> QdrantClient:
    """Reconnect the QdrantClient on ``agent_nick`` after a recovery.

    Creates a fresh client instance and replaces the old one on the shared
    ``agent_nick`` object so all agents pick up the new connection.
    """
    qdrant_url = url or getattr(agent_nick.settings, "qdrant_url", "http://localhost:6333")
    qdrant_api_key = api_key or getattr(agent_nick.settings, "qdrant_api_key", None)

    new_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

    # Verify the new client works
    try:
        new_client.get_collections()
    except Exception as exc:
        logger.warning("New Qdrant client verification failed: %s", exc)

    agent_nick.qdrant_client = new_client
    logger.info("Qdrant client reconnected at %s", qdrant_url)
    return new_client


def with_qdrant_recovery(agent_nick, operation, *args, **kwargs):
    """Execute a Qdrant operation with automatic recovery on failure.

    If the operation fails due to a connection error, this function:
    1. Ensures Qdrant is running (restarts if needed)
    2. Reconnects the client
    3. Retries the operation

    Parameters
    ----------
    agent_nick : AgentNick
        The shared agent context holding the Qdrant client.
    operation : callable
        The Qdrant operation to execute (e.g., client.search).
    *args, **kwargs
        Arguments passed to the operation.

    Returns the result of the operation.
    """
    max_retries = 3
    last_exc = None

    for attempt in range(1, max_retries + 1):
        try:
            return operation(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            error_str = str(exc).lower()
            is_connection_error = any(
                token in error_str
                for token in ("connection", "refused", "unreachable", "timed out", "reset")
            )

            if not is_connection_error and attempt == 1:
                # Not a connection issue — might be a bad query, re-raise
                raise

            logger.warning(
                "Qdrant operation failed (attempt %d/%d): %s — recovering",
                attempt,
                max_retries,
                exc,
            )

            qdrant_url = getattr(
                agent_nick.settings, "qdrant_url", "http://localhost:6333"
            )
            if ensure_qdrant_available(url=qdrant_url):
                reconnect_qdrant_client(agent_nick)
            else:
                logger.error("Qdrant recovery failed on attempt %d", attempt)
                if attempt < max_retries:
                    time.sleep(5 * attempt)

    raise RuntimeError(
        f"Qdrant operation failed after {max_retries} recovery attempts"
    ) from last_exc
