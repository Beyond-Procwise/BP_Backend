"""The embedding worker: claims outbox jobs, re-reads each source, embeds, and replaces that source's vectors atomically.

Safe to run twice: jobs carry no text, a source is always re-read as it is now, and ``VectorStore.replace`` is idempotent. A source that no
longer exists has its vectors removed. Failures retry with backoff and end 'failed' with the error recorded. Run it from the scheduler or
``python -m src.services.vector_store.worker`` (``--loop`` to keep running).
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, Optional

from . import outbox
from .chunking import SUMMARY_CHARS, chunk_text
from .loaders import GenericSourceLoader, default_loaders
from .profiles import PROFILES, EmbeddingProfile

logger = logging.getLogger(__name__)


class VectorWorker:
    def __init__(self, connect: Callable[[], Any], embedder: Any, *, loaders: Optional[Dict[str, Any]] = None,
                 store_factory: Optional[Callable[[Optional[EmbeddingProfile]], Any]] = None,
                 profile_resolver: Optional[Callable[[Optional[str]], Optional[EmbeddingProfile]]] = None,
                 batch: int = 10, visibility_seconds: int = 300) -> None:
        self._connect = connect
        self._embedder = embedder
        self._loaders = loaders or default_loaders()
        self._store_factory = store_factory or self._default_store
        self._resolve = profile_resolver or (lambda name: PROFILES[name] if name else None)
        self._batch, self._visibility = batch, visibility_seconds

    def _default_store(self, profile: Optional[EmbeddingProfile]):
        from .pgvector_store import PgVectorStore
        return PgVectorStore(self._connect, profile=profile)

    def _embedder_for(self, profile: Optional[EmbeddingProfile]) -> Any:
        return self._embedder if hasattr(self._embedder, "encode") else self._embedder(profile)

    def process(self, cur: Any, job: outbox.Job) -> str:
        """Do one job. Returns 'upserted' or 'deleted'. Raises on failure (the caller records it)."""
        profile = self._resolve(job.profile)
        store = self._store_factory(profile)
        if job.action == "delete":
            store.delete(job.tenant_id, job.source_id, collection=job.collection)
            return "deleted"
        loader = self._loaders.get(job.collection) or GenericSourceLoader(job.collection)
        doc = loader.load(cur, job.tenant_id, job.source_id)
        # A source that carries its own chunk_id is already ONE chunk (callers that chunk semantically write a source per chunk): embed it whole,
        # as the old path did. Only a whole-document source is split into windows.
        if doc and doc.metadata.get("chunk_id") is not None:
            chunks = [doc.text.strip()] if doc.text.strip() else []
        else:
            chunks = chunk_text(doc.text) if doc else []
        if not chunks:                                     # source gone, or nothing to embed: no vector may outlive it
            store.delete(job.tenant_id, job.source_id, collection=job.collection)
            return "deleted"
        vectors = self._embedder_for(profile).encode(chunks)
        # A source that is already one chunk of a larger document keeps the chunk_id it was written with; a source this worker splits is numbered.
        # ... and keeps its own summary: the worker's chunk summary is only a default for a source that has none.
        single = len(chunks) == 1

        def meta_for(i: int, chunk: str) -> Dict[str, Any]:
            numbered = {"chunk_id": i, "chunk_index": i, "summary": chunk.strip()[:SUMMARY_CHARS]}
            if single:
                return {**numbered, **doc.metadata, "chunk_index": doc.metadata.get("chunk_index", doc.metadata.get("chunk_id", 0))}
            return {**doc.metadata, **numbered}

        store.replace(job.tenant_id, job.source_id, [
            {"embedding": vec, "content": chunk, "metadata": meta_for(i, chunk)}
            for i, (chunk, vec) in enumerate(zip(chunks, vectors))], collection=job.collection)
        return "upserted"

    def run_once(self) -> Dict[str, int]:
        report = {"claimed": 0, "done": 0, "retried": 0, "failed": 0, "deleted": 0}
        with self._connect() as conn, conn.cursor() as cur:
            jobs = outbox.claim(cur, self._batch, self._visibility)
            report["claimed"] = len(jobs)
            for job in jobs:
                try:
                    if self.process(cur, job) == "deleted":
                        report["deleted"] += 1
                    outbox.complete(cur, job)
                    report["done"] += 1
                except Exception as exc:  # noqa: BLE001 - one bad source must not stop the batch
                    logger.warning("embedding job %s (%s/%s) failed: %s", job.job_id, job.collection, job.source_id, exc)
                    status = outbox.fail(cur, job, f"{type(exc).__name__}: {exc}")
                    report["failed" if status == "failed" else "retried"] += 1
        return report

    def run_until_idle(self, max_rounds: int = 100) -> Dict[str, int]:
        total = {"claimed": 0, "done": 0, "retried": 0, "failed": 0, "deleted": 0}
        for _ in range(max_rounds):
            r = self.run_once()
            for k in total:
                total[k] += r[k]
            if r["claimed"] == 0:
                break
        return total


def main(argv: Optional[list] = None) -> int:
    from services import tenant_context as _tc
    _tc.enter_system()  # a platform process: no signed-in user, spans every customer
    import argparse
    ap = argparse.ArgumentParser(description="Embed queued sources into the vector store.")
    ap.add_argument("--loop", action="store_true", help="keep running, polling for jobs")
    ap.add_argument("--interval", type=float, default=5.0, help="seconds between polls with --loop")
    ap.add_argument("--batch", type=int, default=10)
    args = ap.parse_args(argv)
    from src.services.db import get_conn
    from .embedder import SentenceTransformerEmbedder
    from .profiles import active_profile
    worker = VectorWorker(get_conn, lambda profile: SentenceTransformerEmbedder((profile or active_profile()).model), batch=args.batch)
    while True:
        report = worker.run_once()
        if report["claimed"]:
            logger.info("vector worker: %s", report)
            print(report)
        if not args.loop:
            return 0
        if not report["claimed"]:
            time.sleep(args.interval)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
