"""Rebuild the vectors from their source rows: everything, one tenant, one collection, or a whole NEW embedding profile.

Postgres is the source of truth, so this can always recreate the vector data. By default it replaces each source's vectors in place (idempotent,
never a moment with no vectors for a source the live profile is serving). ``--clear`` wipes the target tenant's vectors first. ``--profile NAME``
targets another embedding profile's table, which is how a model switch is backfilled in parallel while the old profile keeps serving.

    python -m src.services.vector_store.rebuild                       # every tenant, every collection, active profile
    python -m src.services.vector_store.rebuild --tenant default      # one tenant
    python -m src.services.vector_store.rebuild --profile bge-m3@1    # backfill a new model's table (see docs/vector-search.md)
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Iterable, List, Optional

from . import outbox
from .loaders import DEFAULT_TENANT, default_loaders, GenericSourceLoader, source_collections
from .profiles import PROFILES, EmbeddingProfile
from .worker import VectorWorker

logger = logging.getLogger(__name__)


def _tenants(cur: Any, tenant: Optional[str]) -> List[str]:
    if tenant:
        return [tenant]
    cur.execute("SELECT DISTINCT tenant_id FROM proc.bp_vector_source")
    return sorted({DEFAULT_TENANT} | {r[0] for r in cur.fetchall()})


def rebuild(connect: Callable[[], Any], store_factory: Callable[[Optional[EmbeddingProfile]], Any],
            embedder_for: Callable[[Optional[EmbeddingProfile]], Any], *, tenant: Optional[str] = None,
            collections: Optional[Iterable[str]] = None, profile: Optional[EmbeddingProfile] = None, clear: bool = False,
            loaders: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Re-embed sources. Returns {tenants, sources, enqueued, done, failed, errors}. A source that fails is recorded and the rest continue."""
    loaders = dict(loaders or default_loaders())
    wanted = set(collections) if collections else None
    resolver = (lambda name: profile if profile is not None and name == profile.name else PROFILES.get(name))
    worker = VectorWorker(connect, embedder_for, loaders=loaders, store_factory=store_factory, profile_resolver=resolver)
    report: Dict[str, Any] = {"tenants": [], "sources": 0, "done": 0, "failed": 0, "errors": []}
    store = store_factory(profile)
    with connect() as conn, conn.cursor() as cur:
        for tenant_id in _tenants(cur, tenant):
            report["tenants"].append(tenant_id)
            names = set(loaders) | set(source_collections(cur, tenant_id))
            names = {n for n in names if wanted is None or n in wanted}
            if clear:
                if wanted is None:
                    store.clear(tenant_id)
                else:
                    for n in names:
                        store.clear(tenant_id, collection=n)
            for name in sorted(names):
                loader = loaders.get(name) or GenericSourceLoader(name)
                for source_id in list(loader.enumerate(cur, tenant_id)):
                    report["sources"] += 1
                    job = outbox.Job(0, tenant_id, name, source_id, "upsert", profile.name if profile else None, 1, 1)
                    try:
                        worker.process(cur, job)
                        report["done"] += 1
                    except Exception as exc:  # noqa: BLE001
                        report["failed"] += 1
                        if len(report["errors"]) < 5:
                            report["errors"].append(f"{tenant_id}/{name}/{source_id}: {type(exc).__name__}: {exc}")
    report["enqueued"] = report["sources"]
    return report


def main(argv: Optional[list] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Rebuild vectors from their source rows (Postgres is the source of truth).")
    ap.add_argument("--tenant", help="rebuild one tenant only (default: every tenant)")
    ap.add_argument("--collection", action="append", help="rebuild only this collection (repeatable)")
    ap.add_argument("--profile", help="target embedding profile (default: the active one); use to backfill a new model's table")
    ap.add_argument("--clear", action="store_true", help="wipe the target tenant's vectors first (leaves a gap while it rebuilds)")
    ap.add_argument("--no-clear", action="store_true", help="replace in place without wiping (the default)")
    args = ap.parse_args(argv)
    from src.services.db import get_conn
    from .embedder import SentenceTransformerEmbedder
    from .pgvector_store import PgVectorStore
    from .profiles import active_profile
    target = PROFILES[args.profile] if args.profile else None
    if args.profile and args.profile not in PROFILES:
        ap.error(f"unknown profile {args.profile!r}; known: {sorted(PROFILES)}")
    models: Dict[str, Any] = {}

    def embedder_for(p: Optional[EmbeddingProfile]):
        prof = p or active_profile()
        return models.setdefault(prof.name, SentenceTransformerEmbedder(prof.model))

    report = rebuild(get_conn, lambda p: PgVectorStore(get_conn, profile=p), embedder_for, tenant=args.tenant,
                     collections=args.collection, profile=target, clear=bool(args.clear and not args.no_clear))
    print(report)
    return 1 if report["failed"] else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
