"""Embed every already-extracted document into the document vector index.

The live extraction path only started writing embeddings on 2026-10-08, so
documents extracted before then are in the database and missing from the
index. This re-embeds the newest _raw row of each document that stored its
text. It is idempotent: a document's old points are replaced, never duplicated.

Dry run by default (counts only). Pass --apply to write.

    ./.venv/bin/python scripts/embed_backfill.py            # what would be embedded
    ./.venv/bin/python scripts/embed_backfill.py --apply    # embed it
"""
from __future__ import annotations

import argparse
import os
import sys
from types import SimpleNamespace

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [ROOT, os.path.join(ROOT, "src")]

from config.settings import settings  # noqa: E402
from src.services import egress  # noqa: E402
from src.services.db import get_conn  # noqa: E402
from src.services.extraction.embed import embed_document, latest_raw_rows  # noqa: E402
from src.services.extraction.persistence import _RAW_TABLES  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--apply", action="store_true", help="write embeddings (default: dry run)")
    ap.add_argument("--device", default="cpu",
                    help="embedder device; cpu by default so it never competes with the served model")
    args = ap.parse_args()

    with get_conn() as conn:
        cur = conn.cursor()
        todo = {t: latest_raw_rows(cur, t) for t in _RAW_TABLES}
        for t, rows in todo.items():
            print(f"{t:15} {len(rows)} document(s)")
        if not args.apply:
            print("dry run: pass --apply to embed")
            return 0

        from sentence_transformers import SentenceTransformer

        agent = SimpleNamespace(
            settings=settings,
            embedding_model=SentenceTransformer(settings.embedding_model, device=args.device),
            qdrant_client=egress.vector_client(
                purpose=egress.Purpose.VECTOR_INDEX, url=settings.qdrant_url,
                api_key=(settings.qdrant_api_key or "").strip() or None),
        )
        written, failed = 0, []
        for t, rows in todo.items():
            for doc_pk, raw_id in rows:
                try:
                    written += embed_document(agent, cur, doc_type=t, doc_pk=doc_pk, raw_id=raw_id)
                except Exception as exc:  # noqa: BLE001
                    failed.append((t, doc_pk, str(exc)[:120]))
        print(f"points written: {written}")
        for f in failed:
            print("FAILED", *f)
        return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
