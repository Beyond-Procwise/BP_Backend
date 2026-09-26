"""Walk the harvested corpus and write a labelled set with per-field provenance.

Never drops an example. One that cannot be checked is kept and counted as
unverifiable, because a set that quietly shrinks to the checkable cases is how
coverage goes unnoticed -- and coverage is the number that stops accuracy being
gamed by making more fields uncheckable.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Optional

from src.services.truth.label import label_record

logger = logging.getLogger(__name__)

Recover = Callable[[str], Optional[str]]


def _default_recover(file_path: str) -> Optional[str]:
    """Fetch the document's text.

    Imported lazily so the builder stays importable, and testable, on a machine
    with no AWS credentials and no extraction stack installed.
    """
    if not file_path:
        return None
    try:
        from src.services.extraction import parser
        parsed = parser.parse(file_path)
        # ParsedDocument calls it full_text. Reading `text` returned None for
        # every document, so recovery recovered nothing while the S3 download
        # and the PDF conversion both quietly succeeded.
        return getattr(parsed, "full_text", None)
    except Exception as exc:  # noqa: BLE001 - recovery is best effort by design
        logger.info("could not recover source for %s: %s", file_path, exc)
        return None


def build(corpus_path: str, out_path: str, recover: Optional[Recover] = None) -> dict:
    recover = recover or _default_recover
    summary = {"examples": 0, "with_source": 0, "recovered": 0,
               "unrecoverable": 0, "malformed": 0,
               "counts": {"verified": 0, "unsupported": 0, "unverifiable": 0}}

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(corpus_path, encoding="utf-8") as fh, \
            open(out_path, "w", encoding="utf-8") as out:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                summary["malformed"] += 1
                logger.warning("%s:%d is not JSON: %s", corpus_path, lineno, exc)
                continue

            source = (row.get("source_text") or "").strip()
            origin = "corpus"
            if source:
                summary["with_source"] += 1
            else:
                source = (recover(row.get("file_path") or "") or "").strip()
                if source:
                    summary["recovered"] += 1
                    origin = "recovered"
                else:
                    summary["unrecoverable"] += 1
                    origin = "none"

            labelled = label_record(row.get("extracted") or {}, source or None)
            summary["examples"] += 1
            for key, value in labelled["counts"].items():
                summary["counts"][key] = summary["counts"].get(key, 0) + value

            out.write(json.dumps({
                "pk": row.get("pk"),
                "doc_type": row.get("doc_type"),
                "file_path": row.get("file_path"),
                "source": origin,
                "fields": labelled["fields"],
                "counts": labelled["counts"],
            }) + "\n")
    return summary
