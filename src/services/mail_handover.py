"""Hand cleared email attachments to the upload queue, and record what became of them.

The seam between the two planes. Mail-intake (src/services/mail_intake/) ends
with an artifact in READY_FOR_EXTRACTION: safety-gated, stored, de-duplicated.
Plan 1 stopped there. Nothing consumed the queue, so an emailed quote was
stored and never read.

An emailed document enters extraction the way an upload does: its bytes are
written under documents/<doc type>/ and a proc.process_monitor row is queued
for the ProcessMonitorWatcher. That is deliberate. The watcher is where content
de-duplication against uploads, status, session outcomes and deal linking live;
a side door straight into dispatch_document would skip all of them, and an
emailed quote would never appear anywhere a person looks.

What the watcher is handed is a key THIS module built from the artifact id and
the detected type, under the uploads prefix. The sender controls none of it
(the filename contributes only a sanitised stem). handoff.py's rule — never
give extraction a key that reaches the quarantine prefix — holds: the copy is
made only from an artifact that reached READY_FOR_EXTRACTION, which the state
machine makes reachable only through the safety gate.

The document type is read from the document's own words (type_resolver, the
same deterministic reader dispatch uses). An upload's type comes from the
person uploading it; an email has no such person, and routing.pipeline_for_category
refuses to guess. So a document whose page does not settle its type is HELD:
a derivation row records why, the artifact stays READY_FOR_EXTRACTION, and it
is logged at WARNING. It is never routed on a guess.

Tracking: one bp_artifact_derivation row per hand-over, extraction_run_ref
'process_monitor:<id>'. settle() closes it when that row leaves the pending
statuses, and only then does the artifact move to EXTRACTION_COMPLETE.
"""
from __future__ import annotations

import logging
import os
import re
import tempfile
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from src.services.mail_intake import repo
from src.services.mail_intake.handoff import suffix_for
from src.services.mail_intake.states import ArtifactState

log = logging.getLogger(__name__)

PIPELINE_VERSION = "mail-handover-1"
RUN_REF_PREFIX = "process_monitor:"
# The watcher's own definition of "not finished" (ProcessMonitorWatcher._EXTRACTION_PENDING).
_PENDING = ("Completed", "Running", "Extracting")
_STEM_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _row(row: Any, names: Tuple[str, ...]) -> Dict[str, Any]:
    return dict(row) if isinstance(row, Mapping) else dict(zip(names, row))


def safe_stem(filename: Optional[str], limit: int = 80) -> str:
    # Every character outside [A-Za-z0-9._-] becomes "_", "/" included, so the
    # sender's name cannot leave the mail-intake/ prefix it is placed under.
    stem = os.path.splitext(filename or "")[0]
    stem = _STEM_UNSAFE.sub("_", stem).strip("._-")
    return stem[:limit] or "attachment"


def category_for(concept_code: str, vocab: Any,
                 route: Optional[Callable[..., Tuple[str, str]]] = None) -> Optional[str]:
    """A category label the watcher will route back to exactly this concept, or None."""
    if route is None:
        from src.services.concepts.routing import pipeline_for_category as route
    dt = vocab.document_types.get(concept_code)
    if dt is None or not dt.pipeline_doc_type:
        return None
    for label in (dt.pipeline_doc_type, *sorted(dt.aliases), concept_code.split(".", 1)[-1]):
        try:
            if route(label, vocabulary=vocab)[1] == concept_code:
                return label
        except Exception:
            continue
    return None


def read_type(content: bytes, suffix: str, *, parse: Optional[Callable] = None,
              resolve: Optional[Callable] = None, vocab: Any = None):
    """Classify a document from its own text. Returns a TypeResolution."""
    if parse is None:
        from src.services.extraction.parser import parse
    if resolve is None:
        from src.services.extraction.type_resolver import resolve_document_type as resolve
    handle, path = tempfile.mkstemp(prefix="mailhandover-", suffix=suffix)
    try:
        os.fchmod(handle, 0o600)
        with os.fdopen(handle, "wb") as fh:
            fh.write(content)
        parsed = parse(path)
    finally:
        try:
            os.unlink(path)
        except OSError:
            log.warning("mail_handover could not remove %s", path)
    return resolve(declared_concept=None, full_text=parsed.full_text, vocabulary=vocab)


def _pending_artifacts(cur: Any, limit: int) -> List[Dict[str, Any]]:
    """READY artifacts never handed over and never held. One attempt each."""
    names = ("artifact_id", "object_key", "detected_mime", "declared_filename")
    cur.execute(
        """
        SELECT a.artifact_id, a.object_key, a.detected_mime, a.declared_filename
          FROM proc.bp_ingest_artifact a
         WHERE a.state = %s
           AND NOT EXISTS (SELECT 1 FROM proc.bp_artifact_derivation d
                            WHERE d.artifact_id = a.artifact_id)
         ORDER BY a.created_at
         LIMIT %s
        """,
        (ArtifactState.READY_FOR_EXTRACTION.value, limit),
    )
    return [_row(r, names) for r in cur.fetchall() or ()]


def _hold(cur: Any, artifact_id: str, why: str) -> None:
    cur.execute(
        """
        INSERT INTO proc.bp_artifact_derivation
            (artifact_id, pipeline_version, extraction_run_ref, completed_at, outcome)
        VALUES (%s, %s, NULL, now(), %s)
        """,
        (artifact_id, PIPELINE_VERSION, f"held: {why}"[:500]),
    )
    log.warning("mail_handover.held artifact=%s reason=%s", artifact_id, why)


def hand_over(conn: Any, store: Any, *, limit: int = 20,
              read: Callable[..., Any] = read_type,
              vocab_loader: Optional[Callable[[], Any]] = None,
              route: Optional[Callable[..., Tuple[str, str]]] = None) -> Dict[str, int]:
    """Queue every newly READY artifact for extraction, or hold it. One transaction each."""
    if getattr(conn, "autocommit", False):
        raise RuntimeError("mail_handover needs a non-autocommit connection")
    if vocab_loader is None:
        from src.services.concepts.vocabulary import ensure_vocabulary as vocab_loader
    vocab = vocab_loader()

    with conn.cursor() as cur:
        pending = _pending_artifacts(cur, limit)
    conn.commit()

    counts = {"queued": 0, "held": 0, "failed": 0}
    for art in pending:
        artifact_id = str(art["artifact_id"])
        try:
            content = store.get(art["object_key"])
            if content is None:
                raise RuntimeError(f"stored bytes missing at {art['object_key']}")
            suffix = suffix_for(art["detected_mime"], art["declared_filename"])
            resolution = read(content, suffix, vocab=vocab)
            concept = resolution.evidence_concept if resolution.status == "matched" else None
            category = category_for(concept, vocab, route) if concept else None
            with conn.cursor() as cur:
                if category is None:
                    why = (f"document type not settled by its own text "
                           f"(status={resolution.status}, "
                           f"candidates={','.join(resolution.candidates) or 'none'})"
                           if concept is None else
                           f"{concept} has no upload category that routes back to it")
                    _hold(cur, artifact_id, why)
                    counts["held"] += 1
                else:
                    pipeline = vocab.document_types[concept].pipeline_doc_type
                    key = (f"documents/{pipeline}/mail-intake/"
                           f"{artifact_id}-{safe_stem(art['declared_filename'])}{suffix}")
                    store.put_immutable(key, content)
                    cur.execute(
                        """
                        INSERT INTO proc.process_monitor
                            (process_name, type, status, file_path, category,
                             document_type, created_date, lastmodified_date, created_by)
                        VALUES ('mail-intake', 'Email', 'Completed', %s, %s, %s,
                                NOW(), NOW(), %s)
                        RETURNING id
                        """,
                        (key, category, suffix.lstrip("."), f"mail-intake:{artifact_id}"),
                    )
                    pm_row = cur.fetchone()
                    pm_id = pm_row["id"] if isinstance(pm_row, Mapping) else pm_row[0]
                    repo.append_derivation(cur, artifact_id, PIPELINE_VERSION,
                                           run_ref=f"{RUN_REF_PREFIX}{pm_id}")
                    counts["queued"] += 1
                    log.info("mail_handover.queued artifact=%s process_monitor=%s category=%s",
                             artifact_id, pm_id, category)
            conn.commit()
        except Exception:
            # Nothing was recorded, so the next pass tries this artifact again.
            conn.rollback()
            counts["failed"] += 1
            log.exception("mail_handover failed artifact=%s", artifact_id)
    return counts


def settle(conn: Any) -> Dict[str, int]:
    """Close hand-overs whose process_monitor row has finished; mark the artifact complete."""
    if getattr(conn, "autocommit", False):
        raise RuntimeError("mail_handover needs a non-autocommit connection")
    names = ("derivation_id", "artifact_id", "status", "doc_action")
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT d.derivation_id, d.artifact_id, pm.status, pm.doc_action
              FROM proc.bp_artifact_derivation d
              JOIN proc.process_monitor pm
                ON pm.id = substring(d.extraction_run_ref FROM %s)::int
             WHERE d.extraction_run_ref LIKE %s
               AND d.completed_at IS NULL
               AND pm.status <> ALL(%s)
            """,
            (f"^{RUN_REF_PREFIX}(\\d+)$", f"{RUN_REF_PREFIX}%", list(_PENDING)),
        )
        done = [_row(r, names) for r in cur.fetchall() or ()]
        for row in done:
            outcome = row["status"] + (f" ({row['doc_action']})" if row["doc_action"] else "")
            cur.execute(
                "UPDATE proc.bp_artifact_derivation SET completed_at = now(), outcome = %s "
                "WHERE derivation_id = %s AND completed_at IS NULL",
                (outcome, row["derivation_id"]),
            )
            repo.transition_artifact(cur, str(row["artifact_id"]),
                                     ArtifactState.READY_FOR_EXTRACTION,
                                     ArtifactState.EXTRACTION_COMPLETE)
    conn.commit()
    return {"settled": len(done)}
