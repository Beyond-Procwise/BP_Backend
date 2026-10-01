"""Runtime load of the document and relationship vocabulary.

proc.bp_concept and proc.bp_document_type are the source of truth; seed.py is
the fallback. Loading at runtime means a confirmed type takes effect without a
deploy, which is the whole reason the vocabulary became data.

Three properties this must have, each a way it goes wrong:

  * A failed or EMPTY load must never blank the vocabulary. A resolver that
    recognises nothing marks every document unknown, and those absences get
    recorded as though the documents stated nothing. A slightly stale
    vocabulary is enormously preferable to a confident wrong silence.
  * No query in the per-document path. The version probe is what keeps the
    common case (nothing changed) cheap.
  * Only status='active' loads, filtered in SQL. A 'proposed' concept is an
    observation awaiting a human; if it resolved, confirming it would be moot.

This mirrors src/services/facts/uom.py, deliberately: one loading pattern for
reference data is easier to reason about than two.
"""
from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

from .seed import CONCEPTS, DOCUMENT_TYPES, Concept, DocumentType

logger = logging.getLogger(__name__)

_WHITESPACE = re.compile(r"\s+")

_CONCEPT_SQL = """
    SELECT concept_code, domain, definition, not_to_be_confused_with,
           status, rejection_reason, recorded_at
      FROM proc.bp_concept
     WHERE status = 'active'
"""

_DOC_TYPE_SQL = """
    SELECT concept_code, role, default_parent_type, execution_mode,
           aliases, identifiers, structural_signals, pipeline_doc_type,
           status, recorded_at
      FROM proc.bp_document_type
     WHERE status = 'active'
"""

# The version probe. Two scalars over two small tables rather than the
# vocabulary itself: it runs far more often than a reload. count AND
# max(recorded_at), because neither alone is sufficient — editing a row without
# adding one leaves the count identical, and a row added while another is
# removed leaves it identical too.
_VERSION_SQL = """
    SELECT (SELECT count(*) FROM proc.bp_concept WHERE status = 'active'),
           (SELECT max(recorded_at) FROM proc.bp_concept WHERE status = 'active'),
           (SELECT count(*) FROM proc.bp_document_type WHERE status = 'active'),
           (SELECT max(recorded_at) FROM proc.bp_document_type WHERE status = 'active')
"""

_DEFAULT_TTL_SECONDS = 300.0
_DEFAULT_PROBE_SECONDS = 15.0


def fold(text: str) -> str:
    """The one normalisation used for every alias comparison.

    Lowercase, collapse internal whitespace, treat '_' and '-' as spaces so
    'purchase_order' and 'Purchase Order' are the same alias, and drop a
    trailing full stop.
    """
    t = (text or "").replace("_", " ").replace("-", " ")
    t = _WHITESPACE.sub(" ", t).strip().lower()
    return t[:-1] if t.endswith(".") else t


@dataclass(frozen=True)
class Vocabulary:
    """A resolved vocabulary and where it came from."""

    concepts: Mapping[str, Concept]
    document_types: Mapping[str, DocumentType]
    #: folded alias -> every active document-type concept_code claiming it.
    #: A tuple, not a str: more than one entry is a genuine collision and must
    #: stay representable rather than be silently narrowed to the first row.
    alias_index: Mapping[str, Tuple[str, ...]]
    source: str


def build_vocabulary(
    concept_rows: Iterable[Mapping[str, Any]],
    doc_type_rows: Iterable[Mapping[str, Any]],
    *,
    source: str,
) -> Vocabulary:
    """Assemble a Vocabulary from table rows. Pure — no I/O."""
    concepts: Dict[str, Concept] = {}
    for row in concept_rows:
        code = (row.get("concept_code") or "").strip()
        if not code or row.get("status") != "active":
            continue
        concepts[code] = Concept(
            concept_code=code,
            domain=str(row.get("domain") or ""),
            definition=row.get("definition") or "",
            not_to_be_confused_with=tuple(row.get("not_to_be_confused_with") or ()),
            status="active",
            rejection_reason=row.get("rejection_reason"),
        )

    document_types: Dict[str, DocumentType] = {}
    alias_index: Dict[str, Tuple[str, ...]] = {}
    for row in doc_type_rows:
        code = (row.get("concept_code") or "").strip()
        if not code or row.get("status") != "active":
            continue
        aliases = tuple(row.get("aliases") or ())
        identifiers = tuple(row.get("identifiers") or ())
        document_types[code] = DocumentType(
            concept_code=code,
            role=str(row.get("role") or ""),
            default_parent_type=row.get("default_parent_type"),
            execution_mode=row.get("execution_mode"),
            aliases=aliases,
            identifiers=identifiers,
            structural_signals=tuple(row.get("structural_signals") or ()),
            pipeline_doc_type=row.get("pipeline_doc_type"),
            status="active",
        )
        # The concept's own name is always an alias of itself.
        for alias in (*aliases, code.split(".", 1)[-1]):
            key = fold(alias)
            if not key:
                continue
            existing = alias_index.get(key, ())
            if code not in existing:
                alias_index[key] = (*existing, code)

    for key, owners in alias_index.items():
        if len(owners) > 1:
            logger.warning(
                "alias %r is claimed by %d concepts (%s) — it can only resolve "
                "to UNRESOLVED until a ruling settles it",
                key, len(owners), ", ".join(owners),
            )

    return Vocabulary(
        concepts=concepts,
        document_types=document_types,
        alias_index=alias_index,
        source=source,
    )


SEED_VOCABULARY = build_vocabulary(
    [
        {
            "concept_code": c.concept_code,
            "domain": c.domain,
            "definition": c.definition,
            "not_to_be_confused_with": list(c.not_to_be_confused_with),
            "status": c.status,
            "rejection_reason": c.rejection_reason,
        }
        for c in CONCEPTS.values()
    ],
    [
        {
            "concept_code": d.concept_code,
            "role": d.role,
            "default_parent_type": d.default_parent_type,
            "execution_mode": d.execution_mode,
            "aliases": list(d.aliases),
            "identifiers": list(d.identifiers),
            "structural_signals": list(d.structural_signals),
            "pipeline_doc_type": d.pipeline_doc_type,
            "status": d.status,
        }
        for d in DOCUMENT_TYPES.values()
    ],
    source="builtin-seed",
)

_lock = threading.Lock()
_active: Vocabulary = SEED_VOCABULARY
_loaded_at: Optional[float] = None
_probed_at: float = 0.0
_version: Optional[Tuple[Any, ...]] = None
_invalidated: bool = False


def resolve_alias(text: str, vocabulary: Vocabulary) -> Tuple[str, ...]:
    """Every active document-type concept_code claiming ``text`` as an alias.

    Empty when nothing claims it — which is an answer ("unknown"), not a
    failure. More than one is a genuine collision and the caller must treat it
    as UNRESOLVED rather than choosing.
    """
    return vocabulary.alias_index.get(fold(text), ())


def invalidate() -> None:
    """Force the next ``ensure_vocabulary`` to re-read, ignoring the TTL."""
    global _invalidated
    with _lock:
        _invalidated = True


def _fetch_rows() -> Tuple[list, list]:
    """Read both tables. Separated so tests can replace it."""
    from src.services.db import get_conn

    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(_CONCEPT_SQL)
        cols = [d[0] for d in (cur.description or [])]
        concept_rows = [dict(zip(cols, r)) for r in cur.fetchall()]
        cur.execute(_DOC_TYPE_SQL)
        cols = [d[0] for d in (cur.description or [])]
        doc_type_rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    return concept_rows, doc_type_rows


def _fetch_version() -> Optional[Tuple[Any, ...]]:
    from src.services.db import get_conn

    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(_VERSION_SQL)
        row = cur.fetchone()
    return tuple(row) if row else None


def ensure_vocabulary(
    *,
    ttl_seconds: float = _DEFAULT_TTL_SECONDS,
    probe_seconds: float = _DEFAULT_PROBE_SECONDS,
) -> Vocabulary:
    """The current vocabulary, reloading when it is stale or has changed.

    Never raises and never returns an empty vocabulary.
    """
    global _active, _loaded_at, _probed_at, _version, _invalidated

    now = time.monotonic()
    with _lock:
        fresh_enough = (
            _loaded_at is not None
            and not _invalidated
            and (now - _loaded_at) < ttl_seconds
        )
        due_a_probe = _loaded_at is not None and (now - _probed_at) >= probe_seconds
        current = _active

    if fresh_enough and not due_a_probe:
        return current

    if fresh_enough and due_a_probe:
        # Still inside the TTL: ask the cheap question, and only then the
        # expensive one.
        try:
            version = _fetch_version()
        except Exception:
            logger.debug("vocabulary version probe failed; keeping current", exc_info=True)
            with _lock:
                _probed_at = now
            return current
        with _lock:
            _probed_at = now
            unchanged = version == _version
        if unchanged:
            return current

    try:
        concept_rows, doc_type_rows = _fetch_rows()
    except Exception:
        logger.exception(
            "could not read proc.bp_concept / proc.bp_document_type; continuing "
            "with the %s vocabulary", current.source,
        )
        with _lock:
            _probed_at = now
            _invalidated = False
        return current

    candidate = build_vocabulary(
        concept_rows, doc_type_rows,
        source=f"bp_concept@{len(concept_rows)}+bp_document_type@{len(doc_type_rows)}",
    )
    if not candidate.document_types:
        logger.error(
            "proc.bp_document_type yielded no active types from %d row(s); "
            "keeping the %s vocabulary rather than recognising nothing",
            len(doc_type_rows), current.source,
        )
        with _lock:
            _probed_at = now
            _invalidated = False
        return current

    try:
        version = _fetch_version()
    except Exception:
        version = None

    with _lock:
        _active = candidate
        _loaded_at = now
        _probed_at = now
        _version = version
        _invalidated = False
    logger.info(
        "loaded %d active concepts and %d active document types",
        len(candidate.concepts), len(candidate.document_types),
    )
    return candidate


__all__ = [
    "Vocabulary", "SEED_VOCABULARY", "build_vocabulary", "ensure_vocabulary",
    "invalidate", "resolve_alias", "fold",
]
