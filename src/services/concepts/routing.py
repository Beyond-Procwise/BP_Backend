"""Turn the category the uploader typed into a physical pipeline.

This replaces the four-entry dict in process_monitor_watcher.py:523. Two things
change and one deliberately does not.

Changes: the set of acceptable spellings now comes from proc.bp_document_type,
so adding a document type is a row rather than a code edit; and a refusal says
what it was given and why.

Does NOT change: an unrecognised category still RAISES. Today that surfaces as
a hard error a human sees, and a permissive gate would lose that signal — see
the Discovery Report §6.4. "Unknown" becomes expressible on the classification
side (type_resolver), where it costs nothing; it does not become a licence to
guess a pipeline.
"""
from __future__ import annotations

from typing import Optional, Tuple

from .vocabulary import Vocabulary, ensure_vocabulary, resolve_alias


class UnknownDocumentCategory(RuntimeError):
    """Nothing in the vocabulary claims this category, or it has no pipeline."""


class AmbiguousDocumentCategory(RuntimeError):
    """More than one active document type claims this category as an alias.

    Raised rather than resolved: picking one would make the answer depend on
    row order, and the pipeline it chose would be recorded as a fact.
    """


def pipeline_for_category(
    category: Optional[str],
    *,
    vocabulary: Optional[Vocabulary] = None,
) -> Tuple[str, str]:
    """``(pipeline_doc_type, concept_code)`` for an uploader's category label."""
    vocab = vocabulary if vocabulary is not None else ensure_vocabulary()
    raw = (category or "").strip()
    if not raw:
        raise UnknownDocumentCategory(
            "no document category was supplied; refusing to guess a type for an "
            "unlabelled document"
        )

    candidates = resolve_alias(raw, vocab)
    if not candidates:
        raise UnknownDocumentCategory(
            f"no document type in proc.bp_document_type claims {raw!r} as an "
            "alias (only status='active' rows resolve)"
        )
    if len(candidates) > 1:
        raise AmbiguousDocumentCategory(
            f"{raw!r} is claimed by {len(candidates)} document types "
            f"({', '.join(sorted(candidates))}); no ruling exists to settle it"
        )

    concept_code = candidates[0]
    pipeline = vocab.document_types[concept_code].pipeline_doc_type
    if not pipeline:
        raise UnknownDocumentCategory(
            f"{raw!r} resolves to {concept_code} which has no pipeline "
            "(pipeline_doc_type is NULL): the type is recognised but nothing "
            "can ingest it yet"
        )
    return pipeline, concept_code


__all__ = [
    "pipeline_for_category",
    "UnknownDocumentCategory",
    "AmbiguousDocumentCategory",
]
