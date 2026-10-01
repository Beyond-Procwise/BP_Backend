"""The reference-data checks from the build spec §8.1.

Each returns a list of Violation. An empty list is a pass. They take a
Vocabulary rather than a connection so the same function serves CI, a live
check and a unit test.

Two of the spec's six checks are deliberately absent, and the absence is the
honest state rather than an omission:

  * "conflict_rulings matches what is generated from its master" has no
    subject. The Discovery Report (§7.3) recommends one table holding a
    collision and its ruling together, so there is no mirror to drift.
  * "every R-REL rule has a severity and every SOFT rule a penalty" has no
    subject until R-REL exists.

Either, written now, would be a check that passes over an empty set — which
reports success while checking nothing.

One check here is NOT from the spec: check_concepts_exist_for_every_document_type.
The spec did not foresee that `status` would end up on both tables for the same
type, and a half-promoted type resolved and routed live uploads while every
spec check passed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .vocabulary import Vocabulary, fold


@dataclass(frozen=True)
class Violation:
    check: str
    subject: str
    detail: str


def check_aliases_are_unambiguous(vocabulary: Vocabulary) -> List[Violation]:
    """No alias may be claimed by more than one active document type.

    The spec asks that such an alias carry a Conflict_Register entry with a
    ruling. There is no conflict register, so the only honest state is that no
    seeded alias collides — a collision the resolver cannot settle can only
    produce UNRESOLVED.
    """
    out: List[Violation] = []
    for alias, owners in sorted(vocabulary.alias_index.items()):
        if len(owners) > 1:
            out.append(Violation(
                "aliases_are_unambiguous", alias,
                f"claimed by {len(owners)} concepts: {', '.join(sorted(owners))} "
                "— no ruling exists to settle it",
            ))
    return out


def check_every_reference_resolves(vocabulary: Vocabulary) -> List[Violation]:
    """Every role, parent type, execution mode and not_to_be_confused_with
    entry must name a concept that exists, and every DOCUMENT_TYPE concept must
    have an attributes row."""
    out: List[Violation] = []
    known = set(vocabulary.concepts)
    by_domain = {}
    for code, concept in vocabulary.concepts.items():
        by_domain.setdefault(concept.domain, set()).add(code)

    for code, concept in sorted(vocabulary.concepts.items()):
        for other in concept.not_to_be_confused_with:
            if other not in known:
                out.append(Violation(
                    "every_reference_resolves", code,
                    f"not_to_be_confused_with names {other!r}, which is not a concept",
                ))

    for code in sorted(by_domain.get("DOCUMENT_TYPE", set())):
        if code not in vocabulary.document_types:
            out.append(Violation(
                "every_reference_resolves", code,
                "DOCUMENT_TYPE concept has no bp_document_type row, so it has "
                "no role and cannot take part in a relationship",
            ))

    for code, dt in sorted(vocabulary.document_types.items()):
        if dt.role not in by_domain.get("RELATIONSHIP_ROLE", set()):
            out.append(Violation(
                "every_reference_resolves", code,
                f"role {dt.role!r} is not a RELATIONSHIP_ROLE concept",
            ))
        if dt.default_parent_type and dt.default_parent_type not in by_domain.get("DOCUMENT_TYPE", set()):
            out.append(Violation(
                "every_reference_resolves", code,
                f"default_parent_type {dt.default_parent_type!r} is not a "
                "DOCUMENT_TYPE concept",
            ))
        if dt.execution_mode and dt.execution_mode not in by_domain.get("EXECUTION_MODE", set()):
            out.append(Violation(
                "every_reference_resolves", code,
                f"execution_mode {dt.execution_mode!r} is not an EXECUTION_MODE concept",
            ))
    return out


def check_concepts_exist_for_every_document_type(
    vocabulary: Vocabulary,
) -> List[Violation]:
    """Every loaded document type must have a loaded concept.

    `status` lives on proc.bp_concept AND proc.bp_document_type for the same
    type — two columns for one fact — so a person can promote one half. The
    half that matters is the type: with bp_document_type.status='active' and
    bp_concept.status='proposed' the type resolved aliases and routed live
    uploads while its concept (its definition, its domain, its
    not_to_be_confused_with list) was absent, and every other check here passed.

    build_vocabulary now drops such a type, so on a built Vocabulary this check
    can only fire if that skip is removed — which is precisely what it is for.
    check_every_reference_resolves catches the OTHER direction (an active
    concept with no type row); this one catches the dangerous direction.
    """
    return [
        Violation("concepts_exist_for_every_document_type", code,
                  "bp_document_type row is loaded but its concept is not — the "
                  "type would resolve and route with no definition behind it; "
                  "promote proc.bp_concept.status too, or demote the type")
        for code in sorted(vocabulary.document_types)
        if code not in vocabulary.concepts
    ]


def check_active_concepts_are_defined(vocabulary: Vocabulary) -> List[Violation]:
    """An active concept with no definition is a name with no meaning behind
    it, and the next reader will supply their own."""
    return [
        Violation("active_concepts_are_defined", code,
                  "active concept has no definition")
        for code, concept in sorted(vocabulary.concepts.items())
        if not (concept.definition or "").strip()
    ]


#: The four physical table families the extraction pipeline actually has.
_PIPELINES = frozenset({"invoice", "purchase_order", "quote", "contract"})


def check_pipeline_targets_exist(vocabulary: Vocabulary) -> List[Violation]:
    """pipeline_doc_type must name a pipeline that exists, or be absent.

    A fifth value would route a document at _raw/_stg/_trgt tables that are not
    there, and the failure would surface as a SQL error mid-extraction.
    """
    return [
        Violation("pipeline_targets_exist", code,
                  f"pipeline_doc_type {dt.pipeline_doc_type!r} names no physical pipeline")
        for code, dt in sorted(vocabulary.document_types.items())
        if dt.pipeline_doc_type is not None and dt.pipeline_doc_type not in _PIPELINES
    ]


def run_all(vocabulary: Vocabulary) -> List[Violation]:
    """Every check, in a stable order."""
    out: List[Violation] = []
    out.extend(check_aliases_are_unambiguous(vocabulary))
    out.extend(check_every_reference_resolves(vocabulary))
    out.extend(check_concepts_exist_for_every_document_type(vocabulary))
    out.extend(check_active_concepts_are_defined(vocabulary))
    out.extend(check_pipeline_targets_exist(vocabulary))
    return out


__all__ = [
    "Violation", "run_all",
    "check_aliases_are_unambiguous", "check_every_reference_resolves",
    "check_concepts_exist_for_every_document_type",
    "check_active_concepts_are_defined", "check_pipeline_targets_exist",
]
