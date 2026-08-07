"""Explicit lifecycle states and the only legal moves between them.

This module is where "fail closed" stops being a policy and becomes a
mechanism. READY_FOR_EXTRACTION is reachable from exactly one predecessor,
STORED, which is itself reachable only from GATED. There is therefore no
sequence of legal transitions that puts bytes in front of the extractor
without passing the safety gate — a caller that tries raises rather than
proceeding.

Spec §5.4. Note the deliberate rename: the spec's shorthand writes the
artifact chain as "extracted -> ... -> extracted", using one word for being
lifted out of the MIME tree and for extraction having run. Here those are
DETACHED and EXTRACTION_COMPLETE.
"""
from __future__ import annotations

from enum import Enum
from typing import Dict, FrozenSet


class IllegalTransition(RuntimeError):
    """A state move that the lifecycle does not permit.

    Deliberately not a warning and never swallowed. Reaching this means a
    caller tried to skip a stage, which is the exact failure the gate exists
    to prevent.
    """


class BundleState(str, Enum):
    RECEIVED = "received"
    PARSED = "parsed"
    AUTHENTICATED = "authenticated"
    BOUND = "bound"
    COMPLETE = "complete"
    QUARANTINED = "quarantined"


class ArtifactState(str, Enum):
    DETACHED = "detached"
    TYPED = "typed"
    GATED = "gated"
    STORED = "stored"
    READY_FOR_EXTRACTION = "ready_for_extraction"
    EXTRACTION_COMPLETE = "extraction_complete"
    QUARANTINED = "quarantined"
    DUPLICATE = "duplicate"


class QuarantineReason(str, Enum):
    MALFORMED_MIME = "malformed_mime"
    UNKNOWN_TYPE = "unknown_type"
    ENCRYPTED = "encrypted"
    MACRO = "macro"
    ARCHIVE = "archive"
    EXECUTABLE = "executable"
    OVERSIZE = "oversize"
    TYPE_MISMATCH = "type_mismatch"
    HIDDEN_TEXT = "hidden_text"
    UNRESOLVABLE_REFERENCE = "unresolvable_reference"


_BUNDLE_MOVES: Dict[BundleState, FrozenSet[BundleState]] = {
    BundleState.RECEIVED: frozenset({BundleState.PARSED, BundleState.QUARANTINED}),
    BundleState.PARSED: frozenset({BundleState.AUTHENTICATED, BundleState.QUARANTINED}),
    BundleState.AUTHENTICATED: frozenset({BundleState.BOUND, BundleState.QUARANTINED}),
    BundleState.BOUND: frozenset({BundleState.COMPLETE, BundleState.QUARANTINED}),
    BundleState.COMPLETE: frozenset(),
    BundleState.QUARANTINED: frozenset(),
}

_ARTIFACT_MOVES: Dict[ArtifactState, FrozenSet[ArtifactState]] = {
    ArtifactState.DETACHED: frozenset(
        {ArtifactState.TYPED, ArtifactState.QUARANTINED}
    ),
    ArtifactState.TYPED: frozenset(
        {ArtifactState.GATED, ArtifactState.QUARANTINED}
    ),
    ArtifactState.GATED: frozenset(
        {ArtifactState.STORED, ArtifactState.QUARANTINED}
    ),
    ArtifactState.STORED: frozenset(
        {
            ArtifactState.READY_FOR_EXTRACTION,
            ArtifactState.DUPLICATE,
            ArtifactState.QUARANTINED,
        }
    ),
    ArtifactState.READY_FOR_EXTRACTION: frozenset(
        {ArtifactState.EXTRACTION_COMPLETE}
    ),
    ArtifactState.EXTRACTION_COMPLETE: frozenset(),
    ArtifactState.QUARANTINED: frozenset(),
    ArtifactState.DUPLICATE: frozenset(),
}


def assert_transition(current, target) -> None:
    """Raise IllegalTransition unless current -> target is a permitted move."""
    if isinstance(current, BundleState):
        allowed = _BUNDLE_MOVES.get(current, frozenset())
    elif isinstance(current, ArtifactState):
        allowed = _ARTIFACT_MOVES.get(current, frozenset())
    else:
        raise IllegalTransition(f"unknown state type: {type(current).__name__}")
    if type(target) is not type(current):
        raise IllegalTransition(
            f"cannot move {type(current).__name__} to {type(target).__name__}"
        )
    if target not in allowed:
        raise IllegalTransition(f"{current.value} -> {target.value} is not permitted")
