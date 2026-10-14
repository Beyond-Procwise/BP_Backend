"""Input and output contracts for the global resolution layer.

The pairwise scorer (``src/services/linking_engine``) answers "how well does this
document match that one?". It has no view of the set, so two invoices can each
independently claim the same purchase-order line and nothing notices.

This layer takes the scorer's pairwise verdicts as *candidate edges* and decides
which of them can hold at the same time. Nothing here reads a document, calls a
model, or re-scores anything: ``log_odds`` arrives already computed and leaves
untouched.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Mapping, Optional

Shape = Literal["1:1", "N:1", "1:N", "N:M"]
Status = Literal["RESOLVED", "INFEASIBLE", "DEGENERATE"]


@dataclass(frozen=True)
class CandidateEdge:
    """One pairwise verdict from the scorer, plus what taking it would consume.

    ``consumes`` maps a resource id to the quantity this link would draw from it,
    e.g. ``{"po_line:PO-100:3": 1400.00}``. It is what generalises this layer
    beyond invoice-to-PO: a purchase-order line consumes value, a clause slot
    consumes exactly one mapping, an amendment consumes a position in a sequence.
    An empty mapping means the link competes only on cardinality.
    """

    source_id: str
    target_id: str
    log_odds: float           # from the existing scorer, pre-sigmoid
    confidence: float         # post-sigmoid, for reporting only
    profile_id: str           # relationship profile that scored this pair
    consumes: Mapping[str, float] = field(default_factory=dict)

    @property
    def key(self) -> tuple[str, str]:
        """Canonical identity, and the documented tie-break order."""
        return (self.source_id, self.target_id)


@dataclass(frozen=True)
class ResourceCapacity:
    """A finite thing edges draw down. ``tolerance`` is ABSOLUTE, in the same
    unit as ``capacity`` — see DECISIONS.md, the existing engine's tolerances are
    relative and callers must convert."""

    resource_id: str
    capacity: float
    tolerance: float = 0.0

    @property
    def bound(self) -> float:
        return self.capacity + self.tolerance


@dataclass(frozen=True)
class CardinalityRule:
    """How many links a profile permits at each end. ``None`` means unbounded."""

    profile_id: str
    shape: Shape
    max_sources_per_target: Optional[int] = None
    max_targets_per_source: Optional[int] = None


@dataclass(frozen=True)
class ResolutionRequest:
    request_id: str
    edges: tuple[CandidateEdge, ...]
    capacities: tuple[ResourceCapacity, ...]
    rules: tuple[CardinalityRule, ...]
    profile_registry_version: str


@dataclass(frozen=True)
class ResolvedLink:
    source_id: str
    target_id: str
    log_odds: float                 # unchanged, passed through
    margin: float                   # objective delta vs. the best solution without this link
    margin_normalised: float        # margin / |objective|, in [0, 1]
    displaced_by: tuple[str, ...]   # targets the source would take if this link were forbidden


@dataclass(frozen=True)
class ResolutionResult:
    request_id: str
    status: Status
    links: tuple[ResolvedLink, ...]
    unassigned_sources: tuple[str, ...]
    objective: float
    infeasibility_certificate: Optional[tuple[str, ...]]
    inputs_hash: str
    solver_version: str
