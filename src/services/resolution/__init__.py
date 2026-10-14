"""Global resolution layer.

The pairwise scorer says how well two documents match. This layer decides which
of those matches can hold at once, and reports how forced each one was.
"""
from .contracts import (  # noqa: F401
    CandidateEdge,
    CardinalityRule,
    ResolutionRequest,
    ResolutionResult,
    ResolvedLink,
    ResourceCapacity,
)
from .resolve import resolve  # noqa: F401

__all__ = [
    "CandidateEdge",
    "CardinalityRule",
    "ResolutionRequest",
    "ResolutionResult",
    "ResolvedLink",
    "ResourceCapacity",
    "resolve",
]
