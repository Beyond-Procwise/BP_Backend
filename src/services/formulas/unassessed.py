"""The two things a formula result can carry besides a number.

``UNASSESSED`` is the answer when a formula could not be evaluated. It is
deliberately hostile to being mistaken for a value: truth-testing it raises,
and so does arithmetic on it. That is the whole point. A ``0.0`` returned for
"we could not work this out" is indistinguishable from a measured zero, and
this codebase has already paid for that confusion more than once --- see
``supplier_ranking_agent`` (a supplier whose payment terms we had never read
was ranked as though they had offered the worst terms on the table) and
``linking_engine`` (a completeness figure used as an evidence proxy made the
promotion gate unreachable by a perfect match).

``Confidence`` is a three-level ladder, not a probability. A number derived
from other numbers can never be more trustworthy than its least trustworthy
input, so the ladder exists to be taken a minimum over.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Iterable


class UnassessedError(TypeError):
    """Raised when UNASSESSED is used as if it were a value."""


class _Unassessed:
    """The singleton "no answer" result.

    Not an enum member and not ``None``: both of those are falsy, and falsy is
    exactly the property that lets ``result or 0`` quietly turn "unknown" into
    "zero". Every implicit use raises instead.
    """

    __slots__ = ()
    _instance: "_Unassessed | None" = None

    def __new__(cls) -> "_Unassessed":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "UNASSESSED"

    def __str__(self) -> str:
        return "UNASSESSED"

    def __bool__(self) -> bool:
        raise UnassessedError(
            "UNASSESSED has no truth value. A formula could not be evaluated; "
            "handle that case explicitly rather than treating it as false. "
            "Use `x is UNASSESSED` to test for it."
        )

    def _arith(self, *_a: Any, **_k: Any) -> Any:
        raise UnassessedError(
            "arithmetic on UNASSESSED. A formula could not be evaluated; its "
            "result cannot participate in a calculation."
        )

    __add__ = __radd__ = __sub__ = __rsub__ = _arith
    __mul__ = __rmul__ = __truediv__ = __rtruediv__ = _arith
    __floordiv__ = __rfloordiv__ = __mod__ = __rmod__ = _arith
    __pow__ = __rpow__ = __neg__ = __abs__ = _arith
    __lt__ = __le__ = __gt__ = __ge__ = _arith
    __float__ = __int__ = __round__ = _arith

    def __eq__(self, other: Any) -> bool:
        return other is self

    def __ne__(self, other: Any) -> bool:
        return other is not self

    def __hash__(self) -> int:
        return hash("__UNASSESSED__")

    def __reduce__(self):
        return (_Unassessed, ())


UNASSESSED = _Unassessed()


class Confidence(str, Enum):
    """How much weight the result of a formula can bear.

    Ordered. ``OBSERVED`` was read from a source document or a system of
    record. ``ASSERTED`` was claimed by a model, or derived from something
    claimed by a model --- true of anything downstream of the extraction
    pipeline's AI judge. ``UNVERIFIED`` means the provenance was not stated,
    which is the honest default rather than an optimistic one.

    This is not ``UNASSESSED``'s peer: a result either has a value (with one of
    these three confidences) or it does not have a value at all.
    """

    OBSERVED = "observed"
    ASSERTED = "asserted"
    UNVERIFIED = "unverified"

    @property
    def rank(self) -> int:
        return _CONFIDENCE_RANK[self]

    def __lt__(self, other: Any) -> bool:  # type: ignore[override]
        if isinstance(other, Confidence):
            return self.rank < other.rank
        return NotImplemented

    def __le__(self, other: Any) -> bool:  # type: ignore[override]
        if isinstance(other, Confidence):
            return self.rank <= other.rank
        return NotImplemented

    def __gt__(self, other: Any) -> bool:  # type: ignore[override]
        if isinstance(other, Confidence):
            return self.rank > other.rank
        return NotImplemented

    def __ge__(self, other: Any) -> bool:  # type: ignore[override]
        if isinstance(other, Confidence):
            return self.rank >= other.rank
        return NotImplemented


_CONFIDENCE_RANK = {
    Confidence.OBSERVED: 2,
    Confidence.ASSERTED: 1,
    Confidence.UNVERIFIED: 0,
}


def weakest(confidences: Iterable[Confidence]) -> Confidence:
    """The floor of a set of confidences.

    An empty set yields UNVERIFIED: a formula with no stated input provenance
    has not earned more than that. Silence is not evidence.
    """
    lowest = Confidence.OBSERVED
    seen = False
    for conf in confidences:
        seen = True
        if conf.rank < lowest.rank:
            lowest = conf
    return lowest if seen else Confidence.UNVERIFIED


__all__ = [
    "UNASSESSED",
    "UnassessedError",
    "Confidence",
    "weakest",
]
