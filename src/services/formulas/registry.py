"""The ``@formula`` decorator and the in-memory registry it fills.

A registered formula is a pure function plus everything needed to say what it
computed and whether you can still trust the answer six months later: a typed
contract, a version, a hash over its own source, and golden vectors that must
pass before the module it lives in will finish importing.

That last part is the load-bearing one. Golden vectors that only run under
pytest are documentation; golden vectors checked at import time are a gate.
Two calibrated constant sets already exist in this codebase --- the benchmark
engine's penny-parity fixture and the deal-clustering golden batch that
``quote_rival``'s alpha was tuned against --- and both can currently be broken
by an edit that imports perfectly cleanly.
"""
from __future__ import annotations

import hashlib
import inspect
import logging
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .contract import Contract, Output, Term

logger = logging.getLogger(__name__)


class FormulaKind(str, Enum):
    """Whether the formula reads one context or a whole set of them.

    ``SET`` is not a batching convenience. Some of the calculations here are
    genuinely population-scoped: a min-max normalised criterion score and a
    ratio-to-cheapest price score are properties of the *frame*, not of a row,
    and evaluating one row in isolation cannot reproduce them. Declaring the
    difference is what stops a later refactor from "optimising" a set formula
    into a loop and silently changing every number it produces.
    """

    SCALAR = "scalar"
    SET = "set"


class GoldenVectorFailure(AssertionError):
    """A registered formula did not reproduce one of its own golden vectors."""


class FormulaError(Exception):
    """Registration-time problem with a formula declaration."""


@dataclass(frozen=True)
class GoldenVector:
    """One pinned input/output pair.

    ``expected`` may be a partial mapping: only the keys present are compared.
    That keeps a vector for a formula returning a twenty-field audit dict
    readable, while still pinning the fields that matter.
    """

    inputs: Any
    expected: Any
    note: str = ""
    tolerance: float = 0.0
    #: SET formulas only: the shared parameters for this vector's population.
    shared: Optional[Mapping[str, Any]] = None

    def digest(self) -> str:
        return (
            f"{_stable_repr(self.inputs)}"
            f"|{_stable_repr(self.shared or {})}"
            f"=>{_stable_repr(self.expected)}@{self.tolerance:g}"
        )


@dataclass(frozen=True)
class FormulaSpec:
    """Everything the registry knows about one formula."""

    name: str
    version: str
    contract: Contract
    fn: Callable[..., Any]
    kind: FormulaKind
    effective_from: date
    owner: str
    purpose: str
    gpss_version: Optional[str]
    golden: Tuple[GoldenVector, ...]
    version_hash: str
    source_module: str
    source_line: int
    replaces: Tuple[str, ...] = ()
    notes: str = ""

    @property
    def qualified_version(self) -> str:
        return f"{self.version}+{self.version_hash}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "version_hash": self.version_hash,
            "qualified_version": self.qualified_version,
            "kind": self.kind.value,
            "owner": self.owner,
            "purpose": self.purpose,
            "effective_from": self.effective_from.isoformat(),
            "gpss_version": self.gpss_version,
            "contract": self.contract.to_dict(),
            "golden_vectors": len(self.golden),
            "source": f"{self.source_module}:{self.source_line}",
            "replaces": list(self.replaces),
            "notes": self.notes,
        }


REGISTRY: Dict[str, FormulaSpec] = {}

#: When each formula's golden vectors last passed. Set at registration (which is
#: import time) and refreshed by ``verify_all()``. This is the "last validated"
#: column of the model inventory, and it is a real timestamp rather than a
#: hand-maintained date precisely because a hand-maintained one drifts.
VALIDATED_AT: Dict[str, "datetime"] = {}


def _stable_repr(value: Any) -> str:
    """Order-independent text form, so a dict literal's ordering is not part
    of the hash."""
    if isinstance(value, Mapping):
        items = sorted((str(k), _stable_repr(v)) for k, v in value.items())
        return "{" + ",".join(f"{k}:{v}" for k, v in items) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_stable_repr(v) for v in value) + "]"
    if isinstance(value, set):
        return "{" + ",".join(sorted(_stable_repr(v) for v in value)) + "}"
    if isinstance(value, float):
        return repr(round(value, 12))
    return repr(value)


def _body_source(fn: Callable[..., Any]) -> str:
    """The function's source with the decorator call stripped.

    The decorator's own text carries the contract and the vectors, both of which
    are hashed separately; including it twice would make the hash churn on a
    reformat that changed nothing.
    """
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError):  # pragma: no cover - only for exotic callables
        return f"<no source for {getattr(fn, '__name__', fn)!r}>"
    lines = src.splitlines()
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith("def ") or stripped.startswith("async def "):
            lines = lines[i:]
            break
    return "\n".join(l.rstrip() for l in lines if l.strip())


def compute_version_hash(
    fn: Callable[..., Any], contract: Contract, golden: Sequence[GoldenVector]
) -> str:
    """Hash of source + contract + golden vectors, per the spec.

    Any of the three changing changes the hash, which is the point: a stored
    evaluation record names the exact thing that produced it, and a silent edit
    to the maths, the contract or the pinned expectations cannot masquerade as
    the version that was audited.
    """
    h = hashlib.sha256()
    h.update(_body_source(fn).encode("utf-8"))
    h.update(b"\x00")
    h.update(contract.signature().encode("utf-8"))
    h.update(b"\x00")
    for vec in golden:
        h.update(vec.digest().encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def _close(actual: Any, expected: Any, tol: float) -> bool:
    if isinstance(expected, float) and isinstance(actual, (int, float)):
        if math.isnan(expected):
            return math.isnan(float(actual))
        return abs(float(actual) - expected) <= tol
    return False


def matches(actual: Any, expected: Any, tol: float = 0.0) -> Tuple[bool, str]:
    """Compare a result against a golden expectation.

    A mapping expectation is a *subset* match; a sequence expectation is
    element-wise and length-sensitive. Returns (ok, first-mismatch-path).
    """
    if isinstance(expected, Mapping):
        # A mapping expectation reads attributes too, so a formula returning a
        # dataclass or a pydantic model can be pinned field-by-field without
        # writing the whole object out.
        if isinstance(actual, Mapping):
            def _get(k):
                return (k in actual, actual.get(k))
        else:
            def _get(k):
                return (hasattr(actual, k), getattr(actual, k, None))

        for key, exp in expected.items():
            present, value = _get(key)
            if not present:
                return False, f"missing key {key!r}"
            ok, why = matches(value, exp, tol)
            if not ok:
                return False, f"{key}.{why}" if why else str(key)
        return True, ""
    if isinstance(expected, (list, tuple)) and not isinstance(expected, str):
        if not isinstance(actual, (list, tuple)) or len(actual) != len(expected):
            got = len(actual) if isinstance(actual, (list, tuple)) else type(actual).__name__
            return False, f"expected {len(expected)} items, got {got}"
        for i, exp in enumerate(expected):
            ok, why = matches(actual[i], exp, tol)
            if not ok:
                return False, f"[{i}]{('.' + why) if why else ''}"
        return True, ""
    if _close(actual, expected, tol):
        return True, ""
    if actual == expected:
        return True, ""
    return False, f"expected {expected!r}, got {actual!r}"


def _run_golden(spec_name: str, fn: Callable[..., Any], kind: FormulaKind,
                golden: Sequence[GoldenVector]) -> None:
    for idx, vec in enumerate(golden):
        try:
            if kind is FormulaKind.SET:
                actual = fn(vec.inputs, **(vec.shared or {}))
            else:
                actual = fn(**vec.inputs)
        except Exception as exc:  # noqa: BLE001 - re-raised with context below
            raise GoldenVectorFailure(
                f"{spec_name}: golden vector {idx} raised {type(exc).__name__}: {exc}"
                + (f" ({vec.note})" if vec.note else "")
            ) from exc
        ok, why = matches(actual, vec.expected, vec.tolerance)
        if not ok:
            raise GoldenVectorFailure(
                f"{spec_name}: golden vector {idx} does not reproduce: {why}"
                + (f" ({vec.note})" if vec.note else "")
            )


def formula(
    name: str,
    *,
    version: str,
    inputs: Sequence[Term],
    output: Output,
    effective_from: date,
    gpss_version: Optional[str] = None,
    owner: str,
    purpose: str,
    golden: Sequence[GoldenVector] = (),
    kind: FormulaKind = FormulaKind.SCALAR,
    replaces: Sequence[str] = (),
    notes: str = "",
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register a pure, deterministic function as a named, versioned formula.

    ``gpss_version`` is accepted because the specification names it. It is
    recorded and reported, and it resolves against nothing: there is no GPSS
    dictionary in this project, and the decision not to invent one is recorded
    in docs/remediation/00_seam_map.md B3. Input terms bind to ``concept_code``,
    the vocabulary derived from the extraction schemas, via ``Term.concept_code``.

    Raises at import time if a golden vector does not reproduce, so a module
    whose maths has drifted cannot be imported at all.
    """
    if not golden:
        raise FormulaError(
            f"{name}: at least one golden vector is required. A formula with no "
            f"pinned behaviour cannot be refactored safely, which is the whole "
            f"reason this registry exists."
        )

    contract = Contract(inputs=tuple(inputs), output=output)

    seen: set = set()
    for term in contract.inputs:
        if term.name in seen:
            raise FormulaError(f"{name}: duplicate input term {term.name!r}")
        seen.add(term.name)

    def decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
        if name in REGISTRY and REGISTRY[name].fn is not fn:
            existing = REGISTRY[name]
            raise FormulaError(
                f"formula {name!r} is already registered from "
                f"{existing.source_module}:{existing.source_line}"
            )

        _run_golden(name, fn, kind, tuple(golden))

        try:
            source_line = inspect.getsourcelines(fn)[1]
        except (OSError, TypeError):  # pragma: no cover
            source_line = 0

        spec = FormulaSpec(
            name=name,
            version=version,
            contract=contract,
            fn=fn,
            kind=kind,
            effective_from=effective_from,
            owner=owner,
            purpose=purpose,
            gpss_version=gpss_version,
            golden=tuple(golden),
            version_hash=compute_version_hash(fn, contract, tuple(golden)),
            source_module=getattr(fn, "__module__", "?"),
            source_line=source_line,
            replaces=tuple(replaces),
            notes=notes,
        )
        REGISTRY[name] = spec
        VALIDATED_AT[name] = datetime.now(timezone.utc)
        fn.__formula__ = spec  # type: ignore[attr-defined]
        logger.debug("registered formula %s %s", name, spec.qualified_version)
        return fn

    return decorate


_REGISTERED = False
_REGISTERING = False


def ensure_registered() -> None:
    """Import the definitions package once, so the registry is populated.

    Call this from a formula's own home module before ``evaluate``. Several
    definition modules import the agents and services they delegate to, so
    those modules cannot import the definitions package at module scope
    without a cycle. Deferring the import to first use breaks the cycle
    without leaving anyone to remember an import order.
    """
    global _REGISTERED, _REGISTERING
    if _REGISTERED or _REGISTERING:
        # Re-entrant: a definition module is mid-import and has called into
        # something that calls back here. Returning is correct; completing the
        # outer import is what finishes the job.
        return
    _REGISTERING = True
    try:
        from . import definitions  # noqa: F401  (registration side effect)
    finally:
        _REGISTERING = False
    _REGISTERED = True


def get(name: str) -> FormulaSpec:
    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"no formula named {name!r}. Registered: {sorted(REGISTRY)}"
        ) from None


def names() -> List[str]:
    return sorted(REGISTRY)


def verify_all() -> List[str]:
    """Re-run every registered formula's golden vectors.

    Import-time checking already covers a normal run; this exists for CI, where
    the point is to fail loudly with a list rather than on the first import.
    """
    failures: List[str] = []
    for spec in REGISTRY.values():
        try:
            _run_golden(spec.name, spec.fn, spec.kind, spec.golden)
        except GoldenVectorFailure as exc:
            failures.append(str(exc))
        else:
            VALIDATED_AT[spec.name] = datetime.now(timezone.utc)
    return failures


__all__ = [
    "formula", "FormulaSpec", "FormulaKind", "GoldenVector", "GoldenVectorFailure",
    "FormulaError", "REGISTRY", "VALIDATED_AT", "get", "names", "verify_all",
    "compute_version_hash", "matches", "ensure_registered",
]
