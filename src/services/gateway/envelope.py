"""The result every gateway call returns: data, where it came from, when it was read, and whether it was cut."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

ERROR_CODES = ("NOT_ALLOWED", "SCHEMA_INVALID", "LIMIT_EXCEEDED", "NOT_FOUND", "AMBIGUOUS", "TIMEOUT", "UNAVAILABLE",
               "UNSUPPORTED", "NO_PROVENANCE")


@dataclass
class GatewayError:
    code: str
    message: str = ""
    retryable: bool = False
    candidates: List[Any] = field(default_factory=list)


class GatewayFault(Exception):
    """Raised by a function body to return a structured error rather than a crash."""

    def __init__(self, code: str, message: str = "", *, retryable: bool = False, candidates: Optional[List[Any]] = None):
        super().__init__(message or code)
        self.error = GatewayError(code, message or code, retryable, list(candidates or []))

    @property
    def candidates(self) -> List[Any]:
        return self.error.candidates


@dataclass
class Raw:
    """What a function body returns; the gateway turns it into a GatewayResult."""
    data: Any
    provenance: List[Dict[str, Any]]
    total_available: Optional[int] = None
    as_of: Optional[str] = None
    truncated: bool = False          # for a function that bounds its own result (an aggregate cut to the top N)


@dataclass
class GatewayResult:
    function: str
    version: int = 1
    ok: bool = True
    data: Any = field(default_factory=list)
    provenance: List[Dict[str, Any]] = field(default_factory=list)
    as_of: str = ""
    truncated: bool = False
    total_available: Optional[int] = None
    latency_ms: int = 0
    call_id: str = ""
    error: Optional[GatewayError] = None

    def to_dict(self) -> Dict[str, Any]:
        out = {"function": self.function, "ok": self.ok, "data": self.data, "provenance": self.provenance, "as_of": self.as_of,
               "truncated": self.truncated, "total_available": self.total_available, "call_id": self.call_id}
        if self.error:
            out["error"] = {"code": self.error.code, "message": self.error.message, "retryable": self.error.retryable,
                            "candidates": self.error.candidates}
        return out
