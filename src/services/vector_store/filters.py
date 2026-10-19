"""Translate the filter objects the application builds (``filter_models``) into VectorStore filters.

Duck-typed (``.key`` / ``.match`` / ``.must`` / ``.should``). A payload key that has no column filters the stored
metadata, as it did in Qdrant; ``must_not`` is refused rather than silently dropped; ``tenant_id`` can never be filtered on.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .pgvector_store import FILTER_COLUMNS


def _is_condition(obj: Any) -> bool:
    return hasattr(obj, "key") and hasattr(obj, "match")


def _condition(cond: Any) -> Dict[str, Any]:
    key = str(getattr(cond, "key", "") or "").strip()
    if not key or key == "tenant_id" or key.startswith("tenant"):
        raise ValueError(f"a filter on {key!r} is not allowed")
    match = getattr(cond, "match", None)
    if hasattr(match, "any"):
        values = [v for v in (getattr(match, "any", None) or [])]
        if key in FILTER_COLUMNS:
            return {key: values}
        return {"metadata_in": {key: values}}
    if hasattr(match, "value"):
        value = getattr(match, "value")
        return {key: value} if key in FILTER_COLUMNS else {"metadata": {key: value}}
    raise ValueError(f"unsupported match on {key!r}")


def _merge(into: Dict[str, Any], part: Dict[str, Any]) -> None:
    for key, value in part.items():
        if key in ("metadata", "metadata_in"):
            clash = set(into.get(key, {})) & set(value)
            if clash:
                raise ValueError(f"two conditions on {sorted(clash)}")
            into.setdefault(key, {}).update(value)
        elif key in into:
            raise ValueError(f"two conditions on {key!r}")
        else:
            into[key] = value


def to_store_filter(flt: Optional[Any]) -> Dict[str, Any]:
    if flt is None:
        return {}
    if getattr(flt, "must_not", None):
        raise ValueError("must_not filters are not supported by the vector store")
    out: Dict[str, Any] = {}
    for item in list(getattr(flt, "must", None) or []):
        _merge(out, _condition(item) if _is_condition(item) else to_store_filter(item))
    should = list(getattr(flt, "should", None) or [])
    if should:
        alternatives: List[Dict[str, Any]] = []
        for item in should:
            alt = _condition(item) if _is_condition(item) else to_store_filter(item)
            if "any_of" in alt:                      # an OR nested in an OR: flatten its alternatives
                alternatives.extend(alt.pop("any_of"))
                if alt:
                    alternatives.append(alt)
            elif alt:
                alternatives.append(alt)
        if alternatives:
            if "any_of" in out:
                raise ValueError("two OR groups in one filter")
            out["any_of"] = alternatives
    return out
