import re

from src.services.structural_extractor.derivation import rule

_SUPPLIER_LOOKUP = None  # callable: (name: str) -> Optional[str]
_BUYER_LOOKUP = None


def set_supplier_lookup(fn):
    global _SUPPLIER_LOOKUP
    _SUPPLIER_LOOKUP = fn


def set_buyer_lookup(fn):
    global _BUYER_LOOKUP
    _BUYER_LOOKUP = fn


def _normalize_name(name: str) -> str:
    s = str(name or "").strip().lower()
    s = re.sub(r"[^a-z0-9]", "", s)
    return s


def _generate_id(prefix: str, name: str) -> str:
    return f"{prefix}-{_normalize_name(name).upper()}"


@rule("supplier_id_from_lookup", "supplier_id", ["supplier_name"])
def _supplier_id(inputs):
    name = str(inputs["supplier_name"] or "")
    if not name:
        return None
    if _SUPPLIER_LOOKUP is not None:
        hit = _SUPPLIER_LOOKUP(name)
        if hit:
            return hit
    return _generate_id("SUP", name)


@rule("buyer_id_from_lookup", "buyer_id", ["buyer_name"])
def _buyer_id(inputs):
    name = str(inputs["buyer_name"] or "")
    if not name:
        return None
    if _BUYER_LOOKUP is not None:
        hit = _BUYER_LOOKUP(name)
        if hit:
            return hit
    return _generate_id("BUYER", name)
