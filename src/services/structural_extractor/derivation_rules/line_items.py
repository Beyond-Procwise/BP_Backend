from src.services.structural_extractor.types import ExtractedValue


def _mk(value, rule_id: str, inputs: dict):
    return ExtractedValue(
        value=value,
        provenance="derived",
        derivation_trace={"rule_id": rule_id, "inputs": inputs},
        source="derivation_registry",
        confidence=1.0,
        attempt=1,
    )


def derive_line_item_fields(item: dict) -> dict:
    """Apply math inversion rules to a single line item. Returns a new dict
    with original fields + derived ones."""
    out = dict(item)
    qty = out.get("quantity")
    price = out.get("unit_price")
    total = out.get("line_total") or out.get("line_amount")

    qty_v = qty.value if qty else None
    price_v = price.value if price else None
    total_v = total.value if total else None

    if qty_v is not None and price_v is not None and total_v is None:
        out["line_total"] = _mk(
            round(float(qty_v) * float(price_v), 2),
            "line_total_from_qty_price",
            {"quantity": qty_v, "unit_price": price_v},
        )
    elif qty_v is not None and price_v is None and total_v is not None and float(qty_v) != 0:
        out["unit_price"] = _mk(
            round(float(total_v) / float(qty_v), 2),
            "unit_price_from_qty_total",
            {"quantity": qty_v, "line_total": total_v},
        )
    elif qty_v is None and price_v is not None and total_v is not None and float(price_v) != 0:
        out["quantity"] = _mk(
            round(float(total_v) / float(price_v)),
            "quantity_from_price_total",
            {"unit_price": price_v, "line_total": total_v},
        )
    return out
