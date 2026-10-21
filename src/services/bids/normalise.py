"""Normalise supplier bids to a like-for-like comparison (SC-021): pure, and every step visible.

Each cell traces from the supplier's raw price to the normalised price: tier, unit of measure,
currency, payment terms. Adjustments the buyer bears (freight, insurance, an exception's cost) sit
beside the price, never inside it. A term that cannot be priced is flagged as not normalised; it is
never assumed, and a rate that is missing is never replaced by one-to-one. Submitted values are
never changed.
"""
from __future__ import annotations

import re
from datetime import date, datetime
from difflib import SequenceMatcher
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

METHOD_VERSION = "1.0.0"
POLICIES = ("flag_only", "exclude_supplier_from_total", "fill_with_highest_bid", "fill_with_average_bid",
            "fill_with_last_paid_price")
_MATCH_MIN, _MATCH_MARGIN = 0.85, 0.05
_INCOTERM_TYPES = {"freight", "insurance", "duty", "incoterm", "destination_charges", "clearance"}


class NormaliseError(ValueError):
    """The inputs cannot be normalised; the message says what to fix."""


def _f(value: Any, what: str) -> float:
    if isinstance(value, bool) or value is None:
        raise NormaliseError(f"{what} is missing or not a number")
    try:
        out = float(value)
    except (TypeError, ValueError):
        raise NormaliseError(f"{what} is not a number") from None
    if out != out or abs(out) == float("inf"):
        raise NormaliseError(f"{what} is not a finite number")
    return out


def _key(text: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text or "").lower())


def _parse_time(value: Any, what: str) -> datetime:
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        raise NormaliseError(f"{what} is not an ISO 8601 time") from None


def _rate(currency: str, base: str, fx: Mapping[str, Any]) -> Optional[float]:
    if currency == base:
        return 1.0
    v = fx.get(currency)
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if v > 0 else None


def _match(line: Mapping[str, Any], master: Mapping[str, Mapping[str, Any]]):
    lid = line.get("line_id")
    if lid is not None and str(lid) in master:
        return str(lid), {"method": "line_id", "confidence": 1.0}
    desc = _key(line.get("description"))
    if not desc:
        return None, {"method": "none", "confidence": 0.0, "reason": "no line_id or description"}
    scored = sorted(((SequenceMatcher(None, desc, _key(m.get("description"))).ratio(), mid)
                     for mid, m in master.items()), reverse=True)
    best, second = scored[0], (scored[1] if len(scored) > 1 else (0.0, None))
    if best[0] >= _MATCH_MIN and best[0] - second[0] >= _MATCH_MARGIN:
        return best[1], {"method": "description", "confidence": round(best[0], 4)}
    why = "ambiguous" if best[0] >= _MATCH_MIN else "below 0.85"
    return None, {"method": "description", "confidence": round(best[0], 4), "reason": why}


def _tier(line: Mapping[str, Any], qty: float) -> Tuple[float, Optional[Dict[str, Any]]]:
    tiers = [t for t in (line.get("price_tiers") or []) if isinstance(t, Mapping)]
    usable = [t for t in tiers if float(t["min_qty"]) <= qty]
    if usable:
        t = max(usable, key=lambda x: float(x["min_qty"]))
        return float(t["unit_price"]), {"min_qty": t["min_qty"], "unit_price": t["unit_price"]}
    return _f(line.get("unit_price"), "unit_price"), None


def _uom_factor(line: Mapping[str, Any], master_uom: str, conversions: Sequence[Mapping[str, Any]], lid: str):
    uom = str(line.get("uom") or master_uom)
    if uom == master_uom:
        return 1.0
    for scope in (lid, None):
        for c in conversions:
            if c.get("from_uom") == uom and c.get("to_uom") == master_uom and c.get("line_id") == scope:
                return _f(c.get("factor"), "uom conversion factor")
    return None


def _shape_problems(line_master: Any, bids: Any) -> List[str]:
    """Every problem with the shape of the rows, once each, so a caller can fix them in one go."""
    out: List[str] = []
    if not isinstance(line_master, (list, tuple)):
        out.append("line_master must be a list of rows like "
                   '{"line_id": "L1", "description": "Steel bracket", "uom": "EA", "qty": 1000}')
    if not isinstance(bids, (list, tuple)):
        out.append("bids must be a list with one entry per supplier, like "
                   '{"supplier_id": "A", "currency": "EUR", "lines": [{"line_id": "L1", "unit_price": 10.0, "uom": "EA"}]}')
    if out:
        return out
    seen = set()

    def need(row: Any, keys: Sequence[str], what: str, example: str) -> None:
        if not isinstance(row, Mapping):
            out.append(f"{what} must be an object, expected like {example}")
            return
        missing = tuple(k for k in keys if row.get(k) is None)
        if missing and (what, missing) not in seen:
            seen.add((what, missing))
            out.append(f"{what} {dict(row)} has no {', '.join(missing)}; expected like {example}")

    for m in line_master:
        need(m, ("line_id", "uom", "qty"), "a line_master row",
             '{"line_id": "L1", "description": "Steel bracket", "uom": "EA", "qty": 1000}')
    for b in bids:
        need(b, ("supplier_id", "lines"), "a bid",
             '{"supplier_id": "A", "currency": "EUR", "lines": [{"line_id": "L1", "unit_price": 10.0, "uom": "EA"}]}')
        if isinstance(b, Mapping) and isinstance(b.get("lines"), (list, tuple)):
            for line in b["lines"]:
                need(line, ("unit_price",), "a bid line", '{"line_id": "L1", "unit_price": 10.0, "uom": "EA"}')
                if isinstance(line, Mapping) and line.get("line_id") is None and not line.get("description") \
                        and ("a bid line", ("line_id or description",)) not in seen:
                    seen.add(("a bid line", ("line_id or description",)))
                    out.append(f"a bid line {dict(line)} has no line_id or description; expected like "
                               '{"line_id": "L1", "unit_price": 10.0, "uom": "EA"}')
        elif isinstance(b, Mapping) and b.get("lines") is not None:
            out.append(f"the lines of bid {b.get('supplier_id')} must be a list")
    return out


def normalise_bids(
    line_master: Sequence[Mapping[str, Any]], bids: Sequence[Mapping[str, Any]], base_currency: str,
    fx_rates: Mapping[str, Any], fx_type: Optional[str] = None, fx_date: Optional[str] = None,
    target_incoterm: Optional[str] = None, target_payment_days: float = 60,
    cost_of_capital: Optional[float] = None, uom_conversions: Sequence[Mapping[str, Any]] = (),
    missing_line_policy: str = "flag_only", last_paid_prices: Optional[Mapping[str, Any]] = None,
    award_date: Optional[str] = None, sealed: bool = False, opened_at: Optional[str] = None,
    as_of: Optional[str] = None,
) -> Dict[str, Any]:
    problems = _shape_problems(line_master, bids)
    if problems:
        raise NormaliseError(" | ".join(problems))
    if missing_line_policy not in POLICIES:
        raise NormaliseError(f"missing_line_policy must be one of {', '.join(POLICIES)}")
    if sealed:
        if not opened_at or not as_of:
            raise NormaliseError("a sealed event needs opened_at and as_of")
        if _parse_time(as_of, "as_of") < _parse_time(opened_at, "opened_at"):
            raise NormaliseError("these bids are sealed until the bid opening time and cannot be compared before it")
    if missing_line_policy == "fill_with_last_paid_price" and not last_paid_prices:
        raise NormaliseError("fill_with_last_paid_price needs last_paid_prices for the missing lines")
    master: Dict[str, Mapping[str, Any]] = {}
    for m in line_master:
        lid = str(m.get("line_id"))
        if lid in master:
            raise NormaliseError(f"line {lid} appears twice in the line master")
        _f(m.get("qty"), f"qty of line {lid}")
        master[lid] = m
    if not master:
        raise NormaliseError("the line master is empty")
    seen = set()
    for b in bids:
        sid = str(b.get("supplier_id"))
        if sid in seen:
            raise NormaliseError(f"supplier {sid} appears twice; one bid per supplier")
        seen.add(sid)
    target_days = _f(target_payment_days, "target_payment_days")
    r = None if cost_of_capital is None else _f(cost_of_capital, "cost_of_capital")

    assumptions: List[str] = []
    warnings: List[str] = []
    matrix: Dict[str, Dict[str, Any]] = {lid: {} for lid in master}
    suppliers: Dict[str, Dict[str, Any]] = {}
    exceptions: Dict[str, List[Dict[str, Any]]] = {}
    alternates: Dict[str, List[Dict[str, Any]]] = {}
    fx_label = {"type": fx_type, "date": fx_date, "base_currency": base_currency}

    def normalise_line(b, line, lid, match):
        m = master[lid]
        qty = float(m["qty"])
        cur = str(line.get("currency") or b.get("currency") or base_currency)
        flags: List[str] = []
        adjustments: List[Dict[str, Any]] = []
        price, tier = _tier(line, qty)
        unit = float(line.get("price_unit") or 1)
        if unit <= 0:
            raise NormaliseError(f"price_unit of line {lid} must be above zero")
        price /= unit
        raw = float(line.get("unit_price"))
        factor = _uom_factor(line, str(m.get("uom")), uom_conversions, lid)
        if factor is None:
            flags.append("uom_unconvertible")
        else:
            price /= factor
        rate = _rate(cur, base_currency, fx_rates)
        if rate is None:
            flags.append("no_fx_rate")
        if raw < 0:
            flags.append("rebate")
        norm: Optional[float] = None
        if "uom_unconvertible" not in flags and rate is not None:
            norm = price * rate
            pdays = b.get("payment_days")
            if r is not None and pdays is not None:
                if float(pdays) != target_days:
                    f = (1 + r) ** ((target_days - float(pdays)) / 365.0)
                    norm *= f
                    adjustments.append({"type": "payment_terms", "factor": f,
                                        "basis": f"price x (1 + {r}) ^ (({target_days:g} - {float(pdays):g}) / 365)"})
        cell = {"raw_unit_price": raw, "raw_currency": cur, "tier_used": tier, "uom_factor": factor if factor else None,
                "fx_rate": rate, "fx_date": fx_date, "fx_type": fx_type, "adjustments": adjustments,
                "normalised_unit_price": norm, "extended_value": None if norm is None else round(norm * qty, 2),
                "qty": qty, "comparable": norm is not None, "flags": flags, "filled": False, "match": match}
        return cell

    for b in bids:
        sid = str(b.get("supplier_id"))
        cur = str(b.get("currency") or base_currency)
        sup: Dict[str, Any] = {"currency": cur, "incoterm": b.get("incoterm"), "payment_days": b.get("payment_days"),
                               "terms_not_normalised": [], "unmatched_lines": [], "missing_lines": [],
                               "bid_adjustments": [], "non_comparable_lines": []}
        exc: List[Dict[str, Any]] = []
        alts: List[Dict[str, Any]] = []
        for e in b.get("exceptions") or []:
            exc.append({"text": e.get("text"), "category": e.get("category") or "other",
                        "severity": e.get("severity") or "unpriced", "applies_to": list(e.get("applies_to") or []),
                        "cost_loading": e.get("cost_loading")})
        blocked_lines = {str(l) for e in exc if e["severity"] in ("unpriced", "blocking") for l in e["applies_to"]}
        for line in b.get("lines") or []:
            lid, match = _match(line, master)
            if lid is None:
                sup["unmatched_lines"].append({**dict(line), "match": match})
                continue
            cell = normalise_line(b, line, lid, match)
            if line.get("alternate_flag"):
                alts.append({"line_id": lid, "normalised_unit_price": cell["normalised_unit_price"],
                             "raw_unit_price": cell["raw_unit_price"], "delta_vs_compliant": None, "flags": cell["flags"]})
                continue
            if lid in matrix and sid in matrix[lid]:
                raise NormaliseError(f"supplier {sid} has two compliant lines for {lid}; mark one as an alternate")
            if lid in blocked_lines:
                cell["comparable"] = False
                cell["flags"].append("unpriced_exception")
            matrix[lid][sid] = cell
        for a in alts:
            base = matrix[a["line_id"]].get(sid)
            if base and base["normalised_unit_price"] is not None and a["normalised_unit_price"] is not None:
                a["delta_vs_compliant"] = a["normalised_unit_price"] - base["normalised_unit_price"]
        # Terms the buyer cannot compare unless someone prices them.
        t_inc = target_incoterm
        adj_rows = [dict(a) for a in (b.get("adjustments") or [])]
        if t_inc and b.get("incoterm") != t_inc:
            priced = any(str(a.get("type")) in _INCOTERM_TYPES for a in adj_rows)
            if not priced:
                exc.append({"text": f"bid is on {b.get('incoterm') or 'no stated Incoterm'}; target is {t_inc}; the "
                                    "difference was not priced", "category": "incoterm", "severity": "unpriced",
                            "applies_to": [], "cost_loading": None})
                sup["terms_not_normalised"].append("incoterm")
        if (r is None and b.get("payment_days") is not None and float(b["payment_days"]) != target_days) \
                or b.get("payment_days") is None:
            sup["terms_not_normalised"].append("payment_terms")
        for a in adj_rows:
            ccy = str(a.get("currency") or base_currency)
            rate = _rate(ccy, base_currency, fx_rates)
            amt = _f(a.get("amount"), "adjustment amount")
            if rate is None:
                raise NormaliseError(f"adjustment in {ccy} has no exchange rate to {base_currency}")
            sign = -1.0 if a.get("effect") == "subtract" else 1.0
            sup["bid_adjustments"].append({**a, "amount_base": sign * amt * rate})
        for e in exc:
            if e["severity"] == "priced" and e["cost_loading"] is not None:
                ccy = str(e.get("currency") or base_currency)
                rate = _rate(ccy, base_currency, fx_rates) or 1.0
                sup["bid_adjustments"].append({"type": "exception_loading", "amount": float(e["cost_loading"]),
                                               "currency": ccy, "effect": "add", "note": e["text"],
                                               "amount_base": float(e["cost_loading"]) * rate})
        if b.get("validity_date") and award_date:
            try:
                if date.fromisoformat(str(b["validity_date"])) < date.fromisoformat(str(award_date)):
                    warnings.append(f"the validity of bid {sid} ({b['validity_date']}) ends before the award date ({award_date})")
            except ValueError:
                raise NormaliseError("validity_date and award_date must be ISO 8601 dates") from None
        suppliers[sid] = sup
        exceptions[sid] = exc
        alternates[sid] = alts

    # Missing lines.
    for lid in master:
        for sid, sup in suppliers.items():
            if sid not in matrix[lid]:
                sup["missing_lines"].append(lid)
    if missing_line_policy.startswith("fill_with"):
        for lid in master:
            for sid, sup in suppliers.items():
                if lid not in sup["missing_lines"]:
                    continue
                others = [c["normalised_unit_price"] for s, c in matrix[lid].items() if c["comparable"]]
                if missing_line_policy == "fill_with_last_paid_price":
                    paid = (last_paid_prices or {}).get(lid)
                    if paid is None:
                        continue
                    price, raw = float(paid), float(paid)
                elif others:
                    price = max(others) if missing_line_policy == "fill_with_highest_bid" else sum(others) / len(others)
                    raw = price
                else:
                    continue
                matrix[lid][sid] = {"raw_unit_price": raw, "raw_currency": base_currency, "tier_used": None,
                                    "uom_factor": None, "fx_rate": 1.0, "fx_date": fx_date, "fx_type": fx_type,
                                    "adjustments": [], "normalised_unit_price": price,
                                    "extended_value": round(price * float(master[lid]["qty"]), 2),
                                    "qty": float(master[lid]["qty"]), "comparable": True, "flags": ["filled"],
                                    "filled": True, "match": {"method": "fill", "confidence": None}}
                sup["missing_lines"].remove(lid)
                sup.setdefault("filled_lines", []).append(lid)
                assumptions.append(f"Line {lid} for {sid} was filled by {missing_line_policy}; it is an assumption, not a bid.")

    for sid, sup in suppliers.items():
        cells = [matrix[lid][sid] for lid in master if sid in matrix[lid]]
        comparable = [c for c in cells if c["comparable"] and c["extended_value"] is not None]
        sup["non_comparable_lines"] = [lid for lid in master if sid in matrix[lid] and not matrix[lid][sid]["comparable"]]
        priced = [c for c in cells if not c["filled"]]
        sup["completeness_pct"] = round(100.0 * len(priced) / len(master), 2)
        line_total = sum(c["extended_value"] for c in comparable)
        adj_total = sum(a["amount_base"] for a in sup["bid_adjustments"])
        incomplete = bool(sup["missing_lines"])
        sup["lines_total"] = round(line_total, 2)
        sup["total"] = None if (missing_line_policy == "exclude_supplier_from_total" and incomplete) \
            else round(line_total + adj_total, 2)

    lowest: Dict[str, Any] = {}
    for lid in master:
        cand = [(c["normalised_unit_price"], s) for s, c in matrix[lid].items() if c["comparable"] and not c["filled"]]
        if cand:
            p, s = min(cand)
            lowest[lid] = {"supplier_id": s, "unit_price": p}
    cherry = None
    if len(lowest) == len(master):
        cherry = round(sum(min(c["extended_value"] for c in matrix[lid].values() if c["comparable"] and not c["filled"])
                           for lid in master), 2)
    full = {sid: sup["lines_total"] for sid, sup in suppliers.items()
            if all(sid in matrix[lid] and matrix[lid][sid]["comparable"] and not matrix[lid][sid]["filled"] for lid in master)}
    best = None
    if full:
        sid = min(full, key=lambda s: (full[s], s))
        best = {"supplier_id": sid, "total": full[sid], "basis": "lines only; bid-level adjustments excluded"}
    assumptions.append(f"Exchange rates are those supplied ({fx_type or 'type not stated'}, {fx_date or 'date not stated'}); none was looked up.")
    return {"matrix": matrix, "suppliers": suppliers, "exceptions": exceptions, "alternates": alternates,
            "lowest_per_line": lowest, "cherry_pick_total": cherry, "best_single_supplier": best,
            "warnings": warnings, "assumptions": assumptions, "missing_line_policy": missing_line_policy,
            "fx": fx_label, "target": {"incoterm": target_incoterm, "payment_days": target_days,
                                       "cost_of_capital": r}, "method_version": METHOD_VERSION}
