"""Landed cost: what it costs to get the ordered goods to the buyer's delivery point.

Pure arithmetic, no I/O. The caller supplies the dated FX table and the policy values.

Four rules run through it:
  * A cost that is not mentioned is not zero. Every checklist item gets one status, and an
    item that is owed but has no amount and no rule to derive one is ``missing``.
  * The Incoterm decides what the buyer owes. With no Incoterm every unstated item except
    goods is ``missing``.
  * Nothing is converted at an invented rate. An amount whose currency has no rate in the
    dated table is ``missing``, never taken 1:1.
  * Everything is computed at full precision; rounding is for display only.

``not_stated`` is the one status beyond the six in the specification: an item that is only
owed *if the supplier charges it* (surcharges, tooling, packaging, bank fees ...) and that
the quote does not mention. It is listed to confirm with the supplier; it is not counted as
missing, because nothing says it is owed.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, List, Mapping, Optional, Sequence

STATED, INCLUDED, NOT_APPLICABLE = "stated", "included", "not_applicable"
DERIVED, ESTIMATED, MISSING, NOT_STATED = "derived", "estimated", "missing", "not_stated"

#: Statuses whose amount is part of the total.
COUNTED = (STATED, INCLUDED, DERIVED, ESTIMATED)

#: Cover is the invoice value plus 10%, the market convention for cargo insurance.
INSURED_VALUE_FACTOR = 1.10


@dataclass(frozen=True)
class Item:
    key: str
    group: str
    stage: str
    #: Owed whenever the buyer bears the stage. False = owed only if the supplier charges it.
    required: bool


CHECKLIST: Sequence[Item] = (
    Item("supplier_surcharges", "goods", "goods", False),
    Item("one_off_charges", "goods", "goods", False),
    Item("packaging", "goods", "goods", False),
    Item("compliance_testing", "goods", "goods", False),
    Item("inland_haulage_origin", "origin", "origin", True),
    Item("export_clearance", "origin", "export", True),
    Item("export_duties_licences", "origin", "export", False),
    Item("origin_handling", "origin", "origin", True),
    Item("freight", "main_carriage", "carriage", True),
    Item("fuel_surcharge_baf", "main_carriage", "carriage", False),
    Item("security_doc_fees", "main_carriage", "carriage", False),
    Item("cargo_insurance", "insurance", "insurance", True),
    Item("cargo_insurance_top_up", "insurance", "insurance_top_up", False),
    Item("import_duty", "import", "import", True),
    Item("additional_duties", "import", "import", False),
    Item("brokerage", "import", "import", True),
    Item("customs_exam_fees", "import", "import", False),
    Item("duty_deferment_fees", "import", "import", False),
    Item("import_vat", "import", "import", True),
    Item("terminal_handling", "destination", "destination", True),
    Item("delivery_order_fee", "destination", "destination", True),
    Item("storage_free_time", "destination", "destination", False),
    Item("on_carriage", "destination", "destination", True),
    Item("unloading", "destination", "unloading", True),
    Item("bank_fees", "payment", "payment", False),
    Item("lc_fees", "payment", "payment", False),
    Item("fx_margin", "payment", "payment", False),
)
_BY_KEY = {i.key: i for i in CHECKLIST}

# Stages the buyer bears under each Incoterm (the specification's table). "goods" and
# "payment" are not Incoterm stages: they are always the buyer's.
_FROM_FCA = {"carriage", "insurance", "import", "destination", "unloading"}
BUYER_PAYS: Mapping[str, frozenset] = {
    "EXW": frozenset({"origin", "export"} | _FROM_FCA),
    "FCA": frozenset(_FROM_FCA), "FAS": frozenset(_FROM_FCA), "FOB": frozenset(_FROM_FCA),
    "CFR": frozenset({"insurance", "import", "destination", "unloading"}),
    "CPT": frozenset({"insurance", "import", "destination", "unloading"}),
    "CIF": frozenset({"insurance_top_up", "import", "destination", "unloading"}),
    "CIP": frozenset({"insurance_top_up", "import", "destination", "unloading"}),
    "DAP": frozenset({"import", "unloading"}),
    "DPU": frozenset({"import"}),
    "DDP": frozenset(),
}
_ALWAYS_BUYER = frozenset({"goods", "payment"})
_CROSS_BORDER_ONLY = frozenset({"export", "import"})


def parse_incoterm(value: Any) -> Optional[str]:
    """The three-letter rule from text such as 'FOB Shanghai'; None if absent or unknown."""
    token = str(value or "").strip().upper()[:3]
    return token if token in BUYER_PAYS else None


class FxTable:
    """USD-quoted rates (units of a currency per 1 USD), as ``proc.bp_fx_rates`` stores them."""

    def __init__(self, rates: Optional[Mapping[str, Any]], as_of: Any, buyer: str):
        self.rates = {str(k).upper(): float(v) for k, v in (rates or {}).items()
                      if _positive(v)}
        self.as_of = as_of
        self.buyer = buyer
        self.used = False

    def convert(self, amount: float, currency: Optional[str]) -> Optional[float]:
        """``amount`` in ``currency`` in buyer currency, or None when no rate is known."""
        ccy = str(currency or self.buyer).upper()
        if ccy == self.buyer:
            return float(amount)
        src, dst = self.rates.get(ccy), self.rates.get(self.buyer)
        if not src or not dst:
            return None
        if self.as_of is None:
            raise ValueError("the FX table must be dated: pass fx_table_date")
        self.used = True
        return float(amount) / src * dst


def _iso(value: Any) -> Optional[str]:
    """A date or the text of one (agents pass JSON), as ISO text."""
    if not value:
        return None
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


def _positive(v: Any) -> bool:
    try:
        return float(v) > 0
    except (TypeError, ValueError):
        return False


def _num(v: Any) -> Optional[float]:
    try:
        return None if v is None or isinstance(v, bool) else float(v)
    except (TypeError, ValueError):
        return None


def _goods(lines: Sequence[Mapping[str, Any]], fx: FxTable):
    total, qty, uoms = 0.0, 0.0, set()
    for ln in lines:
        price, q = _num(ln.get("unit_price")), _num(ln.get("quantity"))
        if price is None or q is None:
            return None, None, "a line has no unit price or quantity"
        net = price * q * (1.0 - (_num(ln.get("discount_pct")) or 0.0))
        converted = fx.convert(net, ln.get("currency"))
        if converted is None:
            return None, None, f"no FX rate for {str(ln.get('currency')).upper()}"
        total += converted
        qty += q
        uoms.add(str(ln.get("uom") or "").lower())
    return total, (qty if len(uoms) <= 1 else None), None


def landed_cost(
    *, lines: Sequence[Mapping[str, Any]], buyer_currency: str, cross_border: bool,
    incoterm: Optional[str] = None, evaluation_date: Any = None,
    fx_rates: Optional[Mapping[str, Any]] = None, fx_table_date: Any = None,
    components: Sequence[Mapping[str, Any]] = (), duty_rate: Optional[float] = None,
    customs_basis: Optional[str] = None, default_insurance_rate: Optional[float] = None,
    vat_recoverable: Optional[bool] = None,
) -> Optional[Dict[str, Any]]:
    """The landed cost of a quote or PO, or ``None`` when the goods themselves cannot be priced."""
    buyer = str(buyer_currency).upper()
    fx = FxTable(fx_rates, fx_table_date, buyer)
    rule = parse_incoterm(incoterm)

    goods, qty, why = _goods(lines, fx)
    if goods is None:
        return None

    stated: Dict[str, Mapping[str, Any]] = {}
    unrecognised: List[str] = []
    for comp in components:
        key = comp.get("key")
        if key in _BY_KEY:
            stated[key] = comp
        else:
            unrecognised.append(str(key))

    out: Dict[str, Dict[str, Any]] = {
        "goods": _row("goods", "goods", STATED, goods, "net unit price x quantity, after line discounts"),
    }

    def amount_of(key: str) -> Optional[float]:
        """A resolved item's amount for use in a derivation; None if it is not known."""
        r = out.get(key)
        if r is None or r["status"] not in COUNTED:
            return None if r is None or r["status"] == MISSING else 0.0
        return r["amount"]

    for item in CHECKLIST:
        out[item.key] = _resolve(item, stated.get(item.key), rule, cross_border, fx,
                                 out, goods, duty_rate, customs_basis,
                                 default_insurance_rate, vat_recoverable, amount_of)

    rows = list(out.values())
    counted = [r for r in rows if r["status"] in COUNTED]
    total = sum(r["amount"] for r in counted)
    missing = [r["name"] for r in rows if r["status"] == MISSING]
    estimated = [r["name"] for r in rows if r["status"] == ESTIMATED]
    status = ("incomplete" if missing
              else "complete_with_estimates" if estimated else "complete")
    return {
        "currency": buyer,
        "fx_table_date": _iso(fx_table_date),
        "evaluation_date": _iso(evaluation_date),
        "incoterm": incoterm,
        "components": rows,
        "goods_total": goods,
        "landed_total": total,
        "unit_landed": total / qty if qty else None,
        "landed_status": status,
        "missing": missing,
        "estimated": estimated,
        "not_stated": [r["name"] for r in rows if r["status"] == NOT_STATED],
        "unrecognised_components": unrecognised,
    }


def _row(name: str, group: str, status: str, amount: Optional[float], basis: str,
         pay_day: Optional[float] = None) -> Dict[str, Any]:
    return {"group": group, "name": name, "status": status, "amount": amount,
            "basis": basis, "pay_day": pay_day}


def _resolve(item: Item, given: Optional[Mapping[str, Any]], rule: Optional[str],
             cross_border: bool, fx: FxTable, done: Mapping[str, Dict[str, Any]],
             goods: float, duty_rate, customs_basis, insurance_rate, vat_recoverable,
             amount_of) -> Dict[str, Any]:
    def row(status, amount, basis, pay_day=None):
        return _row(item.key, item.group, status, amount, basis, pay_day)

    # 1. What the quote or PO says.
    if given is not None:
        wanted = given.get("status")
        if wanted == INCLUDED:
            return row(INCLUDED, 0.0, given.get("basis") or "stated as included in the price")
        if wanted == NOT_APPLICABLE:
            return row(NOT_APPLICABLE, None, given.get("basis") or "stated as not applicable")
        value = _num(given.get("amount"))
        if value is not None:
            converted = fx.convert(value, given.get("currency"))
            if converted is None:
                return row(MISSING, None,
                           f"stated in {str(given.get('currency')).upper()} but the dated FX "
                           f"table has no rate for it")
            return row(STATED, converted, given.get("basis") or "on the quote or PO",
                       _num(given.get("pay_day")))

    # 2. Not owed at all.
    if item.stage in _CROSS_BORDER_ONLY and not cross_border:
        return row(NOT_APPLICABLE, None, "domestic supply: no export or import")
    if item.key == "import_vat" and vat_recoverable is True:
        return row(NOT_APPLICABLE, None, "import VAT is recoverable and excluded from landed cost")

    # 3. No Incoterm: nothing says who owes it.
    if rule is None:
        return row(MISSING, None, "no Incoterm, so who owes this is not known")

    # 4. The seller bears it, so it is inside the goods price.
    if item.stage not in _ALWAYS_BUYER and item.stage not in BUYER_PAYS[rule]:
        return row(INCLUDED, 0.0, f"{rule}: the seller bears this")

    # 5. The buyer owes it and the quote does not say what it is: derive it if a rule exists.
    if item.key == "cargo_insurance":
        freight = amount_of("freight")
        if insurance_rate is None or freight is None:
            return row(MISSING, None, "no policy insurance rate, or freight is not known")
        return row(ESTIMATED, insurance_rate * INSURED_VALUE_FACTOR * (goods + freight),
                   f"policy rate {insurance_rate:g} x 110% x (goods + freight)")
    if item.key == "import_duty":
        if duty_rate is None or customs_basis is None:
            return row(MISSING, None, "no duty rate or customs value basis")
        if str(customs_basis).upper() == "CIF":
            freight, cover = amount_of("freight"), amount_of("cargo_insurance")
            if freight is None or cover is None:
                return row(MISSING, None, "customs value is CIF but freight or insurance is not known")
            value = goods + freight + cover
        else:
            value = goods
        return row(DERIVED, duty_rate * value,
                   f"duty rate {duty_rate:g} x customs value ({str(customs_basis).upper()})")

    # 6. No amount and no rule.
    if item.required:
        return row(MISSING, None, "owed by the buyer under this Incoterm but not stated")
    return row(NOT_STATED, None, "only owed if the supplier charges it; the quote does not mention it")
