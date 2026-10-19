"""Batch 1 gateway functions. Fixed, parameterised SQL only; user text is only ever a bound parameter.

Row scope: purchase orders (and anything reached through them) and, since Batch 2, the invoice aggregate are restricted to the caller's buyer ids
(``CallerContext.buyer_scope``; ``None`` = unrestricted, an empty tuple sees nothing). Suppliers, contracts and
policies carry no buyer column that maps to ``bp_user_buyer_scope`` and are NOT row-scoped yet; that waits on
the tenant/reseller decision (spec section F, item 1).

A store that cannot be read raises, and the gateway turns that into UNAVAILABLE. It is never returned as an empty result.
"""
from __future__ import annotations

import datetime as dt
import os
from decimal import Decimal
from typing import Any, Dict, List, Literal, Optional, Tuple

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.services import ask_exact_id, ask_po_status

from .broker import CONFIG_DIR
from .common import _P, _PO_TABLE, _blind, _connect, _prov, _rows, _scope_sql
from .context import CallerContext
from .envelope import GatewayFault, Raw
from .gateway import Backends, FunctionSpec

EntityType = Literal["supplier", "customer", "reseller", "sku", "quote", "order", "po", "invoice", "contract", "rfq", "programme"]
_IMPLEMENTED = ("po", "supplier", "contract", "sku")


def _visible_po(cur: Any, ctx: CallerContext, po_id: str) -> Optional[Dict[str, Any]]:
    if _blind(ctx):
        return None
    clause, extra = _scope_sql(ctx, "buyer_id")
    rows = _rows(cur, f"SELECT po_id, supplier_name FROM proc.{_PO_TABLE} WHERE upper(po_id) = upper(%s){clause}", (po_id,) + extra)
    return rows[0] if rows else None


# --- resolve_entity -------------------------------------------------------------------------------------------------------------

class ResolveEntityParams(_P):
    text: str = Field(min_length=1, max_length=500)
    types: List[EntityType] = Field(default_factory=lambda: list(_IMPLEMENTED))
    limit: int = Field(default=5, ge=1, le=25)


def resolve_entity(ctx: CallerContext, p: ResolveEntityParams, backends: Optional[Backends]) -> Raw:
    wanted = [t for t in p.types if t in _IMPLEMENTED]
    if not wanted:
        raise GatewayFault("UNSUPPORTED", f"entity types {p.types} cannot be resolved yet; logged to the function backlog")
    connect = _connect(backends)
    ids = ask_exact_id.extract(p.text)
    found: List[Dict[str, Any]] = []
    searched: List[Dict[str, Any]] = []
    with connect() as conn, conn.cursor() as cur:
        if "po" in wanted:
            for po_id in ids.get("po", []):
                row = _visible_po(cur, ctx, po_id)
                if row:
                    found.append({"type": "po", "id": row["po_id"], "label": row["supplier_name"]})
            if ids.get("po"):
                searched.append(_prov(_PO_TABLE, [f["id"] for f in found if f["type"] == "po"]))
        if "supplier" in wanted:
            for sid in ids.get("supplier", []):
                for r in _rows(cur, "SELECT supplier_id, supplier_name FROM proc.bp_supplier WHERE supplier_id = %s", (sid,)):
                    found.append({"type": "supplier", "id": r["supplier_id"], "label": r["supplier_name"]})
            if ids.get("supplier"):
                searched.append(_prov("bp_supplier", [f["id"] for f in found if f["type"] == "supplier"]))
        if "contract" in wanted:
            for cid in ids.get("contract", []):
                rows = _rows(cur, "SELECT contract_id, contract_title FROM proc.bp_contracts WHERE upper(contract_id) = upper(%s)", (cid,))
                rows = rows or _rows(cur, "SELECT contract_id, contract_title FROM proc.bp_contract_master WHERE upper(contract_id) = upper(%s)", (cid,))
                for r in rows:
                    found.append({"type": "contract", "id": r["contract_id"], "label": r["contract_title"]})
            if ids.get("contract"):
                searched.append(_prov("bp_contracts", [f["id"] for f in found if f["type"] == "contract"]))
        if "sku" in wanted and ids.get("part") and not _blind(ctx):
            clause, extra = _scope_sql(ctx, "p.buyer_id")
            for item in ids["part"]:
                rows = _rows(cur, "SELECT l.item_id, l.item_description FROM proc.bp_po_line_items_trgt l "
                                  f"JOIN proc.{_PO_TABLE} p ON p.po_id = l.po_id WHERE upper(l.item_id) = upper(%s){clause} LIMIT 1", (item,) + extra)
                for r in rows:
                    found.append({"type": "sku", "id": r["item_id"], "label": r["item_description"]})
            searched.append(_prov("bp_po_line_items_trgt", [f["id"] for f in found if f["type"] == "sku"]))
        # No identifier in the text: try the whole text as a supplier name, exact (case-insensitive) match only.
        if not ids and "supplier" in wanted and len(p.text.strip()) <= 120:
            name = p.text.strip()
            rows = _rows(cur, "SELECT supplier_id, supplier_name FROM proc.bp_supplier WHERE lower(supplier_name) = lower(%s) "
                              "OR lower(trading_name) = lower(%s) ORDER BY supplier_id LIMIT %s", (name, name, p.limit + 1))
            found = [{"type": "supplier", "id": r["supplier_id"], "label": r["supplier_name"]} for r in rows]
            searched.append(_prov("bp_supplier", [f["id"] for f in found]))
            if len(found) > 1:
                raise GatewayFault("AMBIGUOUS", f"{len(found)} suppliers are named {name!r}", candidates=found[:p.limit])
    return Raw(data=found[:p.limit], provenance=searched or [_prov("bp_supplier", [])], total_available=len(found))


# --- get_record -----------------------------------------------------------------------------------------------------------------

class GetRecordParams(_P):
    type: Literal["po", "contract", "supplier", "part"]
    id: str = Field(min_length=1, max_length=80)


_KIND = {"po": "purchase order", "contract": "contract", "supplier": "supplier", "part": "part"}
_TABLE = {"po": _PO_TABLE, "contract": "bp_contracts", "supplier": "bp_supplier", "part": "bp_po_line_items_trgt"}


def get_record(ctx: CallerContext, p: GetRecordParams, backends: Optional[Backends]) -> Raw:
    connect = _connect(backends)
    if p.type == "part":                                      # a part is read through the POs the caller may see
        if _blind(ctx):
            return Raw(data=[], provenance=[_prov(_TABLE["part"], [])])
        clause, extra = _scope_sql(ctx, "p.buyer_id")
        with connect() as conn, conn.cursor() as cur:
            lines = _rows(cur, "SELECT l.po_id, l.line_number, l.item_id, l.item_description, l.quantity, l.unit_of_measure, l.unit_price, "
                               f"l.line_total, l.currency FROM proc.bp_po_line_items_trgt l JOIN proc.{_PO_TABLE} p ON p.po_id = l.po_id "
                               f"WHERE upper(l.item_id) = upper(%s){clause} ORDER BY l.po_id, l.line_number LIMIT %s",
                          (p.id,) + extra + (ask_exact_id.MAX_LINES + 1,))
        records = [{"kind": "part", "id": p.id, "rows": [], "lines": lines}] if lines else []
        return Raw(data=records, provenance=[_prov(_TABLE["part"], [p.id] if records else [])])
    if p.type == "po":
        with connect() as conn, conn.cursor() as cur:
            visible = _visible_po(cur, ctx, p.id)
        if not visible:                                       # absent and out-of-scope look the same, deliberately
            return Raw(data=[], provenance=[_prov(_PO_TABLE, [])])
    found = ask_exact_id.fetch(connect, {p.type: [p.id]}, strict=True)
    records = [r for r in found["records"] if r["kind"] == _KIND[p.type]]
    return Raw(data=records, provenance=[_prov(_TABLE[p.type], [r["id"] for r in records])])


# --- get_po_status --------------------------------------------------------------------------------------------------------------

class GetPoStatusParams(_P):
    po_id: str = Field(min_length=1, max_length=40)


_SOURCE_TABLE = {"Purchase order record": _PO_TABLE, "Invoices": "bp_invoice_trgt", "Goods receipts": "bp_goods_receipt_trgt",
                 "Approvals": "bp_approval"}


def get_po_status(ctx: CallerContext, p: GetPoStatusParams, backends: Optional[Backends]) -> Raw:
    connect = _connect(backends)
    with connect() as conn, conn.cursor() as cur:
        visible = _visible_po(cur, ctx, p.po_id)
    if not visible:
        return Raw(data=[], provenance=[_prov(_PO_TABLE, [])])
    results = ask_po_status.gather(connect, visible["po_id"])
    data = [{"source": name, "state": r["state"], "rows": r["rows"], "why": r["why"]} for name, r in results.items()]
    prov = [_prov(_PO_TABLE, [visible["po_id"]])] + [_prov(_SOURCE_TABLE.get(n, n), [visible["po_id"]]) for n, r in results.items()
                                                      if r["state"] == "ok" and n != "Purchase order record"]
    return Raw(data=data, provenance=prov)


# --- get_spend_summary ----------------------------------------------------------------------------------------------------------

class GetSpendSummaryParams(_P):
    period_start: dt.date
    period_end: dt.date
    limit: int = Field(default=10, ge=1, le=50)

    @model_validator(mode="after")
    def _ordered(self) -> "GetSpendSummaryParams":
        if self.period_start > self.period_end:
            raise ValueError("period_start is after period_end")
        return self


def get_spend_summary(ctx: CallerContext, p: GetSpendSummaryParams, backends: Optional[Backends]) -> Raw:
    from src.services.analytics import repository
    from src.services.analytics.period import Period
    connect = _connect(backends)
    if _blind(ctx):
        rows = []
    else:
        with connect() as conn, conn.cursor() as cur:
            rows = repository.fetch_supplier_spend(cur, Period(p.period_start, p.period_end, f"{p.period_start}..{p.period_end}"),
                                                   buyer_ids=ctx.buyer_scope)
    totals: Dict[str, Decimal] = {}
    for r in rows:
        totals[r.currency] = totals.get(r.currency, Decimal(0)) + r.amount
    ranked = sorted(rows, key=lambda r: (-r.amount, r.supplier_name))
    top = [{"supplier_id": r.supplier_id, "supplier_name": r.supplier_name, "currency": r.currency, "amount": str(r.amount),
            "invoices": r.invoices} for r in ranked[:p.limit]]
    data = {"suppliers": top, "totals_by_currency": {c: str(v) for c, v in sorted(totals.items())},
            "currencies_not_comparable": len(totals) > 1}
    return Raw(data=data, provenance=[{**_prov("bp_invoice_trgt", []), "aggregate": True,
                                       "period": [p.period_start.isoformat(), p.period_end.isoformat()]}],
               total_available=len(rows), truncated=len(rows) > p.limit)


# --- search_documents -----------------------------------------------------------------------------------------------------------

class DocFilters(_P):
    document_type: Optional[str] = Field(default=None, max_length=40)
    supplier_id: Optional[str] = Field(default=None, max_length=80)
    contract_id: Optional[str] = Field(default=None, max_length=80)


class SearchDocumentsParams(_P):
    query: str = Field(min_length=1, max_length=500)
    filters: DocFilters = Field(default_factory=DocFilters)
    top_k: int = Field(default=5, ge=1, le=10)


_TEXT_KEYS = ("content", "text", "chunk_text", "page_content")
_PASSAGE_CHARS = 800


def search_documents(ctx: CallerContext, p: SearchDocumentsParams, backends: Optional[Backends]) -> Raw:
    if backends is None or backends.search is None:
        raise GatewayFault("UNAVAILABLE", "document search is not configured for the gateway", retryable=True)
    hits = backends.search(p.query, p.top_k, p.filters.model_dump(exclude_none=True)) or []
    passages, ids = [], []
    for h in hits[:p.top_k]:
        payload = getattr(h, "payload", None) or {}
        doc_id = str(payload.get("document_id") or getattr(h, "id", "") or "")
        text = next((str(payload[k]) for k in _TEXT_KEYS if payload.get(k)), "")
        passages.append({"document_id": doc_id, "clause": payload.get("section") or payload.get("clause"),
                         "document_type": payload.get("document_type"), "text": text[:_PASSAGE_CHARS],
                         "score": getattr(h, "score", None),
                         # Rule: a number inside document text is never presented as a fact.
                         "numbers_are_document_text": True})
        if doc_id and doc_id not in ids:
            ids.append(doc_id)
    return Raw(data=passages, provenance=[_prov("documents", ids, store="vector")])


# --- get_policy -----------------------------------------------------------------------------------------------------------------

class GetPolicyParams(_P):
    policy_name: str = Field(min_length=1, max_length=120)
    limit: int = Field(default=5, ge=1, le=10)


def get_policy(ctx: CallerContext, p: GetPolicyParams, backends: Optional[Backends]) -> Raw:
    connect = _connect(backends)
    with connect() as conn, conn.cursor() as cur:
        rows = _rows(cur, "SELECT policy_id, policy_name, policy_type, policy_desc, policy_details FROM proc.bp_policy "
                          "WHERE policy_name = %s AND policy_status = 1 ORDER BY policy_id LIMIT %s", (p.policy_name, p.limit))
    return Raw(data=rows, provenance=[_prov("bp_policy", [r["policy_id"] for r in rows])])


# --- find_applicable_contracts --------------------------------------------------------------------------------------------------

class FindContractsParams(_P):
    supplier_id: str = Field(min_length=1, max_length=80)
    date: dt.date
    limit: int = Field(default=10, ge=1, le=25)


_CONTRACT_COLS = ("contract_id, contract_title, contract_type, supplier_id, contract_start_date, contract_end_date, currency, "
                  "total_contract_value, contract_lifecycle_status")


def find_applicable_contracts(ctx: CallerContext, p: FindContractsParams, backends: Optional[Backends]) -> Raw:
    connect = _connect(backends)
    seen: Dict[str, Dict[str, Any]] = {}
    with connect() as conn, conn.cursor() as cur:
        for table in ("bp_contracts", "bp_contract_master"):
            for r in _rows(cur, f"SELECT {_CONTRACT_COLS} FROM proc.{table} WHERE supplier_id = %s AND contract_start_date <= %s "
                                "AND contract_end_date >= %s ORDER BY contract_id LIMIT %s", (p.supplier_id, p.date, p.date, p.limit)):
                seen.setdefault(r["contract_id"], r)
    rows = list(seen.values())
    return Raw(data=rows, provenance=[_prov("bp_contracts", [r["contract_id"] for r in rows])])


from .approvals import (EvaluateApprovalParams, GetApprovalChainParams, evaluate_approval_status,  # noqa: E402
                        get_approval_chain)


# --- catalogue ------------------------------------------------------------------------------------------------------------------

_DECLARED = {
    "resolve_entity": (ResolveEntityParams, resolve_entity),
    "get_record": (GetRecordParams, get_record),
    "get_po_status": (GetPoStatusParams, get_po_status),
    "get_spend_summary": (GetSpendSummaryParams, get_spend_summary),
    "search_documents": (SearchDocumentsParams, search_documents),
    "get_policy": (GetPolicyParams, get_policy),
    "find_applicable_contracts": (FindContractsParams, find_applicable_contracts),
    "get_approval_chain": (GetApprovalChainParams, get_approval_chain),
    "evaluate_approval_status": (EvaluateApprovalParams, evaluate_approval_status),
}


from .batch2 import DECLARED as _BATCH2  # noqa: E402

_DECLARED.update(_BATCH2)

from .batch3 import DECLARED as _BATCH3  # noqa: E402

_DECLARED.update(_BATCH3)

from .sales import DECLARED as _SALES  # noqa: E402

_DECLARED.update(_SALES)


def build_catalogue(config_path: Optional[str] = None) -> Dict[str, FunctionSpec]:
    with open(config_path or os.path.join(CONFIG_DIR, "functions.yaml"), encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    declared = cfg.get("functions") or {}
    if set(declared) != set(_DECLARED):
        raise ValueError(f"functions.yaml and the catalogue disagree: only in yaml {sorted(set(declared) - set(_DECLARED))}, "
                         f"only in code {sorted(set(_DECLARED) - set(declared))}")
    version = int(cfg.get("version", 1))
    return {name: FunctionSpec(name=name, params=model, fn=fn, max_rows=int(declared[name]["max_rows"]),
                               kind=declared[name].get("kind", "read"), version=version,
                               allowed_roles=tuple(declared[name]["allowed_roles"]) if declared[name].get("allowed_roles") else None,
                               needs_principal=bool(declared[name].get("needs_principal", False)))
            for name, (model, fn) in _DECLARED.items()}
