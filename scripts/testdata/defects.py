"""Plant known defects and publish the ground truth.

Without an answer key a detector that finds nothing and a detector that is broken
look identical. The six negative controls matter more than the true positives:
they are what separates a working detector from one that flags everything.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Sequence

from scripts.testdata.documents import Chain, Document, LineItem
from scripts.testdata.org import CostCentre
from scripts.testdata.rng import make_rng


@dataclass(frozen=True)
class DefectSpec:
    ref: str
    name: str
    count: int
    kind: str  # "true_positive" | "negative_control"
    scenario_refs: tuple[str, ...]
    description: str


DEFECT_SPECS: tuple[DefectSpec, ...] = (
    DefectSpec("D01", "Duplicate invoice, exact resubmission", 180, "true_positive", ("A7", "B5"), "Same supplier, invoice number and total, submitted twice"),
    DefectSpec("D02", "Near-duplicate invoice", 120, "true_positive", ("B5",), "Reference differs by one character"),
    DefectSpec("D03", "PO to invoice unit-price mismatch", 340, "true_positive", ("B1",), "Invoice unit price above the matching PO line"),
    DefectSpec("D04", "PO to invoice quantity mismatch", 210, "true_positive", ("B2",), "Invoice quantity above PO quantity"),
    DefectSpec("D05", "Invoice with no purchase order", 620, "true_positive", ("B3",), "Invoice with no PO chain behind it"),
    DefectSpec("D06", "Invoice exceeds PO beyond tolerance", 260, "true_positive", ("B4",), "Invoice total above the 5% tolerance"),
    DefectSpec("D07", "Approval bypass", 145, "true_positive", ("B4", "E3"), "Above threshold with no approval record"),
    DefectSpec("D08", "Split PO to evade threshold", 75, "true_positive", ("B4",), "Sibling POs each just under threshold"),
    DefectSpec("D09", "Award not to lowest compliant quote", 400, "true_positive", ("C5",), "Award above the lowest compliant quote, unjustified"),
    DefectSpec("D10", "Wide unit-price spread for same item", 520, "true_positive", ("D1",), "Over 25% spread across suppliers"),
    DefectSpec("D11", "Tail-spend consolidation opportunity", 340, "true_positive", ("D2",), "Fragmented low-value buying"),
    DefectSpec("D12", "Single-source concentration", 160, "true_positive", ("D3",), "Over 80% of category spend, no competition"),
    DefectSpec("D13", "Expired insurance certificate", 310, "true_positive", ("D4",), "Expiry in the past on a trading supplier"),
    DefectSpec("D14", "Lapsed ESG certification", 275, "true_positive", ("D4",), "Certification absent where the contract requires it"),
    DefectSpec("D15", "Supplier near-duplicate names", 240, "true_positive", ("C1",), "Name clusters differing by spacing and case"),
    DefectSpec("D16", "Contract obligation breached", 90, "true_positive", ("D5",), "Past due with no evidence of completion"),
    DefectSpec("D17", "Contract obligation expiring soon", 130, "true_positive", ("D5",), "Due within 60 days"),
    DefectSpec("D18", "Auto-renewal notice window missed", 45, "true_positive", ("D5",), "Notice window closing within 30 days"),
    DefectSpec("D19", "Payment terms breach", 230, "true_positive", ("B4",), "Paid outside contracted terms"),
    DefectSpec("D20", "Currency and total mismatch", 95, "true_positive", ("A2", "A6"), "Line totals inconsistent with the stated total"),
    DefectSpec("D21", "Missing required extraction fields", 380, "true_positive", ("A8",), "Required field absent"),
    DefectSpec("D22", "Services line with no quantity", 800, "negative_control", ("A5", "B6"), "Lump-sum services, correct as issued"),
    DefectSpec("D23", "Legitimate credit note", 190, "negative_control", ("B6",), "Negative value correctly issued"),
    DefectSpec("D24", "Contracted price increase within index", 150, "negative_control", ("D6",), "Rise permitted by a CPI clause"),
    DefectSpec("D25", "Justified sole source", 60, "negative_control", ("C5", "D3"), "Justification and approval on file"),
    DefectSpec("D26", "Genuinely distinct near-name suppliers", 80, "negative_control", ("C1",), "Different VAT and registration numbers"),
    DefectSpec("D27", "Cost-centre budget overrun", 85, "true_positive", ("E2",), "Actual spend above allocated budget"),
    DefectSpec("D28", "Cross-entity price inconsistency", 220, "true_positive", ("E4",), "Different prices to different entities"),
    DefectSpec("D29", "Supplier onboarded separately per entity", 140, "true_positive", ("E4",), "Separate records, no group agreement"),
    DefectSpec("D30", "Justified cross-entity price difference", 180, "negative_control", ("E5",), "Explained by currency, region or volume"),
)

SPEC_BY_REF: dict[str, DefectSpec] = {spec.ref: spec for spec in DEFECT_SPECS}


@dataclass(frozen=True)
class PlantedDefect:
    ref: str
    kind: str
    subject_id: str
    subject_type: str
    detail: dict[str, Any]


@dataclass
class PlantResult:
    chains: list[Chain]
    cost_centres: list[CostCentre]
    planted: list[PlantedDefect]


def _money(value: Decimal) -> Decimal:
    return value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def _retotal(document: Document, lines: Sequence[LineItem]) -> Document:
    net = _money(sum((line.line_total for line in lines), Decimal("0")))
    tax = _money(net * Decimal("0.20"))
    return replace(
        document, lines=tuple(lines), net_total=net, tax_amount=tax,
        gross_total=_money(net + tax),
    )


def _target_count(spec: DefectSpec, available: int) -> int:
    """Scale a spec's count down when the fixture is smaller than a full build."""
    return min(spec.count, available)


def plant(
    seed: int, chains: list[Chain], cost_centres: list[CostCentre]
) -> PlantResult:
    """Mutate chains and cost centres to plant defects. Returns the ground truth."""
    rng = make_rng(seed, "defects")
    chains = list(chains)
    cost_centres = list(cost_centres)
    planted: list[PlantedDefect] = []

    # Every planting block replaces the chain it touches with a new frozen copy,
    # so a chain captured in an earlier list is no longer the object sitting in
    # `chains`. Look positions up by requirement_id -- which never changes --
    # rather than searching the list for an object that has since been replaced.
    position_of = {chain.requirement_id: index for index, chain in enumerate(chains)}

    # --- D05: invoices with no purchase order --------------------------------
    no_po = [
        chain for chain in chains
        if chain.purchase_order is None and chain.invoices
    ]
    for chain in no_po[: _target_count(SPEC_BY_REF["D05"], len(no_po))]:
        for invoice in chain.invoices:
            planted.append(
                PlantedDefect(
                    ref="D05", kind="true_positive", subject_id=invoice.doc_id,
                    subject_type="invoice",
                    detail={
                        "supplier_id": invoice.supplier_id,
                        "org_id": invoice.org_id,
                        "net_total": float(invoice.net_total),
                    },
                )
            )

    with_po = [chain for chain in chains if chain.purchase_order is not None and chain.invoices]

    # --- D03: invoice unit price above the PO line ---------------------------
    for index in range(_target_count(SPEC_BY_REF["D03"], len(with_po))):
        position = position_of[with_po[index].requirement_id]
        chain = chains[position]
        invoice = chain.invoices[0]
        uplift = Decimal(str(round(rng.uniform(1.03, 1.40), 4)))

        original = invoice.lines[0]
        inflated_price = _money(original.unit_price * uplift)
        inflated = replace(
            original,
            unit_price=inflated_price,
            line_total=_money(original.quantity * inflated_price),
        )
        new_lines = (inflated, *invoice.lines[1:])
        updated_invoice = _retotal(invoice, new_lines)

        chains[position] = replace(
            chain, invoices=(updated_invoice, *chain.invoices[1:])
        )
        planted.append(
            PlantedDefect(
                ref="D03", kind="true_positive", subject_id=updated_invoice.doc_id,
                subject_type="invoice",
                detail={
                    "po_doc_id": chain.purchase_order.doc_id,
                    "po_unit_price": float(original.unit_price),
                    "invoice_unit_price": float(inflated_price),
                    "delta_gbp": float(inflated.line_total - original.line_total),
                },
            )
        )

    # --- D04: invoice quantity above PO quantity -----------------------------
    # Runs before D22 nulls quantities, and still picks the first line that
    # carries one, so a lump-sum line elsewhere in the document cannot silence it.
    for index in range(_target_count(SPEC_BY_REF["D04"], len(with_po))):
        position = position_of[with_po[index].requirement_id]
        chain = chains[position]
        invoice = chain.invoices[0]
        target = next(
            (
                line for line in invoice.lines
                if line.quantity is not None and line.unit_price is not None
            ),
            None,
        )
        if target is None:
            continue
        extra = Decimal(str(rng.randint(1, 6)))
        inflated = replace(
            target,
            quantity=target.quantity + extra,
            line_total=_money((target.quantity + extra) * target.unit_price),
        )
        new_lines = tuple(
            inflated if line.line_number == target.line_number else line
            for line in invoice.lines
        )
        updated = _retotal(invoice, new_lines)
        chains[position] = replace(chain, invoices=(updated, *chain.invoices[1:]))
        planted.append(
            PlantedDefect(
                ref="D04", kind="true_positive", subject_id=updated.doc_id,
                subject_type="invoice",
                detail={
                    "po_doc_id": chain.purchase_order.doc_id,
                    "po_quantity": float(target.quantity),
                    "invoice_quantity": float(inflated.quantity),
                },
            )
        )

    # --- D06: invoice total above the 5% PO tolerance ------------------------
    for index in range(_target_count(SPEC_BY_REF["D06"], len(with_po))):
        position = position_of[with_po[index].requirement_id]
        chain = chains[position]
        invoice = chain.invoices[0]
        factor = Decimal(str(round(rng.uniform(1.06, 1.22), 4)))
        scaled = tuple(
            replace(line, line_total=_money(line.line_total * factor))
            for line in invoice.lines
        )
        updated = _retotal(invoice, scaled)
        chains[position] = replace(chain, invoices=(updated, *chain.invoices[1:]))
        planted.append(
            PlantedDefect(
                ref="D06", kind="true_positive", subject_id=updated.doc_id,
                subject_type="invoice",
                detail={
                    "po_net_total": float(chain.purchase_order.net_total),
                    "invoice_net_total": float(updated.net_total),
                    "tolerance": 0.05,
                },
            )
        )

    # --- D01: exact duplicate invoices ---------------------------------------
    for index in range(_target_count(SPEC_BY_REF["D01"], len(with_po))):
        position = position_of[with_po[index].requirement_id]
        chain = chains[position]
        original = chain.invoices[0]
        duplicate = replace(original, doc_id=f"{original.doc_id}-DUP")
        chains[position] = replace(chain, invoices=(*chain.invoices, duplicate))
        planted.append(
            PlantedDefect(
                ref="D01", kind="true_positive", subject_id=duplicate.doc_id,
                subject_type="invoice",
                detail={
                    "original_doc_id": original.doc_id,
                    "duplicate_doc_id": duplicate.doc_id,
                    "invoice_number": original.doc_id,
                    "net_total": float(original.net_total),
                },
            )
        )

    # --- D02: near-duplicate invoices ----------------------------------------
    for index in range(_target_count(SPEC_BY_REF["D02"], len(with_po))):
        position = position_of[with_po[index].requirement_id]
        chain = chains[position]
        original = chain.invoices[0]
        near = replace(original, doc_id=f"{original.doc_id}A")
        chains[position] = replace(chain, invoices=(*chain.invoices, near))
        planted.append(
            PlantedDefect(
                ref="D02", kind="true_positive", subject_id=near.doc_id,
                subject_type="invoice",
                detail={
                    "original_doc_id": original.doc_id,
                    "net_total": float(original.net_total),
                },
            )
        )

    # --- D22 (negative): lump-sum services lines with no quantity ------------
    for index in range(_target_count(SPEC_BY_REF["D22"], len(chains))):
        chain = chains[index]
        if not chain.invoices:
            continue
        invoice = chain.invoices[0]
        original = invoice.lines[0]
        lump_sum = replace(
            original,
            quantity=None,
            unit_price=None,
            description=f"{original.description} (lump sum, services)",
        )
        updated = _retotal(invoice, (lump_sum, *invoice.lines[1:]))
        chains[index] = replace(chain, invoices=(updated, *chain.invoices[1:]))
        planted.append(
            PlantedDefect(
                ref="D22", kind="negative_control", subject_id=updated.doc_id,
                subject_type="invoice",
                detail={
                    "line_number": lump_sum.line_number,
                    "line_total": float(lump_sum.line_total),
                    "reason": "Lump-sum services legitimately carry no qty or unit price",
                },
            )
        )

    # --- D23 (negative): legitimate credit notes -----------------------------
    for index in range(_target_count(SPEC_BY_REF["D23"], len(with_po))):
        position = position_of[with_po[index].requirement_id]
        chain = chains[position]
        source = chain.invoices[0]
        credit_lines = tuple(
            replace(
                line,
                quantity=None if line.quantity is None else -line.quantity,
                line_total=-line.line_total,
            )
            for line in source.lines
        )
        credit = _retotal(
            replace(source, doc_id=f"{source.doc_id}-CN", parent_doc_id=source.doc_id),
            credit_lines,
        )
        chains[position] = replace(chain, invoices=(*chain.invoices, credit))
        planted.append(
            PlantedDefect(
                ref="D23", kind="negative_control", subject_id=credit.doc_id,
                subject_type="invoice",
                detail={
                    "credits_doc_id": source.doc_id,
                    "net_total": float(credit.net_total),
                    "reason": "Credit note correctly issued against an earlier invoice",
                },
            )
        )

    # --- D27: cost-centre budget overruns ------------------------------------
    for index in range(_target_count(SPEC_BY_REF["D27"], len(cost_centres))):
        centre = cost_centres[index]
        overrun_factor = round(rng.uniform(1.05, 1.62), 4)
        overspent = round(centre.budget_allocated_annual * overrun_factor, 2)
        cost_centres[index] = replace(
            centre,
            actual_spend_ytd=overspent,
            forecast_spend_annual=round(overspent * 1.08, 2),
        )
        planted.append(
            PlantedDefect(
                ref="D27", kind="true_positive", subject_id=centre.cc_id,
                subject_type="cost_centre",
                detail={
                    "budget_allocated_annual": centre.budget_allocated_annual,
                    "actual_spend_ytd": overspent,
                    "overrun_gbp": round(overspent - centre.budget_allocated_annual, 2),
                    "manager_email": centre.manager_email,
                },
            )
        )

    return PlantResult(chains=chains, cost_centres=cost_centres, planted=planted)


def write_answer_key(result: PlantResult, json_path: Path, md_path: Path) -> None:
    """Write the machine-readable and human-readable ground truth."""
    by_ref: dict[str, list[dict[str, Any]]] = {}
    for item in result.planted:
        by_ref.setdefault(item.ref, []).append(
            {
                "subject_id": item.subject_id,
                "subject_type": item.subject_type,
                "detail": item.detail,
            }
        )

    payload = {
        "total_planted": len(result.planted),
        "specs": [
            {
                "ref": spec.ref,
                "name": spec.name,
                "kind": spec.kind,
                "target_count": spec.count,
                "planted_count": len(by_ref.get(spec.ref, [])),
                "scenario_refs": list(spec.scenario_refs),
            }
            for spec in DEFECT_SPECS
        ],
        "by_ref": by_ref,
    }

    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    lines = [
        "# Test Dataset Answer Key",
        "",
        f"{len(result.planted)} instances planted across {len(by_ref)} defect types.",
        "",
        "Negative controls are the important half: any finding against them is a",
        "false positive, and the scenario fails.",
        "",
        "| Ref | Defect | Kind | Target | Planted | Test cases |",
        "|---|---|---|---|---|---|",
    ]
    for spec in DEFECT_SPECS:
        lines.append(
            f"| {spec.ref} | {spec.name} | {spec.kind} | {spec.count} | "
            f"{len(by_ref.get(spec.ref, []))} | {', '.join(spec.scenario_refs)} |"
        )
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines) + "\n")
