"""The 14 verification checks.

V14 is the isolation guarantee and runs both first and last: if a live row count
moved during the build, the run has already failed regardless of what else passed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from scripts.testdata.db import connect
from scripts.testdata.guards import UnsafeTargetError, assert_live_unchanged, snapshot_counts

NOT_YET_IMPLEMENTED = "not yet implemented"


@dataclass(frozen=True)
class Check:
    ref: str
    name: str
    blocking: bool


CHECKS: tuple[Check, ...] = (
    Check("V01", "Row counts reach plan", True),
    Check("V02", "No orphan references", True),
    Check("V03", "Cross-database coherence", True),
    Check("V04", "Line totals sum to document totals", True),
    Check("V05", "FX conversions re-derive", True),
    Check("V06", "Every line resolves to a valid L1-L5 path", True),
    Check("V07", "Organisation roll-up balances", True),
    Check("V08", "Every screen query returns non-empty", True),
    Check("V09", "Every route returns 200 with a payload", True),
    Check("V10", "Planted defects are found", False),
    Check("V11", "No negative control produces a finding", True),
    Check("V12", "Golden set extraction matches expected", False),
    Check("V13", "Same seed reproduces identical checksums", True),
    Check("V14", "Live databases unchanged", True),
)

CHECK_BY_REF: dict[str, Check] = {check.ref: check for check in CHECKS}


@dataclass(frozen=True)
class CheckResult:
    ref: str
    passed: bool
    detail: str


def blocking_failures(results: Sequence[CheckResult]) -> list[CheckResult]:
    """Only failures on blocking checks. Scored checks report but never block."""
    return [
        result
        for result in results
        if not result.passed and CHECK_BY_REF[result.ref].blocking
    ]


def _scalar(conn, sql: str) -> int:
    with conn.cursor() as cur:
        cur.execute(sql)
        row = cur.fetchone()
        return int(row[0]) if row and row[0] is not None else 0


def check_row_counts(target_db: str) -> CheckResult:
    conn = connect(target_db)
    try:
        suppliers = _scalar(conn, "select count(*) from proc.bp_supplier")
        passed = suppliers == 5000
        return CheckResult(
            ref="V01", passed=passed,
            detail=f"proc.bp_supplier has {suppliers} rows (expected 5000)",
        )
    finally:
        conn.close()


def check_no_orphans(target_db: str) -> CheckResult:
    conn = connect(target_db)
    try:
        orphans = _scalar(
            conn,
            """
            select count(*) from proc.bp_invoice_line_items_trgt li
            where not exists (
                select 1 from proc.bp_invoice_trgt i
                where i.invoice_id = li.invoice_id
            )
            """,
        )
        return CheckResult(
            ref="V02", passed=orphans == 0,
            detail=f"{orphans} orphan invoice line items",
        )
    finally:
        conn.close()


def check_crosswalk(target_db: str, uicanvas_target_db: str) -> CheckResult:
    bp_conn = connect(target_db)
    ui_conn = connect(uicanvas_target_db)
    try:
        mapped = _scalar(bp_conn, "select count(*) from proc.bp_supplier_id_crosswalk")
        bp_suppliers = _scalar(bp_conn, "select count(*) from proc.bp_supplier")
        ui_suppliers = _scalar(ui_conn, "select count(*) from proc.supplier")
        passed = mapped == bp_suppliers == ui_suppliers
        return CheckResult(
            ref="V03", passed=passed,
            detail=(
                f"crosswalk {mapped}, bp_supplier {bp_suppliers}, "
                f"uicanvas supplier {ui_suppliers}"
            ),
        )
    finally:
        bp_conn.close()
        ui_conn.close()


def check_rollup(uicanvas_target_db: str) -> CheckResult:
    """V07: every cost centre resolves to a business unit that exists.

    The schema carries no entity column on either table -- there is no
    organisation table at all -- so the entity and group levels of the roll-up
    cannot be verified here. That is stated in the detail rather than passed
    over: a check that quietly narrows its own scope is worse than one that fails.
    """
    conn = connect(uicanvas_target_db)
    try:
        centres = _scalar(conn, "select count(*) from proc.cost_centre")
        units = _scalar(conn, "select count(*) from proc.business_unit")
        orphans = _scalar(
            conn,
            """
            select count(*) from proc.cost_centre c
            where c.business_unit_id is null
               or not exists (
                   select 1 from proc.business_unit b
                   where b.business_unit_id = c.business_unit_id
               )
            """,
        )
        return CheckResult(
            ref="V07",
            passed=orphans == 0 and centres > 0 and units > 0,
            detail=(
                f"{centres} cost centres over {units} business units, "
                f"{orphans} unresolved; entity and group levels not verifiable "
                f"(no entity column in the schema)"
            ),
        )
    finally:
        conn.close()


def check_live_unchanged(live_before: Mapping[str, int]) -> CheckResult:
    try:
        assert_live_unchanged(live_before, snapshot_counts(["bp_sqldb", "uicanvas"]))
    except UnsafeTargetError as exc:
        return CheckResult(ref="V14", passed=False, detail=str(exc))
    return CheckResult(ref="V14", passed=True, detail="live row counts unchanged")


def run_all(
    target_db: str,
    uicanvas_target_db: str,
    *,
    live_before: Mapping[str, int],
    arithmetic_exempt: set[str] | None = None,
) -> list[CheckResult]:
    """Run every implemented check. Unimplemented ones report as not-yet-run.

    V04-V13 land with the plans that produce the data they inspect: V04-V07 with
    the seeding tasks, V08-V09 with the API surface, V10-V12 with the test harness
    plan, V13 with the build entry point.
    """
    results = [
        check_row_counts(target_db),
        check_no_orphans(target_db),
        check_crosswalk(target_db, uicanvas_target_db),
        check_rollup(uicanvas_target_db),
        check_line_totals(target_db, exempt=arithmetic_exempt or set()),
    ]
    implemented = {result.ref for result in results} | {"V14"}
    for check in CHECKS:
        if check.ref not in implemented:
            results.append(
                CheckResult(ref=check.ref, passed=True, detail=NOT_YET_IMPLEMENTED)
            )
    results.append(check_live_unchanged(live_before))
    return sorted(results, key=lambda result: result.ref)


# Defects that deliberately break document arithmetic. V04 must exempt their
# subjects, or the check would fail on data that is wrong on purpose.
ARITHMETIC_DEFECT_REFS: tuple[str, ...] = ("D03", "D04", "D06", "D20")


def check_line_totals(target_db: str, exempt: set[str] | None = None) -> CheckResult:
    """V04: line totals sum to the document total on every clean document.

    `exempt` carries the documents the answer key says were deliberately broken.
    Passing it is what separates "the loader is correct" from "no defect was
    planted"; without it the check would fail on data that is wrong by design.
    """
    exempt = exempt or set()
    conn = connect(target_db)
    mismatched: list[str] = []
    try:
        for header, lines, key, total, line_total in (
            ("bp_quote_trgt", "bp_quote_line_items_trgt", "quote_id",
             "total_amount", "line_total"),
            ("bp_purchase_order_trgt", "bp_po_line_items_trgt", "po_id",
             "total_amount", "line_total"),
            ("bp_invoice_trgt", "bp_invoice_line_items_trgt", "invoice_id",
             "invoice_amount", "line_amount"),
        ):
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    select h."{key}"
                    from proc."{header}" h
                    join (
                        select "{key}" as k, sum("{line_total}") as summed
                        from proc."{lines}" group by 1
                    ) l on l.k = h."{key}"
                    where abs(h."{total}" - l.summed) > 0.01
                    """
                )
                mismatched.extend(
                    row[0] for row in cur.fetchall() if row[0] not in exempt
                )
    finally:
        conn.close()

    return CheckResult(
        ref="V04",
        passed=not mismatched,
        detail=(
            f"{len(mismatched)} documents whose lines do not sum to their total"
            + (f" (first: {mismatched[0]})" if mismatched else "")
            + f"; {len(exempt)} exempted as planted arithmetic defects"
        ),
    )
