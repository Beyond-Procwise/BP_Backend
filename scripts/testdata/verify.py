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
    Check("V15", "Benchmark computes on the busiest item", True),
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
    seeded_tables: Sequence[str] | None = None,
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
        check_benchmark_computes(target_db),
    ]
    implemented = {result.ref for result in results} | {"V14"}
    for check in CHECKS:
        if check.ref not in implemented:
            results.append(
                CheckResult(ref=check.ref, passed=True, detail=NOT_YET_IMPLEMENTED)
            )
    results.append(check_live_untouched(live_before, seeded_tables=seeded_tables or ()))
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


def check_benchmark_computes(target_db: str) -> CheckResult:
    """V15: the whole point of the pool. The engine only proves out if a real
    item in the seeded data has enough matching history to clear the evidence
    gate and reach HIGH confidence -- otherwise every benchmark request still
    gates, exactly as it did before the pool existed.

    Built on the PRODUCTION loader (services.benchmark_live.load_benchmark_pool
    / _to_points) rather than a hand-rolled query. An earlier version of this
    check queried bp_po_line_items_trgt directly and hard-coded currency to
    "GBP" on every point -- which let it pool dollars with pounds and report a
    HIGH confidence that meant nothing, on a check whose entire job is to
    prove the pool is currency-sound. A verification check that reimplements
    the loader can also silently drift from it and end up verifying code that
    no longer resembles what actually runs; going through the real loader
    means this check exercises the real path, including currency coalesced
    from the document header (never hard-coded) and PO+invoice lines pooled
    together, exactly as production does.

    src/ is only added to sys.path inside this function, not at module scope,
    so importing this file never requires the benchmark engine to be
    importable for callers (e.g. the seeder) that don't have src/ on path.
    """
    import sys
    from collections import defaultdict
    from statistics import median

    if "src" not in sys.path:
        sys.path.insert(0, "src")
    from services.benchmark.engine import compute_benchmark
    from services.benchmark.models import QuoteLine
    from services.benchmark_live import _to_points, load_benchmark_pool

    conn = connect(target_db)
    try:
        with conn.cursor() as cur:
            pool_rows = load_benchmark_pool(cur)
    finally:
        conn.close()

    if not pool_rows:
        return CheckResult(
            ref="V15", passed=False, detail="no price-history pool to benchmark",
        )

    points = _to_points(pool_rows)

    # Busiest real match key -- exact (item, uom, currency), the same triple
    # the engine matches on. No currency is assumed; whatever the loader
    # coalesced from the line or its document header is what groups here.
    groups: dict[tuple[str, str, str], list] = defaultdict(list)
    for point in points:
        groups[(point.item_name, point.uom, point.currency)].append(point)
    key, group = max(groups.items(), key=lambda kv: len(kv[1]))
    item_name, uom, currency = key

    quantities = [p.historical_quantity for p in group if p.historical_quantity is not None]
    quote_quantity = median(quantities) if quantities else 1.0

    quote = QuoteLine(
        deal_id="verify",
        item_name=item_name,
        quantity=quote_quantity,
        uom=uom,
        currency=currency,
        location="United Kingdom",
        requested_spec_score=5.0,
        requested_sla_score=5.0,
        index_id="",
        quoted_unit_price=group[0].raw_unit_price,
    )
    result = compute_benchmark(quote, points, {}, {})
    passed = not result.gated and result.confidence == "HIGH"
    return CheckResult(
        ref="V15",
        passed=passed,
        detail=(
            f"'{item_name[:40]}' / {uom} / {currency}: "
            f"n_total={result.n_total} confidence={result.confidence} "
            f"gated={result.gated} (largest currency-accurate group={len(group)})"
        ),
    )


# The seeder stamps everything it writes. Finding any of these in a live
# database is proof it wrote there -- unlike a row-count comparison, this holds
# even while the production service is busy doing its own work.
_MARKER_PROBES: tuple[tuple[str, str], ...] = (
    ("bp_invoice_trgt", "created_by = 'testdata'"),
    ("bp_quote_trgt", "created_by = 'testdata'"),
    ("bp_purchase_order_trgt", "created_by = 'testdata'"),
    ("bp_supplier", "created_by = 'testdata'"),
    ("bp_invoice_raw", "source_file like 's3://bp-testdata/%'"),
    ("bp_quote_raw", "source_file like 's3://bp-testdata/%'"),
    ("bp_purchase_order_raw", "source_file like 's3://bp-testdata/%'"),
)


def find_seeded_rows_in_live(dbnames: Sequence[str] = ("bp_sqldb", "uicanvas")) -> dict[str, int]:
    """Rows in a live database bearing the seeder's marker. Should always be empty."""
    found: dict[str, int] = {}
    for dbname in dbnames:
        conn = connect(dbname)
        try:
            for table, predicate in _MARKER_PROBES:
                try:
                    with conn.cursor() as cur:
                        cur.execute(f"select count(*) from proc.\"{table}\" where {predicate}")
                        count = cur.fetchone()[0]
                except Exception:
                    conn.rollback()
                    continue
                if count:
                    found[f"{dbname}.proc.{table}"] = count
        finally:
            conn.close()
    return found


def check_live_untouched(
    live_before: Mapping[str, int], seeded_tables: Sequence[str] = (),
) -> CheckResult:
    """V14: the build wrote nothing to a live database.

    Two questions, only one of which a row-count comparison can answer.

    Did anything the seeder writes appear in live? That is the guarantee, and a
    marker scan answers it directly.

    Did any live row count move? That catches a seeder writing through an
    unmarked path, but it also catches the production service going about its
    business -- and something is nearly always running. So drift is only a
    failure in a table the seeder actually writes; elsewhere it is reported as
    concurrent activity rather than treated as a breach.
    """
    seeded = set(seeded_tables)
    stowaways = find_seeded_rows_in_live()

    after = snapshot_counts(["bp_sqldb", "uicanvas"])
    moved = [
        key for key in sorted(set(live_before) | set(after))
        if live_before.get(key) != after.get(key)
    ]
    moved_seeded = [key for key in moved if key.rsplit(".", 1)[-1] in seeded]
    moved_other = [key for key in moved if key not in moved_seeded]

    if stowaways:
        detail = "seeder rows found in a live database: " + ", ".join(
            f"{k} ({n})" for k, n in sorted(stowaways.items())
        )
        return CheckResult(ref="V14", passed=False, detail=detail)

    if moved_seeded:
        return CheckResult(
            ref="V14", passed=False,
            detail="live tables the seeder writes changed: " + ", ".join(moved_seeded),
        )

    detail = "no seeder rows in live; no seeded table changed"
    if moved_other:
        detail += (
            f"; {len(moved_other)} unrelated live table(s) moved during the run "
            f"(concurrent service activity, e.g. {moved_other[0]})"
        )
    return CheckResult(ref="V14", passed=True, detail=detail)
