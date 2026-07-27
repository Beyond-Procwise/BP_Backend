"""Build the test dataset.

    .venv/bin/python -m scripts.testdata.build --target bp_testdb --seed 42

Exit codes: 0 success, 1 blocking verification failure, 2 refused target.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from scripts.testdata import (
    persist, persist_org, persist_suppliers, persist_tiers, profiles, verify,
)
from scripts.testdata.catalogue import build_catalogue
from scripts.testdata.loader import load_tables
from scripts.testdata.db import copy_rows, connect
from scripts.testdata.defects import plant, write_answer_key
from scripts.testdata.documents import build_chains
from scripts.testdata.guards import UnsafeTargetError, assert_safe_target, snapshot_counts
from scripts.testdata.org import build_business_units, build_cost_centres
from scripts.testdata.reference import copy_reference, load_fx_rates, load_taxonomy
from scripts.testdata.schema import clone_schema
from scripts.testdata.suppliers import (
    CROSSWALK_DDL,
    SUPPLIER_COLUMNS,
    build_crosswalk,
    build_suppliers,
)

ROOT = Path(__file__).resolve().parents[2]
ANSWER_KEY_JSON = ROOT / "docs" / "testdata" / "answer-key.json"
ANSWER_KEY_MD = ROOT / "docs" / "testdata" / "answer-key.md"


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="scripts.testdata.build")
    parser.add_argument("--target", default="bp_testdb")
    parser.add_argument("--uicanvas-target", default="uicanvas_test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--drop-first", action="store_true")
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument(
        "--profile", default="test", choices=sorted(profiles.PROFILES),
        help=(
            "test: every defect planted with an answer key, for measuring the "
            "detectors. demo: a healthy estate with nothing planted, so a "
            "document added afterwards is the only thing that raises a finding."
        ),
    )
    return parser.parse_args(list(argv))


def _write_suppliers(target_db: str, uicanvas_target_db: str, suppliers) -> None:
    """Both ID conventions, plus the crosswalk that joins them.

    uicanvas.proc.supplier carries the same 51 columns as bp_sqldb.proc.bp_supplier
    and differs only in the supplier_id convention, so the same rows are written
    to both with the identifier swapped.
    """
    conn = connect(target_db)
    try:
        with conn.cursor() as cur:
            cur.execute(CROSSWALK_DDL)
            cur.execute("truncate proc.bp_supplier")
            cur.execute("truncate proc.bp_supplier_id_crosswalk")
        conn.commit()

        copy_rows(
            conn, "proc", "bp_supplier", list(SUPPLIER_COLUMNS),
            [[s.columns[c] for c in SUPPLIER_COLUMNS] for s in suppliers],
        )
        copy_rows(
            conn, "proc", "bp_supplier_id_crosswalk",
            ["bp_supplier_id", "uicanvas_supplier_id", "legal_entity_key"],
            build_crosswalk(suppliers),
        )
        conn.commit()
    finally:
        conn.close()

    ui_conn = connect(uicanvas_target_db)
    try:
        with ui_conn.cursor() as cur:
            cur.execute("truncate proc.supplier")
        ui_conn.commit()
        copy_rows(
            ui_conn, "proc", "supplier", list(SUPPLIER_COLUMNS),
            [
                [
                    s.uicanvas_supplier_id if column == "supplier_id"
                    else s.columns[column]
                    for column in SUPPLIER_COLUMNS
                ]
                for s in suppliers
            ],
        )
        ui_conn.commit()
    finally:
        ui_conn.close()


def _assign_deals_impl(target_db: str) -> dict:
    """Indirection so the failure path is testable without a linking engine.

    Calls the linking passes directly rather than assign_deals(). assign_deals
    finishes by calling sync_deal_summaries(), which opens its OWN connection
    from the environment DSN — pointing at LIVE bp_sqldb no matter which
    connection we hand in. That would generate summaries against live deals and
    write them to live proc.bp_summary, breaking the isolation guarantee and
    failing check V14. The seeder wants deal grouping, not AI summaries.

    Imported lazily: the seeder must stay runnable when the application's
    dependencies are not importable, and the import is only needed here.
    """
    from scripts.testdata.db import connect as _connect
    from src.services.deal_assignment_service import _run

    conn = _connect(target_db)
    try:
        conn.autocommit = False
        try:
            result = _run(conn.cursor())
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        return result
    finally:
        conn.close()


def assign_deals_on(target_db: str) -> dict:
    """Group the seeded documents into deals using the production service.

    The seeder never stamps deal_id itself. If the service cannot group
    synthetic chains that is a finding about the grouping service, reported
    here and in the build log rather than papered over.
    """
    try:
        return _assign_deals_impl(target_db)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    try:
        assert_safe_target(args.target)
        assert_safe_target(args.uicanvas_target)
    except UnsafeTargetError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    profile = profiles.get(args.profile)
    print(f"profile: {profile.name} -- {profile.purpose}")

    live_before = snapshot_counts(["bp_sqldb", "uicanvas"])

    print(f"cloning schema into {args.target} and {args.uicanvas_target}")
    clone_schema("bp_sqldb", args.target, drop_first=args.drop_first)
    clone_schema("uicanvas", args.uicanvas_target, drop_first=args.drop_first)

    print("copying reference data")
    copy_reference("bp_sqldb", args.target)
    copy_reference("uicanvas", args.uicanvas_target)

    print("building data")
    leaves = load_taxonomy("uicanvas")
    suppliers = build_suppliers(args.seed, leaves)
    units = build_business_units(args.seed)
    centres = build_cost_centres(args.seed, units, leaves)
    items = build_catalogue(args.seed, leaves, [s.bp_supplier_id for s in suppliers])
    fx = load_fx_rates("bp_sqldb")
    print(f"  FX snapshot: {len(fx)} currencies")
    chains = build_chains(
        args.seed, suppliers, items, centres, fx=fx, count=6000,
        no_po_share=profile.no_po_share,
    )
    print(
        f"  {len(suppliers)} suppliers, {len(units)} business units, "
        f"{len(centres)} cost centres, {len(items)} catalogue items, "
        f"{len(chains)} document chains"
    )

    print("planting defects")
    result = plant(args.seed, chains, centres, only=profile.plant_refs)
    write_answer_key(result, ANSWER_KEY_JSON, ANSWER_KEY_MD)
    print(f"  {len(result.planted)} defect instances -> {ANSWER_KEY_JSON}")

    print("writing suppliers and crosswalk")
    _write_suppliers(args.target, args.uicanvas_target, suppliers)

    print("loading organisation and catalogue")
    org_rows = persist_org.rows_for_org(units, centres, items)
    loaded = load_tables(
        args.uicanvas_target,
        persist_org.COLUMNS,
        persist_org.REQUIRED,
        org_rows,
        # Cost centres reference business units, so the units must land first.
        order=("business_unit", "cost_centre", "item"),
    )
    for table, count in loaded.items():
        print(f"  proc.{table}: {count} rows")

    print("loading supplier reference data")
    supplier_rows = persist_suppliers.rows_for_suppliers(suppliers)
    target = persist_suppliers.TARGET_TABLE
    for db_name, keys in (
        (args.uicanvas_target, ("bp_supplier_uicanvas", "esg_data", "contact", "bp_contact")),
        (args.target, ("bp_tprm_supplier",)),
    ):
        written = load_tables(
            db_name,
            {target[key]: persist_suppliers.COLUMNS[key] for key in keys},
            {target[key]: persist_suppliers.REQUIRED[key] for key in keys},
            {target[key]: supplier_rows[key] for key in keys},
        )
        for table, count in written.items():
            print(f"  proc.{table}: {count} rows")

    print("loading raw and staging tiers")
    tier_rows = persist_tiers.rows_for_tiers(result.chains)
    written = load_tables(
        args.target,
        persist_tiers.COLUMNS,
        persist_tiers.REQUIRED,
        tier_rows,
        order=persist_tiers.LOAD_ORDER,
    )
    print(f"  {sum(written.values())} rows across {len(written)} tables")

    print("loading documents")
    document_rows = persist.rows_for(result.chains)
    document_rows["bp_requirement"] = persist.requirement_rows(result.chains)
    written = load_tables(
        args.target,
        {**persist.COLUMNS, "bp_requirement": persist.REQUIREMENT_COLUMNS},
        {**persist.REQUIRED, "bp_requirement": persist.REQUIREMENT_REQUIRED},
        document_rows,
        order=("bp_requirement", *persist.LOAD_ORDER),
    )
    for table, count in written.items():
        print(f"  proc.{table}: {count} rows")

    print("assigning deals")
    deal_result = assign_deals_on(args.target)
    if "error" in deal_result:
        print(f"  DEAL ASSIGNMENT FAILED: {deal_result['error']}", file=sys.stderr)
        print("  documents are loaded; by-deal benchmark queries will return nothing")
    else:
        print(f"  {deal_result}")

    if args.skip_verify:
        print("verification skipped")
        return 0

    # Documents carrying a planted arithmetic defect are wrong on purpose, so
    # V04 must not count them against the loader.
    arithmetic_exempt = {
        item.subject_id
        for item in result.planted
        if item.ref in verify.ARITHMETIC_DEFECT_REFS
    }

    print("verifying")
    results = verify.run_all(
        args.target, args.uicanvas_target,
        live_before=live_before,
        arithmetic_exempt=arithmetic_exempt,
    )
    for item in results:
        if item.detail == verify.NOT_YET_IMPLEMENTED:
            status = "SKIP"
        else:
            status = "PASS" if item.passed else "FAIL"
        print(f"  {item.ref} {status}  {item.detail}")

    failures = verify.blocking_failures(results)
    if failures:
        print(f"\n{len(failures)} blocking check(s) failed", file=sys.stderr)
        return 1

    print("\nbuild complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
