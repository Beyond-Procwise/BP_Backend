"""Build the test dataset.

    .venv/bin/python -m scripts.testdata.build --target bp_testdb --seed 42

Exit codes: 0 success, 1 blocking verification failure, 2 refused target.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from scripts.testdata import verify
from scripts.testdata.catalogue import build_catalogue
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


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    try:
        assert_safe_target(args.target)
        assert_safe_target(args.uicanvas_target)
    except UnsafeTargetError as exc:
        print(str(exc), file=sys.stderr)
        return 2

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
    chains = build_chains(args.seed, suppliers, items, centres, fx=fx, count=6000)
    print(
        f"  {len(suppliers)} suppliers, {len(units)} business units, "
        f"{len(centres)} cost centres, {len(items)} catalogue items, "
        f"{len(chains)} document chains"
    )

    print("planting defects")
    result = plant(args.seed, chains, centres)
    write_answer_key(result, ANSWER_KEY_JSON, ANSWER_KEY_MD)
    print(f"  {len(result.planted)} defect instances -> {ANSWER_KEY_JSON}")

    print("writing suppliers and crosswalk")
    _write_suppliers(args.target, args.uicanvas_target, suppliers)

    if args.skip_verify:
        print("verification skipped")
        return 0

    print("verifying")
    results = verify.run_all(args.target, args.uicanvas_target, live_before=live_before)
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
