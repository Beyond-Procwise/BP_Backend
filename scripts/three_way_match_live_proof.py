#!/usr/bin/env python
"""Re-run the three-way match live proof of 2026-10-05, end to end.

The corpus holds no real goods receipts (design section 13), so the only way to
exercise the positive path against the running server is a SYNTHESISED delivery
note. This builds one against a real purchase order, hands it to the watcher the
way an upload would, and prints what came out.

It is a proof, not a fixture. It writes to whichever database .env names, and
`--clean` removes everything it wrote -- which is how it is left between runs,
because a fabricated note sitting in a demo database accuses real seeded
invoices, drives bp_deal_kpis and is marked synthetic only on the PDF's face.

    ./.venv/bin/python scripts/three_way_match_live_proof.py            # run it
    ./.venv/bin/python scripts/three_way_match_live_proof.py --clean    # undo it

Default order PO000645 / deal DEALV2-000645 carries two invoices that each bill
against it, so the proof also exercises the set-level case: either invoice alone
passes and together they exceed what arrived.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path[:0] = [os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]

GRN = "DN-100645"
PO = "PO000645"
DOC_DIR = "/home/muthu/Downloads/SpendIQDocs/three-way-match-2026-10-05"
DOC = f"{DOC_DIR}/delivery_note_{GRN}.pdf"

#: The order's own lines, so the note is consistent with what it cites.
LINES = [
    dict(line=1, desc="Standard Compliance Subscription ITM004425", qty=5, uom="each", price=13.59),
    dict(line=2, desc="Heavy-Duty Compliance Subscription ITM004436", qty=1, uom="tonne", price=1.15),
    dict(line=3, desc="Heavy-Duty A4/A3 Installation ITM001058", qty=8, uom="pack", price=2.70),
]


def _build_pdf() -> None:
    """A landscape A4 note, with the ORDER'S PRICES on its face.

    The prices are deliberate: a real delivery note restates them for the
    driver's paperwork, the extractor will find them, and none of them may
    reach a receipt row. That is Review Focus #4 and the proof checks it.

    Landscape, and with the columns spaced for their widest value, because the
    first attempt's columns ran together: docling merged two headers into
    "Qty Rejected Unit" and the delivered quantity bled into the description.
    """
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.units import mm
    from reportlab.pdfgen import canvas

    os.makedirs(DOC_DIR, exist_ok=True)
    c = canvas.Canvas(DOC, pagesize=landscape(A4))
    w, h = landscape(A4)
    y = h - 25 * mm
    c.setFont("Helvetica-Bold", 18)
    c.drawString(15 * mm, y, "DELIVERY NOTE")
    y -= 9 * mm
    c.setFont("Helvetica", 10)
    for line in (f"Delivery Note No: {GRN}", f"Against PO {PO}",
                 "Date Received: 18 March 2024",
                 "Supplier: SUP-WindroseSupplies7",
                 "Consignment No: CON-884215"):
        c.drawString(15 * mm, y, line)
        y -= 5.5 * mm
    y -= 4 * mm

    cols = ["Item No", "Description", "Qty Delivered", "Qty Rejected", "Unit",
            "Unit Price", "Line Total"]
    xs = [15, 32, 128, 158, 188, 212, 245]
    c.setFont("Helvetica-Bold", 9)
    for x, label in zip(xs, cols):
        c.drawString(x * mm, y, label)
    y -= 2 * mm
    c.line(11 * mm, y, w - 11 * mm, y)
    y -= 5 * mm
    c.setFont("Helvetica", 9)
    for ln in LINES:
        for x, v in zip(xs, [str(ln["line"]), ln["desc"][:60], f"{ln['qty']:g}", "0",
                             ln["uom"], f"GBP {ln['price']:.2f}",
                             f"GBP {ln['qty'] * ln['price']:.2f}"]):
            c.drawString(x * mm, y, v)
        y -= 5.5 * mm
    y -= 8 * mm
    c.setFont("Helvetica", 10)
    c.drawString(15 * mm, y, "Received by: J. Okafor")
    y -= 5.5 * mm
    c.drawString(15 * mm, y, "Signature: ____________________")
    y -= 12 * mm
    c.setFont("Helvetica-Oblique", 7.5)
    c.drawString(15 * mm, y,
                 "SYNTHESISED TEST DOCUMENT - BP_Backend three-way match live "
                 "proof. Not a customer document.")
    c.showPage()
    c.save()
    print(f"built {DOC}")


def _clean() -> None:
    from src.services.db import get_conn

    with get_conn() as conn, conn.cursor() as cur:
        for sql, args in (
            ("DELETE FROM proc.bp_extraction_discrepancy WHERE doc_pk_candidate = %s", (GRN,)),
            ("DELETE FROM proc.bp_extraction_discrepancy WHERE doc_pk_candidate IN "
             "(SELECT invoice_id FROM proc.bp_invoice_trgt WHERE po_id = %s) "
             "AND issue_type IN ('billed_not_received','nothing_received')", (PO,)),
            ("DELETE FROM proc.bp_goods_receipt_line_items_trgt WHERE grn_id = %s", (GRN,)),
            ("DELETE FROM proc.bp_goods_receipt_trgt WHERE grn_id = %s", (GRN,)),
            ("DELETE FROM proc.bp_goods_receipt_line_items_stg WHERE grn_id = %s", (GRN,)),
            ("DELETE FROM proc.bp_goods_receipt_stg WHERE grn_id = %s", (GRN,)),
            ("DELETE FROM proc.bp_goods_receipt_line_items_raw WHERE raw_id IN "
             "(SELECT raw_id FROM proc.bp_goods_receipt_raw WHERE grn_id = %s)", (GRN,)),
            ("DELETE FROM proc.bp_goods_receipt_raw WHERE grn_id = %s", (GRN,)),
            ("DELETE FROM proc.process_monitor WHERE file_path = %s", (DOC,)),
        ):
            cur.execute(sql, args)
    print(f"removed every row {GRN} produced")


def _report() -> int:
    from src.services.db import get_conn

    problems = 0
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT grn_id, po_id, receipt_date, received_by, carrier_ref, "
                    "deal_id, document_id, lines_assessed, lines_unverifiable "
                    "FROM proc.bp_goods_receipt_trgt WHERE grn_id = %s", (GRN,))
        row = cur.fetchone()
        if row is None:
            print("FAIL: the receipt never reached _trgt")
            return 1
        print(f"header   {row}")

        cur.execute("SELECT line_no, item_description, quantity_received, "
                    "quantity_rejected, unit_of_measure, po_line_ref "
                    "FROM proc.bp_goods_receipt_line_items_trgt WHERE grn_id = %s "
                    "ORDER BY line_no", (GRN,))
        lines = cur.fetchall()
        print(f"lines    {len(lines)}")
        for ln in lines:
            print(f"         {ln}")
        if len(lines) != len(LINES):
            print(f"FAIL: expected {len(LINES)} lines")
            problems += 1

        cur.execute("""SELECT column_name FROM information_schema.columns
                        WHERE table_schema='proc' AND table_name LIKE 'bp_goods_receipt%'
                          AND column_name ~* '(price|amount|total|cost|currency|tax)'""")
        priced = [r[0] for r in cur.fetchall()]
        print(f"priced columns on any goods-receipt table: {priced or 'none'}")
        if priced:
            problems += 1

        cur.execute("SELECT doc_pk_candidate, field_name, issue_type, severity, notes "
                    "FROM proc.bp_extraction_discrepancy WHERE status='open' "
                    "AND issue_type IN ('billed_not_received','nothing_received') "
                    "AND doc_pk_candidate IN (SELECT invoice_id FROM proc.bp_invoice_trgt "
                    "                          WHERE po_id = %s) "
                    "ORDER BY field_name", (PO,))
        gaps = cur.fetchall()
        print(f"gaps     {len(gaps)}")
        for g in gaps:
            print(f"         {g[0]} {g[1]} {g[2]}/{g[3]}: {g[4]}")

        cur.execute("SELECT deal_id, value_reconciled, three_way_matched "
                    "FROM proc.bp_deal_overview WHERE deal_id = "
                    "(SELECT deal_id FROM proc.bp_goods_receipt_trgt WHERE grn_id = %s)",
                    (GRN,))
        print(f"deal     {cur.fetchone()}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", action="store_true",
                    help="remove everything a previous run wrote, and stop")
    ap.add_argument("--wait", type=int, default=150,
                    help="seconds to wait for the watcher")
    args = ap.parse_args()

    if args.clean:
        _clean()
        return 0

    from src.services.db import get_conn

    _clean()          # idempotent: a re-run must not stack rows
    _build_pdf()
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO proc.process_monitor (process_name, type, status, file_path, "
            " category, created_by, created_date, lastmodified_date) "
            "VALUES ('three-way match live proof', 'extraction', 'Completed', %s, "
            "        'delivery note', 'three_way_match_live_proof', NOW(), NOW()) "
            "RETURNING id", (DOC,))
        record_id = cur.fetchone()[0]
    print(f"process_monitor {record_id} queued as category='delivery note'")

    deadline = time.time() + args.wait
    while time.time() < deadline:
        time.sleep(6)
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute("SELECT status FROM proc.process_monitor WHERE id = %s",
                        (record_id,))
            status = (cur.fetchone() or ["gone"])[0]
        if status in ("Extracted", "Staged", "Extraction_Failed"):
            break
        print(f"  …{status}")
    print(f"status   {status}")
    return _report()


if __name__ == "__main__":
    raise SystemExit(main())
