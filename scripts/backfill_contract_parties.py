"""Correct a contract's parties and signatory from its own words.

WHY. Until 2026-10-03 a contract's party fields were answered by the entity sweep,
which had party-aware logic only for the INVOICE field names (`supplier_name`,
`buyer_id`). A contract's `supplier_id` / `buyer_org_id` fell through to the default
path, which emits every entity of the required type for any field -- so both fields
received identical candidate lists and the stored supplier was usually the BUYER.
See specs/2026-10-02-contract-structures-verification.md section 10.

The fix (src/services/extraction/engineered/contract_parties.py) applies to
documents extracted from then on. This script applies it to rows already stored,
WITHOUT re-extracting: every `proc.bp_contract_raw` row keeps the document's text in
`parser_snapshot->>'full_text'`, so the clause can be re-read deterministically --
no GPU, no watcher, no model.

THE RULES are in `contract_parties.decide_correction` and each one is a test:
  1. a human-confirmed value (provenance `hitl`) is never touched;
  2. a row with no stored text is left alone -- guessing there is worse;
  3. if the document states its parties, they are the answer;
  4. if it does not, the stored value is cleared ONLY when it came from the entity
     sweep. The context layer reads the whole document and is grounding-checked, so
     this has no standing to overrule it, and an absent provenance is not evidence.

USAGE. Dry run (prints what it would do, writes nothing):
    set -a && . ./.env && set +a
    ./venv/bin/python scripts/backfill_contract_parties.py

Apply, naming the database explicitly because bp_sqldb is not in .env:
    ./venv/bin/python scripts/backfill_contract_parties.py --apply
    DB_NAME=bp_sqldb ./venv/bin/python scripts/backfill_contract_parties.py --apply

It corrects THREE fields: `supplier_id`, `buyer_org_id` (from the party clause) and
`contract_signatory_name`/`_role` (from the signature block — the same default NER
path stored "Email Marketing" as the person who signed the real Marketing Agreement).

Both tiers are updated together: `proc.bp_contract_raw` and, through the row's
`contract_id`, `proc.bp_contracts`. A corrected field's provenance is rewritten to
say the party clause produced it, with `backfilled_at`, so the row does not keep
claiming the sweep's answer.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import psycopg2  # noqa: E402

from config.settings import Settings  # noqa: E402
from src.services.extraction.engineered.contract_parties import (  # noqa: E402
    BUYER_FIELD, CONFIDENCE, SUPPLIER_FIELD, decide_correction,
)
from src.services.extraction.engineered.contract_dates import (  # noqa: E402
    END_FIELD, START_FIELD, decide_date_correction,
)
from src.services.extraction.engineered.contract_header import (  # noqa: E402
    LAW_FIELD, PAYMENT_FIELD, TITLE_FIELD, decide_header_correction,
)
from src.services.extraction.engineered.contract_value import (  # noqa: E402
    CURRENCY_FIELD, VALUE_FIELD, decide_value_correction,
)
from src.services.extraction.engineered.contract_signatories import (  # noqa: E402
    BUYER_NAME_FIELD, NAME_FIELD, ROLE_FIELD, decide_signatory_correction,
)

_READ = """
    SELECT raw_id, source_file, contract_id, supplier_id, buyer_org_id,
           parser_snapshot->>'full_text'                                AS full_text,
           parser_snapshot->'_field_provenance'->'supplier_id'->>'source' AS sup_src,
           parser_snapshot->'_field_provenance'->'buyer_org_id'->>'source' AS buy_src,
           contract_signatory_name, contract_signatory_role,
           buyer_signatory_name, buyer_signatory_role,
           contract_start_date, contract_end_date,
           total_contract_value, currency,
           contract_title, governing_law, payment_terms,
           parser_snapshot->'_field_provenance'->'contract_title'->>'source' AS hdr_src,
           parser_snapshot->'_field_provenance'->'total_contract_value'->>'source' AS value_src,
           parser_snapshot->'_field_provenance'->'contract_signatory_name'->>'source' AS sig_src,
           parser_snapshot->'_field_provenance'->'contract_start_date'->>'source' AS date_src
      FROM proc.bp_contract_raw
     ORDER BY raw_id
"""


def _conn(db: str | None):
    s = Settings()
    return psycopg2.connect(host=s.db_host, dbname=db or s.db_name, user=s.db_user,
                            password=s.db_password, port=s.db_port)


def _provenance_patch(changed_fields: dict[str, str | None], *,
                      source: str = "parties",
                      pattern: str = "contract_party_clause") -> dict:
    """What the row should say about who produced each value.

    `source` and `pattern` are arguments because a date is not read by the party
    clause: a fresh extraction records source='date' for the term and
    source='parties' for the parties, and a backfill that wrote 'parties' over a
    date would make the audit trail lie about which reader answered.
    """
    now = datetime.now(timezone.utc).isoformat()
    patch = {}
    for field, value in changed_fields.items():
        if value is None:
            patch[field] = {"source": f"{source}-cleared", "confidence": None,
                            "pattern_name": None, "backfilled_at": now}
        else:
            patch[field] = {"source": source, "confidence": CONFIDENCE,
                            "pattern_name": pattern, "backfilled_at": now}
    return patch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="write the corrections (default: dry run)")
    ap.add_argument("--db", default=None, help="database name (default: DB_NAME from .env)")
    args = ap.parse_args()

    conn = _conn(args.db)
    conn.autocommit = False
    cur = conn.cursor()
    cur.execute("SELECT current_database()")
    db_name = cur.fetchone()[0]
    print(f"database: {db_name}   mode: {'APPLY' if args.apply else 'dry run'}\n")

    cur.execute(_READ)
    cols = [d[0] for d in cur.description]
    rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    if not rows:
        print("proc.bp_contract_raw is empty: nothing to correct.")
        return 0

    corrected = cleared = unchanged = signatories = terms = values = headers = 0
    for r in rows:
        # The two fields share one decision, because the bug was that they shared
        # one answer. Provenance is read from supplier_id, falling back to buyer.
        d = decide_correction(
            full_text=r["full_text"] or "",
            stored_supplier=r["supplier_id"],
            stored_buyer=r["buyer_org_id"],
            provenance_source=r["sup_src"] or r["buy_src"],
        )
        sg = decide_signatory_correction(
            full_text=r["full_text"] or "",
            stored_name=r["contract_signatory_name"],
            stored_role=r["contract_signatory_role"],
            provenance_source=r["sig_src"],
            stored_buyer_name=r["buyer_signatory_name"],
            stored_buyer_role=r["buyer_signatory_role"],
        )
        dt = decide_date_correction(
            full_text=r["full_text"] or "",
            stored_start=str(r["contract_start_date"]) if r["contract_start_date"] else None,
            stored_end=str(r["contract_end_date"]) if r["contract_end_date"] else None,
            provenance_source=r["date_src"],
        )
        vl = decide_value_correction(
            full_text=r["full_text"] or "",
            stored_value=str(r["total_contract_value"]) if r["total_contract_value"] is not None else None,
            stored_currency=r["currency"],
            provenance_source=r["value_src"],
        )
        hd = decide_header_correction(
            full_text=r["full_text"] or "",
            stored_title=r["contract_title"],
            stored_law=r["governing_law"],
            stored_payment=r["payment_terms"],
            provenance_source=r["hdr_src"],
        )
        name = (r["source_file"] or "").split("/")[-1] or f"raw_id={r['raw_id']}"
        if hd.changed:
            print(f"  > {name:38s} header: title={hd.title!r} law={hd.governing_law!r} "
                  f"pay={hd.payment_terms!r}")
            print(f"    {'':38s} why: {hd.reason}")
            headers += 1
            if args.apply:
                patch = _provenance_patch({TITLE_FIELD: hd.title, LAW_FIELD: hd.governing_law,
                                           PAYMENT_FIELD: hd.payment_terms},
                                          source="regex", pattern="contract_header_clause")
                cur.execute(
                    """UPDATE proc.bp_contract_raw
                          SET contract_title = %s, governing_law = %s, payment_terms = %s,
                              parser_snapshot = jsonb_set(
                                  coalesce(parser_snapshot, '{}'::jsonb),
                                  '{_field_provenance}',
                                  coalesce(parser_snapshot->'_field_provenance', '{}'::jsonb)
                                      || %s::jsonb,
                                  true)
                        WHERE raw_id = %s""",
                    (hd.title, hd.governing_law, hd.payment_terms,
                     json.dumps(patch), r["raw_id"]),
                )
                if r["contract_id"]:
                    cur.execute(
                        """UPDATE proc.bp_contracts
                              SET contract_title = %s, governing_law = %s, payment_terms = %s,
                                  last_modified_by = 'backfill_contract_parties',
                                  last_modified_date = NOW()
                            WHERE contract_id = %s""",
                        (hd.title, hd.governing_law, hd.payment_terms, r["contract_id"]),
                    )
        if vl.changed:
            print(f"  > {name:38s} value: {r['total_contract_value']} {r['currency']}"
                  f"  ->  {vl.value} {vl.currency}")
            print(f"    {'':38s} why: {vl.reason}")
            values += 1
            if args.apply:
                patch = _provenance_patch({VALUE_FIELD: vl.value, CURRENCY_FIELD: vl.currency},
                                          source="regex", pattern="contract_total_clause")
                cur.execute(
                    """UPDATE proc.bp_contract_raw
                          SET total_contract_value = %s, currency = %s,
                              parser_snapshot = jsonb_set(
                                  coalesce(parser_snapshot, '{}'::jsonb),
                                  '{_field_provenance}',
                                  coalesce(parser_snapshot->'_field_provenance', '{}'::jsonb)
                                      || %s::jsonb,
                                  true)
                        WHERE raw_id = %s""",
                    (vl.value, vl.currency, json.dumps(patch), r["raw_id"]),
                )
                if r["contract_id"]:
                    cur.execute(
                        """UPDATE proc.bp_contracts
                              SET total_contract_value = %s, currency = %s,
                                  last_modified_by = 'backfill_contract_parties',
                                  last_modified_date = NOW()
                            WHERE contract_id = %s""",
                        (vl.value, vl.currency, r["contract_id"]),
                    )
        if dt.changed:
            print(f"  > {name:38s} term: {r['contract_start_date']} .. "
                  f"{r['contract_end_date']}  ->  {dt.start} .. {dt.end}")
            print(f"    {'':38s} why: {dt.reason}")
            terms += 1
            if args.apply:
                patch = _provenance_patch({START_FIELD: dt.start, END_FIELD: dt.end},
                                          source="date",
                                          pattern="contract_term_clause")
                cur.execute(
                    """UPDATE proc.bp_contract_raw
                          SET contract_start_date = %s, contract_end_date = %s,
                              parser_snapshot = jsonb_set(
                                  coalesce(parser_snapshot, '{}'::jsonb),
                                  '{_field_provenance}',
                                  coalesce(parser_snapshot->'_field_provenance', '{}'::jsonb)
                                      || %s::jsonb,
                                  true)
                        WHERE raw_id = %s""",
                    (dt.start, dt.end, json.dumps(patch), r["raw_id"]),
                )
                if r["contract_id"]:
                    cur.execute(
                        """UPDATE proc.bp_contracts
                              SET contract_start_date = %s, contract_end_date = %s,
                                  last_modified_by = 'backfill_contract_parties',
                                  last_modified_date = NOW()
                            WHERE contract_id = %s""",
                        (dt.start, dt.end, r["contract_id"]),
                    )
        if sg.changed:
            mark = "x" if sg.name is None else ">"
            print(f"  {mark} {name:38s} signatory: {str(r['contract_signatory_name'])!r}"
                  f" -> {str(sg.name)!r}   buyer's: "
                  f"{str(r['buyer_signatory_name'])!r} -> {str(sg.buyer_name)!r}")
            print(f"    {'':38s} why: {sg.reason}")
            signatories += 1
            if args.apply:
                patch = _provenance_patch({NAME_FIELD: sg.name,
                                           BUYER_NAME_FIELD: sg.buyer_name},
                                          pattern="contract_signature_block")
                cur.execute(
                    """UPDATE proc.bp_contract_raw
                          SET contract_signatory_name = %s,
                              contract_signatory_role = %s,
                              buyer_signatory_name = %s,
                              buyer_signatory_role = %s,
                              parser_snapshot = jsonb_set(
                                  coalesce(parser_snapshot, '{}'::jsonb),
                                  '{_field_provenance}',
                                  coalesce(parser_snapshot->'_field_provenance', '{}'::jsonb)
                                      || %s::jsonb,
                                  true)
                        WHERE raw_id = %s""",
                    (sg.name, sg.role, sg.buyer_name, sg.buyer_role,
                     json.dumps(patch), r["raw_id"]),
                )
                if r["contract_id"]:
                    cur.execute(
                        """UPDATE proc.bp_contracts
                              SET contract_signatory_name = %s,
                                  contract_signatory_role = %s,
                                  buyer_signatory_name = %s,
                                  buyer_signatory_role = %s,
                                  last_modified_by = 'backfill_contract_parties',
                                  last_modified_date = NOW()
                            WHERE contract_id = %s""",
                        (sg.name, sg.role, sg.buyer_name, sg.buyer_role,
                         r["contract_id"]),
                    )
        if not d.changed:
            unchanged += 1
            print(f"  = {name:38s} supplier={str(r['supplier_id']):26s} -- {d.reason}")
            continue

        if d.supplier is None and d.buyer is None:
            cleared += 1
            mark = "x"
        else:
            corrected += 1
            mark = ">"
        print(f"  {mark} {name:38s} supplier: {str(r['supplier_id'])!r} -> {str(d.supplier)!r}")
        print(f"    {'':38s} buyer:    {str(r['buyer_org_id'])!r} -> {str(d.buyer)!r}")
        print(f"    {'':38s} why: {d.reason}")

        if not args.apply:
            continue

        patch = _provenance_patch({SUPPLIER_FIELD: d.supplier, BUYER_FIELD: d.buyer})
        cur.execute(
            """UPDATE proc.bp_contract_raw
                  SET supplier_id = %s,
                      buyer_org_id = %s,
                      parser_snapshot = jsonb_set(
                          coalesce(parser_snapshot, '{}'::jsonb),
                          '{_field_provenance}',
                          coalesce(parser_snapshot->'_field_provenance', '{}'::jsonb) || %s::jsonb,
                          true)
                WHERE raw_id = %s""",
            (d.supplier, d.buyer, json.dumps(patch), r["raw_id"]),
        )
        if r["contract_id"]:
            # The _trgt tier must not keep the old answer: it is what every
            # reader, every join and the link scorer's supplier signal use.
            cur.execute(
                """UPDATE proc.bp_contracts
                      SET supplier_id = %s, buyer_org_id = %s,
                          last_modified_by = 'backfill_contract_parties',
                          last_modified_date = NOW()
                    WHERE contract_id = %s""",
                (d.supplier, d.buyer, r["contract_id"]),
            )

    print(f"\n{len(rows)} row(s): {corrected} party corrected, {cleared} party cleared, "
          f"{unchanged} party unchanged, {signatories} signatory changed, "
          f"{terms} term changed, {values} value changed, "
          f"{headers} header changed")
    if args.apply:
        conn.commit()
        print("committed.")
    else:
        conn.rollback()
        print("dry run: nothing written. Re-run with --apply.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
