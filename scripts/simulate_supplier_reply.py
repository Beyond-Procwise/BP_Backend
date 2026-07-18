"""Inject a supplier reply so a blocked negotiation round can complete.

POST /workflows/negotiate drafts an email and then BLOCKS in
NegotiationAgent.wait_for_response until a real supplier answers. That is fine in
production and impossible in a test: nothing ever replies, so the request never returns.

This writes the row the agent is polling for, which is what an inbound email would have
produced via the email watcher. It exists to prove the rest of the pipeline (draft ->
send -> match -> evaluate -> counter) actually works end to end.

Usage:  .venv/bin/python scripts/simulate_supplier_reply.py <supplier> <price> [currency]
The workflow_id/unique_id are discovered from the newest awaiting draft.
"""

import os
import sys
from datetime import datetime, timezone

import psycopg2
from dotenv import load_dotenv

load_dotenv("/home/muthu/PycharmProjects/BP_Backend/.env")


def main() -> int:
    supplier = sys.argv[1] if len(sys.argv) > 1 else "PeopleFirst HR Solutions Ltd"
    price = float(sys.argv[2]) if len(sys.argv) > 2 else 94000.0
    currency = sys.argv[3] if len(sys.argv) > 3 else "GBP"

    conn = psycopg2.connect(
        host=os.getenv("DB_HOST"), port=os.getenv("DB_PORT", 5432),
        dbname=os.getenv("DB_NAME"), user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
    )
    cur = conn.cursor()

    # The agent tracks unique_id = "<workflow_id>-<supplier_id>". Prefer the tracked
    # outbound email; fall back to an explicit workflow id, because the negotiation path
    # currently writes NO proc.workflow_email_tracking row (nothing is actually dispatched
    # in this environment), so the tracking table is empty while the agent still waits.
    workflow_id = os.getenv("WORKFLOW_ID")
    if workflow_id:
        unique_id = f"{workflow_id}-{supplier}"
    else:
        cur.execute(
            """SELECT workflow_id, unique_id
                 FROM proc.workflow_email_tracking
                WHERE supplier_id = %s
                ORDER BY created_at DESC NULLS LAST
                LIMIT 1""",
            (supplier,),
        )
        row = cur.fetchone()
        if not row:
            print(f"no tracked outbound email for supplier {supplier!r}; "
                  "pass WORKFLOW_ID=<id> to target a waiting run directly")
            return 1
        workflow_id, unique_id = row

    now = datetime.now(timezone.utc)
    cur.execute(
        """INSERT INTO proc.supplier_response
             (workflow_id, unique_id, supplier_id, rfq_id, response_message_id,
              response_subject, response_text, response_body, response_from,
              round_number, response_date, received_time, price, currency,
              payment_terms, lead_time, match_confidence, processed)
           VALUES (%s,%s,%s,NULL,%s,%s,%s,%s,%s,1,%s,%s,%s,%s,%s,%s,%s,FALSE)
           ON CONFLICT (workflow_id, unique_id) DO NOTHING
           RETURNING id""",
        (
            workflow_id, unique_id, supplier,
            f"<simulated-{workflow_id}@example.invalid>",
            "RE: Negotiation",
            f"Thank you for the proposal. We can offer {price:,.2f} {currency} "
            "with 45 day payment terms and a 14 day lead time.",
            f"We can offer {price:,.2f} {currency}.",
            "billing@peoplefirst.invalid",
            now, now, price, currency, "45 Days", 14, 1.00,
        ),
    )
    got = cur.fetchone()
    conn.commit()
    print(f"workflow_id={workflow_id}")
    print(f"unique_id={unique_id}")
    print("inserted response id:", got[0] if got else "(already present)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
