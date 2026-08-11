# Open operational actions — egress remediation, 2026-08-11

Two items from the Tranche 0 egress work are **not** code changes and were
deliberately left for a person. Both are one-off; neither is blocked on
anything.

---

## 1. Rotate the SES credentials in `.env`

**Status:** open
**Urgency:** do it, but there is no evidence of exposure beyond the one below.

`.env` holds live SES SMTP credentials in plaintext:

```
SES_SMTP_USER, SES_SMTP_PASSWORD, SES_USER_PASSWORD
```

What was checked, and what it found:

* `.env` **is** in `.gitignore` (line 17).
* `git log --all -- .env` is empty — it has **never** been committed. The
  credentials have not reached the remote.

So the file itself is handled correctly. The reason to rotate anyway is narrower
and worth stating precisely rather than as a general "secrets in env are bad":
during the egress audit on 2026-08-11 these values were printed in full into an
assistant conversation transcript while enumerating outbound destinations. That
transcript is a copy of the credentials outside the host.

**Action:** rotate the SES SMTP credentials in AWS and update `.env` on each
host. Nothing in the codebase needs changing — `email_service` and
`email_credentials_manager` read them by name.

**Worth considering separately:** these are read from `.env` directly, while
`email_credentials_manager` already knows how to pull from AWS Secrets Manager
(`boto3.client("secretsmanager")`). Moving them there would make the next
rotation a secret update rather than a file edit on every host. Not required to
close this item.

---

## 2. Purge `uploaded_by` / `s3_key` from the existing Qdrant points

**Status:** open — will resolve itself, but on an unpredictable schedule.

`5e54868` populated `DISALLOWED_METADATA_KEYS`, so **new** writes to Qdrant
Cloud no longer carry a named person's email address (`uploaded_by`) or internal
object paths (`s3_key`). Points written *before* that commit still carry both.

Measured on 2026-08-11:

| collection | points | carries |
|---|---|---|
| `procwise_document_embeddings` | 244 | `s3_key` |
| `uploaded_documents` | 6 | `uploaded_by`, `s3_key` |

The purge is already wired: `document_embedding_service._remove_disallowed_payload_fields`
and `rag_service._purge_disallowed_metadata` issue a `delete_payload` across the
whole collection, and both run on the next write to that collection. So this
**will** clear itself the next time a document is embedded — the open question is
only whether that happens today or in a month, and until it does the data is
sitting in an externally hosted index.

**Decision needed:** run it deliberately now, or wait for the next embed.

To run it deliberately (read the code first — this is an irreversible write to
the production index):

```python
# set -a && . ./.env && set +a && ./venv/bin/python
from qdrant_client import QdrantClient, models
import os
c = QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ.get("QDRANT_API_KEY"))
for collection in ("procwise_document_embeddings", "uploaded_documents"):
    c.delete_payload(
        collection_name=collection,
        keys=["uploaded_by", "s3_key"],
        points=models.FilterSelector(filter=models.Filter(must=[])),
        wait=True,
    )
```

This removes only those two keys. Vectors, `content` and every retrieval field
are untouched.

**Not closed by either option:** the chunk `content` — the document text itself —
is still written to Qdrant Cloud, because it is the field being embedded and
retrieved. Scoping that is per-tenant collections plus a `RAG_EXTERNAL_ENABLED`
switch (Tranche 1, items 9–10), not something the redaction key set can express.

---

## 3. Per-tenant vector collections — BLOCKED, not deferred

**Status:** cannot be built. Recorded here so it is not picked up again as though
it were merely outstanding work.

The remediation backlog listed "per-tenant Qdrant collections". It cannot be
implemented, because there is no tenant to key a collection on:

* `src/api/auth.py:19-21` states it directly — no `customer_id` on the vector
  payloads or the `bp_` tables, and the `x-customer-id` header the UI sends is
  the constant `"001"`.
* `src/services/procurement_knowledge_service.py:105` actively **pops**
  `customer_id` out of payloads, so any tenant signal that did arrive is
  deliberately discarded.
* Five of 146 `proc` tables carry a `tenant_id`, and all five belong to the
  unwired facts layer (`bp_commercial_fact` holds 0 rows).
* All four live Qdrant collections are shared.

A `collection_for(tenant_id)` seam where `tenant_id` is always `"default"` would
produce `procwise_document_embeddings__default`. That is a rename presented as
isolation — a control that looks like a control and separates nothing — and it
would read as "multi-tenant isolation shipped" in any subsequent review.

**What has to exist first**, in order:

1. A tenant dimension on the identity: a claim on the Cognito token, or a real
   `customer_id` that is not the constant `"001"`.
2. That dimension carried onto `proc.bp_*` rows, so the database and the vector
   index can agree on who a document belongs to.
3. `procurement_knowledge_service` to stop discarding it.

Only then does collection-per-tenant mean anything. Until then the honest
statement is that this product is single-tenant, and the isolation control that
matters is `RAG_EXTERNAL_ENABLED` (shipped) plus not putting a second customer
on the same deployment.

---

## 4. Synthetic contracts removed from the knowledge graph — DONE 2026-08-11

**Status:** closed.

The Neo4j graph held a test dataset that had never been cleared, and it was
disagreeing with the relational store in a way that could only mislead:

| | Graph (before) | Relational |
|---|---|---|
| Contract | 3,051 | **0** |
| Supplier | 7,851 | 5,028 |

The 3,051 `Contract` nodes were unmistakably synthetic — sequential
`C00002`..`C00014` ids, every `last_modified_date` identical at
`2025-09-02 10:29:03.494372` (one bulk load), and incoherent field combinations
such as jurisdiction `UK` with governing law `German Civil Code`. They existed
only in the graph; `proc.bp_contracts` has always held 0 rows.

They were attached to 2,529 `Supplier` nodes carrying `S####` ids — a different
scheme from the real `SUP-*` suppliers. Verified disjoint before deleting:

* S#### suppliers touched **0** Invoice, PurchaseOrder, Quote, InvoiceLine,
  POLine or QuoteLine nodes.
* Real `SUP-*` suppliers touched **0** Contract nodes.

So the test data formed an island connected to nothing real, and removing it
could not affect production entities.

**Removed:** 3,051 Contract + 2,529 Supplier nodes and their 2,981
`SUPPLIER_PARTY_TO_CONTRACT` relationships. Graph went 13,130 -> 7,550 nodes.
Real counts unchanged: Invoice 612, Quote 435, PurchaseOrder 295, InvoiceLine
408, QuoteLine 197, POLine 164.

**Backup:** `backups/neo4j/synthetic_contracts_<stamp>.json` — 5,580 nodes and
3,008 relationships, enough to reconstruct the island if it is ever wanted.
Gitignored.

### Two observations that are NOT closed by this

1. **5,051 graph suppliers have no relationships at all.** Pre-existing, not
   caused by the delete (the deleted nodes all had relationships). The graph
   holds 5,205 `SUP-*` suppliers against 5,028 in `proc.bp_supplier`, so the
   supplier set has drifted from the relational source independently.

2. **The graph was last written 2026-07-31.** That is 11 days before this note,
   while `proc.bp_invoice_trgt` holds 12,408 invoices against the graph's 612.
   Whatever stopped `kg_sync` writing is a separate question, and the KG is
   stale for reasons this cleanup does not address. Both `corpus_facts.describe`
   and the `describe_platform` tool read it.

---

## 5. Why the knowledge graph stopped updating — ROOT CAUSE, 2026-08-11

**Status:** fixed in code; the first full rebuild is deliberately still pending.

`ProcurementKGBuilder.ENTITY_TABLE_MAP` named six tables that no longer exist:

    proc.bp_invoice, bp_invoice_line_items, bp_purchase_order,
    bp_po_line_items, bp_quote, bp_quote_line_items

They were dropped when extraction moved to `_stg -> _trgt`. Every document
loader read a missing table, caught the error, logged it at **DEBUG**, and
returned 0. The scheduled job then logged `"KG sync completed: {...}"` at INFO
with every document count at zero. The graph last gained a document on
2026-07-31; nobody noticed for eleven days, while `corpus_facts.describe` and
AgentNick's `describe_platform` tool kept reading it.

Two further entries were wrong in the same silent way:

* `bp_category` has no `category_id` column at all (its columns are
  `item_description`, `category`) — entry removed, the table holds 0 rows.
* `bp_policy` is keyed on `policy_id`, not `id` — so **19 live policy rows never
  loaded**, silently, for as long as the map has existed.
* `proc.bp_approvals` does not exist — entry removed.

### What was fixed

1. **The map now reads the `_trgt` tier**, with primary keys checked against
   `information_schema` rather than assumed.
2. **`LIMIT 5000` was a silent cap, not a batch size** — no OFFSET. `bp_supplier`
   has 5,028 rows so 28 never loaded; `bp_quote_line_items_trgt` has 115,814, of
   which it would have loaded 4% and reported success. Now paged.
3. **The failed read logs at WARNING**, naming the table and the label that will
   be missing. This is the line that hid the fault.
4. **The job no longer claims success on an empty run.** If every document label
   comes back 0 it logs an error saying so.
5. **`KG_FULL_REBUILD_ENABLED` gates the scheduled rebuild, default OFF** —
   same pattern and same reasoning as `DUPLICATE_INVOICE_DETECTOR_ENABLED`
   above it in that file.

`tests/services/test_kg_source_tables.py` pins all of it against the live
schema, and connects with psycopg2 directly rather than through
`services.db.get_conn`, which substitutes a fake connection under pytest — a
schema test asking a stub whether a table exists proves nothing.

### THE OPEN PART: the first full rebuild

The scheduled rebuild is off. Turning it on writes roughly **232,000 nodes**
(38,498 documents + 193,857 line items) into a graph that currently holds a few
thousand — unattended, on a six-hour timer. Run it deliberately and watch it:

```
KG_FULL_REBUILD_ENABLED=1  # then trigger _run_kg_sync once, supervised
```

Note the loader does one `session.run` per row, so ~232k round trips. Expect it
to be slow, and consider batching with UNWIND before making it routine.

**Already loaded during diagnosis** (supervised, verified): PurchaseOrder 295 ->
5,332, plus one invoice and a supplier refresh. Invoice, Quote and all three
line-item labels are still at their stale counts.
