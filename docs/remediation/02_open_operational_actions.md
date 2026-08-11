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
