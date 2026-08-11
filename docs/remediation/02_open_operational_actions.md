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
