"""The redaction key set must not be empty, and must not be two sets.

``DISALLOWED_METADATA_KEYS`` is the only filter standing between a document's
metadata and an externally hosted vector index (``QDRANT_URL`` points at
Qdrant Cloud, eu-central-1). It was declared twice — once in
``document_embedding_service`` and once in ``rag_service`` — and both were the
empty set, so every ``.pop()`` and every ``not in`` guarding a write removed
nothing at all.

Two separate empty sets is the more dangerous half of that: populating one and
not the other leaves whichever write path uses the other still unredacted, and
nothing would fail. These tests pin both properties.

What this does NOT claim: the chunk ``content`` — the document text itself —
still goes to the external index, because that is the field being embedded and
retrieved. Removing it would delete the feature, not protect it. Scoping the
content egress is per-tenant collections and a disable switch, which is separate
work. This is the metadata half only, and the test names say so.
"""
from __future__ import annotations

import pytest


def test_the_key_set_is_not_empty():
    """An empty set makes every call site that consults it a no-op."""
    from services.document_embedding_service import DISALLOWED_METADATA_KEYS

    assert DISALLOWED_METADATA_KEYS, (
        "DISALLOWED_METADATA_KEYS is empty, so the redaction applied at "
        "document_embedding_service, rag_service and static_policy_loader "
        "removes nothing before the payload reaches Qdrant Cloud"
    )


def test_every_module_shares_one_set_rather_than_declaring_its_own():
    """Drift here is silent: one path redacts, the other does not."""
    from services import document_embedding_service as des
    from services import rag_service as rs
    from services import static_policy_loader as spl

    assert des.DISALLOWED_METADATA_KEYS is rs.DISALLOWED_METADATA_KEYS, (
        "rag_service declares its own DISALLOWED_METADATA_KEYS; populating "
        "the document_embedding_service one leaves rag_service.upsert_payloads "
        "writing the keys unredacted"
    )
    assert des.DISALLOWED_METADATA_KEYS is spl.DISALLOWED_METADATA_KEYS


@pytest.mark.parametrize("key,why", [
    ("uploaded_by", "a named person's email address; observed live in the "
                    "uploaded_documents collection"),
    ("s3_key", "an internal object path; observed live in the "
               "procwise_document_embeddings collection"),
])
def test_the_keys_carrying_personal_or_infrastructure_data_are_redacted(key, why):
    from services.document_embedding_service import DISALLOWED_METADATA_KEYS

    assert key in DISALLOWED_METADATA_KEYS, f"{key} is not redacted ({why})"


@pytest.mark.parametrize("key,why", [
    ("content", "the embedded text; retrieval returns it"),
    ("summary", "read back at rag_service.py:2004 as a snippet fallback"),
    ("text_summary", "read back at rag_service.py:973 as a snippet fallback"),
    ("session_id", "used as a retrieval filter at rag_service.py:1424"),
    ("filename", "shown as the citation for a retrieved chunk"),
    ("doc_name", "shown as the citation for a retrieved chunk"),
    ("document_id", "joins a chunk to its document"),
    ("chunk_id", "orders chunks within a document"),
])
def test_keys_that_retrieval_depends_on_are_not_redacted(key, why):
    """Over-redaction here breaks search silently: the write succeeds, the
    payload is thinner, and the answer quality drops with nothing failing."""
    from services.document_embedding_service import DISALLOWED_METADATA_KEYS

    assert key not in DISALLOWED_METADATA_KEYS, (
        f"{key} is redacted but is consumed on the read path ({why})"
    )


def test_a_supplied_metadata_dict_is_filtered_by_the_same_rule():
    """The shape every call site applies, asserted once on real keys."""
    from services.document_embedding_service import DISALLOWED_METADATA_KEYS

    supplied = {
        "document_id": "doc-1",
        "filename": "Invoice_INV2025-292.pdf",
        "uploaded_by": "nicholasgeelen@gmail.com",
        "s3_key": "documents/Invoice/ELEANOR PRICE INV2025-292 for PO389948.pdf",
        "session_id": "verify-session-1783690674",
    }
    filtered = {k: v for k, v in supplied.items()
                if k not in DISALLOWED_METADATA_KEYS}

    assert "uploaded_by" not in filtered
    assert "s3_key" not in filtered
    assert filtered["session_id"] == "verify-session-1783690674"
    assert filtered["filename"] == "Invoice_INV2025-292.pdf"
