"""One real round trip: presign -> PUT with requests -> register -> document_text, on bp_testdb.

Needs PROCWISE_TEST_LIVE_DB=1 and DB_NAME=bp_testdb. The S3 object is deleted afterwards, also
on failure. The document rows stay (versions are the record); each run uses a unique name so it
creates its own document and never revises anyone else's.
"""
import os
import uuid

import pytest

pytestmark = pytest.mark.skipif(os.getenv("PROCWISE_TEST_LIVE_DB") != "1", reason="live DB required")


@pytest.fixture
def conn():
    assert os.getenv("DB_NAME") == "bp_testdb", "the live documents test runs on bp_testdb only"
    from services.db import get_conn
    with get_conn() as c:
        yield c


def test_presign_put_register_and_read_back(conn):
    import requests

    from services.agent_policy import documents as d

    tag = uuid.uuid4().hex[:10]
    name = f"Live Probe Policy {tag}.txt"
    line = f"Refunds over 500 GBP need manager approval. probe {tag}\n"
    body = (line * (1024 // len(line) + 1)).encode("utf-8")[:1024]
    assert len(body) == 1024

    upload = d.presign_uploads([{"name": name, "size": len(body), "contentType": "text/plain"}],
                               actor="test-live")[0]
    assert upload["key"].startswith(f"agent-policy-documents/uploads/{upload['uploadId']}/")
    client, bucket = d._s3(), d._bucket()
    try:
        put = requests.put(upload["url"], data=body, headers=upload["headers"], timeout=60)
        if put.status_code == 403:
            pytest.fail(f"BLOCKED: S3 refused the PUT (403): {put.text[:500]}")
        assert put.status_code == 200, put.text[:500]

        first = d.register_uploads(conn, [{"key": upload["key"], "name": name, "revisionOf": None}],
                                   actor="test-live")[0]
        assert first["version"] == 1 and first["duplicate"] is False and first["isRevision"] is False
        assert first["title"] == f"Live Probe Policy {tag}"

        again = d.register_uploads(conn, [{"key": upload["key"], "name": name, "revisionOf": None}],
                                   actor="test-live")[0]
        assert again == {**first, "duplicate": True}

        text = d.document_text(conn, first["documentId"], 1)
        assert text == body.decode("utf-8")
        assert d.document_text(conn, first["documentId"], 1) == text  # now from the row

        listed = [doc for doc in d.list_documents(conn) if doc["documentId"] == first["documentId"]]
        assert len(listed) == 1 and [v["version"] for v in listed[0]["versions"]] == [1]
        assert listed[0]["versions"][0]["parsed"] is True
    finally:
        client.delete_object(Bucket=bucket, Key=upload["key"])
    from botocore.exceptions import ClientError
    with pytest.raises(ClientError):
        client.head_object(Bucket=bucket, Key=upload["key"])
