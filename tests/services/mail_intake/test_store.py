"""Content-addressed, write-once storage, with quarantine kept on a separate
prefix so the extraction path holds no key that can reach it."""
import io

import pytest

from src.services.mail_intake import store


class FakeS3:
    """Records puts; refuses to overwrite, the way write-once must behave."""

    def __init__(self):
        self.objects = {}

    def put_object(self, Bucket, Key, Body, **kwargs):
        self.objects[(Bucket, Key)] = Body
        return {"ETag": "fake"}

    def head_object(self, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            raise KeyError(Key)
        return {"ContentLength": len(self.objects[(Bucket, Key)])}

    def get_object(self, Bucket, Key):
        # ObjectStore.put_immutable re-reads an existing key to decide
        # whether a re-put is an identical no-op or a genuine violation —
        # real boto3 has no other way to compare content server-side.
        if (Bucket, Key) not in self.objects:
            raise KeyError(Key)
        return {"Body": io.BytesIO(self.objects[(Bucket, Key)])}


def test_digest_is_stable_and_hex():
    digest = store.sha256_hex(b"quote bytes")
    assert digest == store.sha256_hex(b"quote bytes")
    assert len(digest) == 64
    int(digest, 16)


def test_artifact_key_is_content_addressed_and_owner_scoped():
    digest = store.sha256_hex(b"x")
    key = store.artifact_key("user-7", digest)
    assert key.startswith(store.ARTIFACT_PREFIX)
    assert digest in key
    assert "user-7" in key


def test_quarantine_uses_a_different_prefix():
    digest = store.sha256_hex(b"x")
    assert not store.quarantine_key("user-7", digest).startswith(
        store.ARTIFACT_PREFIX
    )
    assert store.quarantine_key("user-7", digest).startswith(
        store.QUARANTINE_PREFIX
    )


def test_put_immutable_writes_once():
    client = FakeS3()
    objects = store.ObjectStore("bucket", client=client)
    key = store.artifact_key("user-7", store.sha256_hex(b"a"))
    objects.put_immutable(key, b"a")
    assert objects.exists(key)


def test_put_immutable_refuses_to_overwrite_different_content():
    client = FakeS3()
    objects = store.ObjectStore("bucket", client=client)
    key = store.artifact_key("user-7", store.sha256_hex(b"a"))
    objects.put_immutable(key, b"a")
    with pytest.raises(store.ImmutabilityViolation):
        objects.put_immutable(key, b"different")


def test_reput_of_identical_content_is_a_no_op():
    client = FakeS3()
    objects = store.ObjectStore("bucket", client=client)
    key = store.artifact_key("user-7", store.sha256_hex(b"a"))
    objects.put_immutable(key, b"a")
    objects.put_immutable(key, b"a")
    assert len(client.objects) == 1
