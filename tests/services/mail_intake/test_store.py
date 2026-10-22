"""Content-addressed, write-once storage, with quarantine kept on a separate
prefix so the extraction path holds no key that can reach it."""
import io
from typing import Optional

import pytest

from src.services.mail_intake import store


def _client_error(code: str, status: int) -> Exception:
    """A botocore-shaped error, because that is the shape the code branches on.

    A bare KeyError cannot exercise `_is_not_found` at all — it has no
    `.response` — so a fake raising one would leave the branch untested while
    the tests looked complete.
    """
    from botocore.exceptions import ClientError

    return ClientError(
        {
            "Error": {"Code": code, "Message": code},
            "ResponseMetadata": {"HTTPStatusCode": status},
        },
        "HeadObject",
    )


class FakeS3:
    """Records puts; refuses to overwrite, the way write-once must behave.

    `fail_with` makes the store answer every call with a given error, so the
    non-404 branch — the one the whole write-once fix exists for — can actually
    be exercised rather than reasoned about.
    """

    def __init__(self, fail_with: Optional[Exception] = None):
        self.objects = {}
        self.fail_with = fail_with

    def put_object(self, Bucket, Key, Body, **kwargs):
        if self.fail_with:
            raise self.fail_with
        self.objects[(Bucket, Key)] = Body
        return {"ETag": "fake"}

    def head_object(self, Bucket, Key):
        if self.fail_with:
            raise self.fail_with
        if (Bucket, Key) not in self.objects:
            raise _client_error("404", 404)
        return {"ContentLength": len(self.objects[(Bucket, Key)])}

    def get_object(self, Bucket, Key):
        if self.fail_with:
            raise self.fail_with
        if (Bucket, Key) not in self.objects:
            raise _client_error("NoSuchKey", 404)
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


@pytest.mark.parametrize(
    "code,status",
    [("AccessDenied", 403), ("SlowDown", 503), ("RequestTimeout", 408)],
)
def test_a_non_404_failure_is_never_read_as_absent(code, status):
    """The property the write-once fix exists for, and the one nothing tested.

    An earlier version caught every exception and reported "absent", so a
    throttle or a permission change let put_immutable overwrite different
    content under the same key. The fix was correct by inspection but no
    committed test exercised a realistic non-404 error — only the 404 path was
    modelled, so the central claim was unproven.
    """
    client = FakeS3(fail_with=_client_error(code, status))
    objects = store.ObjectStore("bucket", client=client)
    key = store.artifact_key("user-7", store.sha256_hex(b"a"))

    with pytest.raises(store.StorageUnavailable):
        objects.exists(key)

    # And crucially: the write must not happen.
    with pytest.raises(store.StorageUnavailable):
        objects.put_immutable(key, b"a")
    assert client.objects == {}


def test_an_unreadable_existing_object_is_not_called_a_mismatch():
    """A storage failure and an integrity violation are both refusals, but only
    one of them is true. Reporting the wrong one sends a reviewer hunting for
    tampering that never happened."""

    class ExistsButUnreadable(FakeS3):
        def get_object(self, Bucket, Key):
            raise _client_error("SlowDown", 503)

    client = ExistsButUnreadable()
    objects = store.ObjectStore("bucket", client=client)
    key = store.artifact_key("user-7", store.sha256_hex(b"a"))
    client.objects[("bucket", key)] = b"a"

    with pytest.raises(store.StorageUnavailable):
        objects.put_immutable(key, b"a")



def test_a_store_without_a_client_builds_it_through_egress(monkeypatch):
    """1.20m: the default S3 client comes from egress.aws_client, so its requests are in the audit."""
    from src.services import egress
    seen = {}
    monkeypatch.setattr(egress, "aws_client", lambda service, *, purpose, **kw: seen.update(service=service, purpose=purpose) or object())
    objects = store.ObjectStore("bucket")
    first = objects._resolve_client()
    assert seen == {"service": "s3", "purpose": egress.Purpose.OBJECT_STORAGE}
    assert objects._resolve_client() is first        # built once, reused
