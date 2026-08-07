"""POST /documents/extract-from-s3 accepts a caller-supplied prefix, lists every
key beneath it and extracts each one. Until this guard exists, the safety gate
is bypassable: anything can point extraction at a quarantined object.

Spec §2.3 crossing 1.
"""
import pytest

from src.api.routers.documents import _resolve_s3_path
from src.services.mail_intake.store import ARTIFACT_PREFIX, QUARANTINE_PREFIX


def test_ordinary_prefix_still_resolves():
    bucket, prefix = _resolve_s3_path("incoming/batch-4/", "procwisemvp")
    assert bucket == "procwisemvp"
    assert prefix == "incoming/batch-4/"


def test_quarantine_prefix_is_rejected():
    with pytest.raises(ValueError, match="quarantine"):
        _resolve_s3_path(f"{QUARANTINE_PREFIX}user-7/ab/cd/abcd", "procwisemvp")


def test_quarantine_prefix_is_rejected_via_s3_url_form():
    with pytest.raises(ValueError, match="quarantine"):
        _resolve_s3_path(
            f"s3://procwisemvp/{QUARANTINE_PREFIX}user-7/", "procwisemvp"
        )


def test_traversal_into_quarantine_is_rejected():
    with pytest.raises(ValueError):
        _resolve_s3_path(
            f"{ARTIFACT_PREFIX}../quarantine/user-7/", "procwisemvp"
        )


def test_leading_slash_variant_is_rejected():
    with pytest.raises(ValueError, match="quarantine"):
        _resolve_s3_path(f"/{QUARANTINE_PREFIX}user-7/", "procwisemvp")


def test_an_ancestor_prefix_is_rejected():
    """The bypass the descendant-only tests missed.

    `list_objects_v2` treats the prefix as literal bytes, so a prefix that
    merely sits *underneath* quarantine enumerates every quarantined object.
    """
    for ancestor in ("mail-intake/", "mail-intake"):
        with pytest.raises(ValueError, match="ancestor"):
            _resolve_s3_path(ancestor, "procwisemvp")


def test_quarantined_keys_are_never_listed_whatever_the_prefix():
    """The control that actually holds. A prefix check cannot be complete —
    'm' is a valid ancestor — so the filter is per key."""
    from src.api.routers.documents import _list_s3_keys

    class FakeClient:
        def list_objects_v2(self, **kwargs):
            return {
                "Contents": [
                    {"Key": "mail-intake/artifacts/user-7/ab/cd/clean"},
                    {"Key": f"{QUARANTINE_PREFIX}user-7/ab/cd/held"},
                ],
                "IsTruncated": False,
            }

    keys = _list_s3_keys(FakeClient(), "procwisemvp", "m")
    assert keys == ["mail-intake/artifacts/user-7/ab/cd/clean"]
    assert not any(k.startswith(QUARANTINE_PREFIX) for k in keys)
