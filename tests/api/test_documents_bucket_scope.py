"""1.3c (backend half): extract-from-s3 reads only the configured bucket.

``s3_path`` accepts the ``s3://bucket/prefix`` form, and until now ANY bucket name was honoured: a
signed-in Buyer could point extraction at any bucket the service's AWS role can read, and the
extracted text would land in this company's tables. The configured bucket is the only one this
service reads from; a different name is refused before any listing or download.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from api.routers.documents import _resolve_s3_path


def test_the_configured_bucket_in_url_form_still_resolves():
    assert _resolve_s3_path("s3://procwisemvp/Invoice/2026/", "procwisemvp") == ("procwisemvp", "Invoice/2026/")


def test_a_bare_prefix_uses_the_configured_bucket():
    assert _resolve_s3_path("Invoice/2026/", "procwisemvp") == ("procwisemvp", "Invoice/2026/")


@pytest.mark.parametrize("path", [
    "s3://someone-elses-bucket/Invoice/",
    "s3://procwisemvp-backup/Invoice/",
    "s3://PROCWISEMVP/Invoice/",           # S3 bucket names are case-sensitive; not the same bucket
])
def test_any_other_bucket_is_refused_before_anything_is_read(path):
    with pytest.raises(ValueError, match="configured bucket"):
        _resolve_s3_path(path, "procwisemvp")


def test_the_refusal_does_not_name_the_bucket_asked_for():
    """The error goes to the caller; the name they tried is not echoed (it reveals nothing useful and
    an attacker should not learn which names are checked how)."""
    with pytest.raises(ValueError) as exc:
        _resolve_s3_path("s3://probe-bucket/x/", "procwisemvp")
    assert "probe-bucket" not in str(exc.value)
