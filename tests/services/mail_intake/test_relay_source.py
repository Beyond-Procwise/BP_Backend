"""The relay adapter: raw MIME already sitting in object storage.

Preferred mode because it is push-based, needs no mailbox-wide read scope, and
preserves raw MIME exactly as delivered. The adapter's only job is to yield
bytes and provider identity — every interpretive decision happens downstream in
shared code, so all four adapters get the identical gate.
"""
from datetime import datetime, timezone

from src.services.mail_intake.sources.relay import RelayMailSource


class FakeS3:
    def __init__(self, objects):
        self._objects = objects

    def get_paginator(self, _name):
        outer = self

        class Paginator:
            def paginate(self, **kwargs):
                prefix = kwargs.get("Prefix", "")
                yield {
                    "Contents": [
                        {
                            "Key": key,
                            "LastModified": meta[1],
                            "Size": len(meta[0]),
                        }
                        for key, meta in outer._objects.items()
                        if key.startswith(prefix)
                    ]
                }

        return Paginator()

    def get_object(self, Bucket, Key):
        class Body:
            def __init__(self, data):
                self._data = data

            def read(self):
                return self._data

        return {"Body": Body(self._objects[Key][0])}


def _binding(**kwargs):
    base = {"binding_id": 1, "owner_ref": "user-7", "folder_scope": "emails/"}
    base.update(kwargs)
    return base


def test_yields_each_object_as_a_raw_message():
    when = datetime(2026, 8, 5, 9, 0, tzinfo=timezone.utc)
    source = RelayMailSource(
        bucket="procwisemvp",
        prefix="emails/",
        client=FakeS3({"emails/a.eml": (b"From: a@b.test\r\n\r\nhi", when)}),
    )
    messages = list(source.iter_messages(_binding(), since=None))
    assert len(messages) == 1
    assert messages[0].raw.startswith(b"From: a@b.test")
    assert messages[0].provider_message_id == "emails/a.eml"
    assert messages[0].received_at == when


def test_since_filter_excludes_older_objects():
    old = datetime(2026, 8, 1, tzinfo=timezone.utc)
    new = datetime(2026, 8, 5, tzinfo=timezone.utc)
    source = RelayMailSource(
        bucket="procwisemvp",
        prefix="emails/",
        client=FakeS3(
            {"emails/old.eml": (b"old", old), "emails/new.eml": (b"new", new)}
        ),
    )
    messages = list(
        source.iter_messages(_binding(), since=datetime(2026, 8, 3, tzinfo=timezone.utc))
    )
    assert [m.provider_message_id for m in messages] == ["emails/new.eml"]


def test_binding_folder_scope_narrows_the_prefix():
    source = RelayMailSource(
        bucket="procwisemvp",
        prefix="emails/",
        client=FakeS3(
            {
                "emails/acme/a.eml": (b"a", datetime(2026, 8, 5, tzinfo=timezone.utc)),
                "emails/other/b.eml": (b"b", datetime(2026, 8, 5, tzinfo=timezone.utc)),
            }
        ),
    )
    messages = list(
        source.iter_messages(_binding(folder_scope="emails/acme/"), since=None)
    )
    assert [m.provider_message_id for m in messages] == ["emails/acme/a.eml"]


def test_adapter_yields_bytes_without_parsing_them():
    # A malformed message must still be yielded — quarantine is the pipeline's
    # decision to make and record, not the adapter's to skip silently.
    source = RelayMailSource(
        bucket="procwisemvp",
        prefix="emails/",
        client=FakeS3(
            {"emails/bad.eml": (b"\xff\xfe not mime", datetime(2026, 8, 5, tzinfo=timezone.utc))}
        ),
    )
    assert len(list(source.iter_messages(_binding(), since=None))) == 1



def test_a_relay_source_without_a_client_builds_it_through_egress(monkeypatch):
    """1.20m: the default S3 client comes from egress.aws_client, so its requests are in the audit."""
    from src.services import egress
    from src.services.mail_intake.sources.relay import RelayMailSource
    seen = {}
    monkeypatch.setattr(egress, "aws_client", lambda service, *, purpose, **kw: seen.update(service=service, purpose=purpose) or object())
    source = RelayMailSource("bucket", "inbox/")
    first = source._resolve_client()
    assert seen == {"service": "s3", "purpose": egress.Purpose.OBJECT_STORAGE}
    assert source._resolve_client() is first
