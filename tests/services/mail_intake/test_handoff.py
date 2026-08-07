"""Handing extraction bytes rather than a key.

Extraction resolves an inbound path by trying the local filesystem first and
falling back to S3 by key. That fallback is the crossing: given a key, it
fetches, and it can fetch any key in the bucket. Given a local file it owns
nothing and reaches nowhere.
"""
import os

import pytest

from src.services.mail_intake import handoff


def test_extraction_receives_a_real_local_file_never_a_key():
    seen = {}

    def fake_dispatch(*, process_monitor_id, file_path, doc_type):
        seen["path"] = file_path
        seen["exists"] = os.path.isfile(file_path)
        seen["content"] = open(file_path, "rb").read()
        seen["doc_type"] = doc_type
        return {"status": "ok"}

    handoff.deliver_to_extraction(
        artifact_id="ART-1",
        content=b"%PDF-1.4\ninvoice bytes\n",
        doc_type="invoice",
        dispatch=fake_dispatch,
    )

    # A real file on disk, so extraction's resolver takes its local branch and
    # never reaches the S3 fallback.
    assert seen["exists"] is True
    assert seen["content"] == b"%PDF-1.4\ninvoice bytes\n"
    assert seen["doc_type"] == "invoice"
    # And nothing that could be read as an object key.
    assert not seen["path"].startswith("mail-intake/")
    assert "s3://" not in seen["path"]


def test_the_temporary_file_is_removed_even_when_extraction_raises():
    captured = {}

    def exploding_dispatch(*, process_monitor_id, file_path, doc_type):
        captured["path"] = file_path
        raise RuntimeError("extraction blew up")

    with pytest.raises(RuntimeError):
        handoff.deliver_to_extraction(
            artifact_id="ART-2",
            content=b"bytes",
            doc_type="invoice",
            dispatch=exploding_dispatch,
        )

    assert not os.path.exists(captured["path"]), (
        "attachment bytes left on local disk after a failure"
    )


def test_the_temporary_file_is_not_world_readable():
    """These are attachments from arbitrary senders, briefly on local disk."""
    modes = {}

    def fake_dispatch(*, process_monitor_id, file_path, doc_type):
        modes["mode"] = os.stat(file_path).st_mode & 0o777
        return {}

    handoff.deliver_to_extraction(
        artifact_id="ART-3", content=b"x", doc_type="invoice",
        dispatch=fake_dispatch,
    )
    assert modes["mode"] & 0o077 == 0, f"mode was {oct(modes['mode'])}"


def test_the_temp_file_carries_the_real_extension():
    """Extraction's parser router dispatches purely on file extension.

    A hardcoded ".bin" suffix made every mail-intake document fail at L0 with
    "unsupported file format" — the production path could never have worked.
    That failure also invites exactly the wrong repair: the quickest way to
    silence it is to hand extraction the S3 key instead, which is the move
    this module exists to prevent.
    """
    seen = {}

    def fake_dispatch(*, process_monitor_id, file_path, doc_type):
        seen["path"] = file_path
        return {}

    handoff.deliver_to_extraction(
        artifact_id="ART-4",
        content=b"%PDF-1.4\n",
        doc_type="invoice",
        filename="quote.PDF",
        dispatch=fake_dispatch,
    )
    assert seen["path"].endswith(".PDF")


def test_a_missing_filename_falls_back_to_bin():
    seen = {}

    def fake_dispatch(*, process_monitor_id, file_path, doc_type):
        seen["path"] = file_path
        return {}

    handoff.deliver_to_extraction(
        artifact_id="ART-5", content=b"x", doc_type="invoice",
        dispatch=fake_dispatch,
    )
    assert seen["path"].endswith(".bin")
