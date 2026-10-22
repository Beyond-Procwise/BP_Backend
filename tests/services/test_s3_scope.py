"""1.3e: one withheld-store guard for every caller-named S3 key or prefix."""
from __future__ import annotations

import pytest

from src.services import s3_scope
from src.services.mail_intake.store import QUARANTINE_PREFIX, RAW_MIME_PREFIX


@pytest.mark.parametrize("path", ["Invoice/2026/a.pdf", "documents/invoice/x.pdf", "mail-intake/artifacts/x.eml", "/Invoice/"])
def test_ordinary_paths_pass_normalised(path):
    assert s3_scope.refuse_withheld(path) == path.lstrip("/")


@pytest.mark.parametrize("path", [
    QUARANTINE_PREFIX, RAW_MIME_PREFIX, QUARANTINE_PREFIX + "x.bin", RAW_MIME_PREFIX + "m.eml",
    "/" + QUARANTINE_PREFIX + "x", "Invoice/../" + RAW_MIME_PREFIX + "m.eml",
    "mail-intake/", "mail-intake", "mail", "m",
])
def test_withheld_stores_their_contents_and_their_ancestors_are_refused(path):
    with pytest.raises(ValueError, match="withheld"):
        s3_scope.refuse_withheld(path)


def test_empty_and_climbing_paths_are_refused():
    for bad in ("", "   ", "../x", "a/../../x"):
        with pytest.raises(ValueError):
            s3_scope.refuse_withheld(bad)


def test_all_checks_every_entry_and_keeps_order():
    assert s3_scope.refuse_withheld_all(["b/1.pdf", "a/2.pdf"]) == ["b/1.pdf", "a/2.pdf"]
    with pytest.raises(ValueError):
        s3_scope.refuse_withheld_all(["b/1.pdf", QUARANTINE_PREFIX + "z"])


def test_the_message_never_echoes_the_callers_value():
    with pytest.raises(ValueError) as exc:
        s3_scope.refuse_withheld("mail-intake/quarantine/SECRET-NAME.bin")
    assert "SECRET-NAME" not in str(exc.value)


def test_any_skips_blanks_and_refuses_a_withheld_one():
    s3_scope.refuse_withheld_any(None, "", "Invoice/")
    with pytest.raises(ValueError, match="withheld"):
        s3_scope.refuse_withheld_any(None, QUARANTINE_PREFIX + "x.pdf")


def test_a_trailing_slash_is_kept_so_a_prefix_stays_a_prefix():
    assert s3_scope.normalise("Invoice/2026/") == "Invoice/2026/"
    assert s3_scope.normalise("Invoice//2026/") == "Invoice/2026/"
