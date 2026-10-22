"""Which S3 paths a caller may name for extraction (1.3c / 1.3e).

Two stores under the mail-intake prefix are WITHHELD from extraction: the quarantine (bytes the
safety gate rejected: encrypted files, macro documents, injections) and the raw MIME store (every
message as received, rejected ones included). A caller-supplied key or prefix that is one of them, is
inside one of them, or is an ANCESTOR of one (``mail-intake/`` or just ``mail``, which a listing
would expand into every withheld object) is refused before anything is listed or downloaded.

One function, used by every entry point that takes a key or prefix from a caller: the extract-from-s3
route and the extraction agent's process-document inputs. A guard that lives in one route is a guard
the next route forgets.
"""
from __future__ import annotations

import posixpath
from typing import Iterable

from src.services.mail_intake.store import QUARANTINE_PREFIX, RAW_MIME_PREFIX

WITHHELD: tuple[str, ...] = (QUARANTINE_PREFIX, RAW_MIME_PREFIX)


def normalise(path: str) -> str:
    """The path as compared: no leading slash, ``..`` collapsed, so ``a/../mail-intake/quarantine/``
    cannot slip past. Raises for an empty path or one that climbs out of its prefix."""
    value = (path or "").strip().lstrip("/")
    if not value:
        raise ValueError("an S3 key or prefix must not be empty")
    normalised = posixpath.normpath(value).lstrip("/")
    if normalised.startswith(".."):
        raise ValueError("an S3 key or prefix must not traverse outside its prefix")
    if value.endswith("/") and not normalised.endswith("/"):
        normalised += "/"      # a prefix stays a prefix; normpath would have made it look like a key
    # Case, backslashes and percent-encoding are deliberately NOT normalised: S3 keys and listing
    # prefixes are byte-literal, so "MAIL-INTAKE/..." or "mail-intake%2Fquarantine" name different
    # (non-existent) objects, not the withheld ones. Only ".." and "//" could disguise a withheld path.
    return normalised


def refuse_withheld(path: str) -> str:
    """Return the normalised path, or raise ValueError if it names, lies under, or lies ABOVE a
    withheld store. The message names the store, never the caller's value."""
    normalised = normalise(path)
    with_slash = normalised if normalised.endswith("/") else normalised + "/"
    for withheld in WITHHELD:
        if with_slash.startswith(withheld):
            raise ValueError(f"the path resolves under {withheld}, which holds content withheld from extraction")
        if withheld.startswith(normalised):
            # An ancestor (any leading part of the store's prefix): listing it would enumerate every
            # withheld object beneath. Word kept: callers and tests match on "ancestor".
            raise ValueError(f"the path is an ancestor of {withheld}, which holds content withheld from extraction")
    return normalised


def refuse_withheld_all(paths: Iterable[str]) -> list[str]:
    """Every path checked; the first withheld one raises. Order kept."""
    return [refuse_withheld(p) for p in paths]


def refuse_withheld_any(*paths: str | None) -> None:
    """For a request that names a prefix and/or a key, some of them blank: every non-blank one is
    checked. Routes call this first, so a refusal is a 400 the caller sees rather than a failed run."""
    for path in paths:
        if path and str(path).strip():
            refuse_withheld(str(path))
