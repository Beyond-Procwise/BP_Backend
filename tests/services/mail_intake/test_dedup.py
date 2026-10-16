"""Duplicate detection at two levels: identical bytes, and the same document
arriving through different forwards with different MIME wrappers."""
from src.services.mail_intake import dedup


def test_identical_content_resolves_to_the_existing_artifact():
    result = dedup.resolve(
        "abc123", None, lambda digest, logical: "ART-1" if digest == "abc123" else None
    )
    assert result.is_duplicate
    assert result.duplicate_of == "ART-1"


def test_novel_content_is_not_a_duplicate():
    result = dedup.resolve("novel", None, lambda digest, logical: None)
    assert not result.is_duplicate
    assert result.duplicate_of is None


def test_logical_duplicate_is_caught_when_bytes_differ():
    # Same supplier, same document type, same reference — re-wrapped by a
    # forwarding client, so the hash differs but the document does not.
    logical = dedup.logical_key("SUP-001", "quote", "QT-4471")
    result = dedup.resolve(
        "different-hash",
        logical,
        lambda digest, key: "ART-9" if key == logical else None,
    )
    assert result.is_duplicate
    assert result.duplicate_of == "ART-9"


def test_logical_key_needs_all_three_parts():
    assert dedup.logical_key("SUP-001", "quote", None) is None
    assert dedup.logical_key(None, "quote", "QT-1") is None
    assert dedup.logical_key("SUP-001", None, "QT-1") is None


def test_logical_key_is_normalised():
    assert dedup.logical_key("SUP-001", "Quote", " qt-4471 ") == dedup.logical_key(
        "sup-001", "quote", "QT-4471"
    )
