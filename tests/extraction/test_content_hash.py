import hashlib

from src.services.extraction.content_hash import (
    compute_content_hash, quality_action_from_result,
)


def test_compute_content_hash_local_file(tmp_path):
    f = tmp_path / "doc.bin"
    f.write_bytes(b"hello world")
    expected = hashlib.sha256(b"hello world").hexdigest()
    assert compute_content_hash(str(f)) == expected


def test_compute_content_hash_missing_returns_none(tmp_path, monkeypatch):
    # nonexistent local path; force no S3 bucket so resolver raises → None
    monkeypatch.delenv("S3_BUCKET_NAME", raising=False)
    monkeypatch.setattr(
        "src.services.extraction.parser._s3_bucket", lambda: None)
    assert compute_content_hash(str(tmp_path / "nope.bin")) is None


def test_quality_needs_review_low_confidence():
    assert quality_action_from_result(
        {"confidence": 0.5, "pk": "INV1", "missing": []}) == "needs_review"


def test_quality_needs_review_no_pk():
    assert quality_action_from_result(
        {"confidence": 0.99, "pk": "", "missing": []}) == "needs_review"


def test_quality_needs_review_missing_required():
    assert quality_action_from_result(
        {"confidence": 0.99, "pk": "INV1", "missing": ["supplier_name"]}) == "needs_review"


def test_quality_clean_returns_none():
    assert quality_action_from_result(
        {"confidence": 0.95, "pk": "INV1", "missing": []}) is None


def test_quality_reads_doc_pk_alias():
    # renovation dispatch returns doc_pk, not pk
    assert quality_action_from_result(
        {"confidence": 0.95, "doc_pk": "INV1", "missing": []}) is None
