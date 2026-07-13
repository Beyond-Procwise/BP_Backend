"""Tests for the offline health-check audit helpers (FINDINGS.md F1 follow-up).

Covers two fixes:
  1. The hallucination audit must use the SAME format-tolerant grounding as the
     live pipeline, and must check ALL snapshots that share a doc_pk, so it stops
     over-reporting (reformatted-but-correct values) and stops false-flagging on
     doc_pk collisions (value present in a sibling snapshot).
  2. doc_pk collision detection: a doc_pk that maps to multiple DISTINCT source
     documents (different file AND different text) is a genuine collision.
"""
from scripts.extraction_health_check import (
    _audit_is_violation,
    _classify_collision,
)


class TestAuditIsViolation:
    FT = "Invoice Number: INV-100\nInvoice Date: 15 Feb 2024\nNet £27,202.00\n"

    def test_exact_value_is_not_a_violation(self):
        assert _audit_is_violation("INV-100", "Invoice Number: INV-100", "vlm", [self.FT]) is False

    def test_reformatted_value_is_not_a_violation(self):
        # ISO date vs textual doc date — tolerant match, not a hallucination
        assert _audit_is_violation("2024-02-15", "2024-02-15", "regex", [self.FT]) is False

    def test_genuine_absence_is_a_violation(self):
        assert _audit_is_violation("Globex Corp", "Globex Corp", "judge", [self.FT]) is True

    def test_grounded_in_a_sibling_collision_snapshot_is_not_a_violation(self):
        # doc_pk collision: value belongs to the OTHER snapshot sharing this doc_pk
        other = "Quote TEC-QTR-2022-Q3\nDate 11 Jul 2022\n"
        assert _audit_is_violation("11 Jul 2022", "Date 11 Jul 2022", "judge",
                                   [self.FT, other]) is False

    def test_no_snapshots_cannot_verify_so_not_a_violation(self):
        assert _audit_is_violation("anything", "anything", "vlm", []) is False

    def test_pipeline_recovery_derivation_is_not_a_violation(self):
        assert _audit_is_violation("30000.0", "", "pipeline_recovery", [self.FT]) is False


class TestClassifyCollision:
    def test_same_file_reextraction_is_not_a_collision(self):
        rows = [("a.pdf", "hash1"), ("a.pdf", "hash1")]
        assert _classify_collision(rows) is False

    def test_distinct_files_and_texts_is_a_collision(self):
        rows = [("TEC-Q.pdf", "hashA"), ("TEC-QTR.pdf", "hashB")]
        assert _classify_collision(rows) is True

    def test_single_document_is_not_a_collision(self):
        assert _classify_collision([("a.pdf", "hash1")]) is False

    def test_same_text_different_filename_is_not_a_collision(self):
        # identical content re-uploaded under two names — not a genuine id clash
        rows = [("a.pdf", "hashX"), ("a_copy.pdf", "hashX")]
        assert _classify_collision(rows) is False
