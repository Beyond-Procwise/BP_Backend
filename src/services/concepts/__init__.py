"""The document and relationship vocabulary.

proc.bp_concept and proc.bp_document_type are the source of truth; seed.py is a
seed that a test asserts matches them. Built in the shape of
proc.bp_uom_canonical (deploy/sql/2026-08-07_uom_canonical.sql): aliases on the
concept row, status active/proposed/rejected where a proposed row NEVER
resolves, and a confirmation trail for the human who promoted it.
"""
from __future__ import annotations

from .seed import CONCEPTS, DOCUMENT_TYPES, Concept, DocumentType

__all__ = ["CONCEPTS", "DOCUMENT_TYPES", "Concept", "DocumentType"]
