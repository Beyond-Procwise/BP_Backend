"""Abstract base for L2 candidate-generating extractors.

An Extractor consumes a ParsedDocument + the doc-type's DocSchema and emits
zero or more Candidate records, each carrying its evidence span. Every
concrete subclass is registered into the runtime registry via the
@register_extractor decorator (from yaml_schema.registry).

The substring guarantee — every Candidate.evidence_text must be a substring
of ParsedDocument.full_text — is enforced by the L3 orchestrator, not here;
extractors are trusted to produce truthful evidence (a unit test will
verify this on every concrete extractor).
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from src.services.extraction_v3.schemas.parsed_document import ParsedDocument
from src.services.extraction_v3.schemas.candidate import Candidate
from src.services.extraction_v3.yaml_schema.loader import DocSchema


class Extractor(ABC):
    """L2 candidate generator. Concrete subclasses are registered by name."""

    @abstractmethod
    def produce_candidates(
        self,
        parsed: ParsedDocument,
        schema: DocSchema,
    ) -> list[Candidate]:
        """Return zero or more candidates for the schema's fields.

        Each candidate must carry:
          - field: the schema field name (e.g. "invoice_id" or "line_items[3].amount")
          - value: the extracted string (pre-type-coercion)
          - evidence_text: a substring of parsed.full_text proving the value
          - page, bbox: location information for the evidence
          - model: this extractor's name (matches its @register_extractor key)
          - confidence: in [0, 1]
        """
        raise NotImplementedError
