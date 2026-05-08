"""LayoutLMv3-backed header-field extractor.

PLAN 1 IMPLEMENTATION: This uses canonical-label proximity over the
ParsedDocument's tokens (which carry layout-aware bboxes from Docling /
PaddleOCR). The pre-trained microsoft/layoutlmv3-base has no procurement
labels, so token classification is deferred to Plan 2 (fine-tune).

The "LayoutLMv3" name is preserved for module identity — this slot will
be replaced by the fine-tuned token classifier in Plan 2 without
disturbing downstream consumers (since the contract is just "produce
Candidates").

Algorithm (canonical-label proximity):
  1. For each schema field with "layoutlmv3" in its extractors list,
     iterate over its canonical_labels.
  2. For each label, scan all tokens in every page for a fuzzy text match
     (rapidfuzz.fuzz.ratio >= LABEL_MATCH_MIN_SCORE, or a contains check
     for partial multi-word overlap).
  3. The value token is the closest non-label token that is either:
     - immediately to the right of the label token on the same line, or
     - directly below it within a small vertical band.
  4. Emit a Candidate with confidence = (label_match_score / 100) * 0.9,
     capped to [0.10, 0.95] to reflect the Plan 1 baseline uncertainty.

Substring guarantee: every Candidate.evidence_text is taken DIRECTLY from
a token's text and is verified to appear literally in parsed.full_text before
being emitted.  Note that Docling's markdown exporter HTML-escapes certain
characters (e.g. & → &amp;) so a token's literal text may not always appear
in full_text verbatim; those tokens are silently skipped to preserve the
guarantee rather than emitting an ungrounded candidate.

DO NOT import transformers at module level — Plan 1 does not run inference
through any model heads. Fine-tuned token classification lands in Plan 2.
"""
from __future__ import annotations

from rapidfuzz import fuzz

from src.services.extraction_v3.extractors.base import Extractor
from src.services.extraction_v3.schemas.candidate import Candidate
from src.services.extraction_v3.schemas.parsed_document import ParsedDocument, Token
from src.services.extraction_v3.yaml_schema.loader import DocSchema, FieldSpec
from src.services.extraction_v3.yaml_schema.registry import register_extractor


# ---------------------------------------------------------------------------
# Proximity constants — tuned for typical A4/Letter invoice/PO/quote layouts.
# Revisit if adding doc types with unusual typography (e.g. multi-column forms).
# ---------------------------------------------------------------------------
LABEL_MATCH_MIN_SCORE: int = 78      # rapidfuzz ratio threshold [0–100]
RIGHT_OF_LABEL_MAX_DX: float = 250.0  # px; max horizontal gap to the right of a label
BELOW_LABEL_MAX_DY: float = 50.0      # px; max vertical gap below a label
SAME_LINE_TOLERANCE: float = 1.0      # multiplier of label height for same-line check
BELOW_X_TOLERANCE: float = 100.0     # px; max horizontal offset for "below" candidates


@register_extractor("layoutlmv3")
class LayoutLMv3Extractor(Extractor):
    """Plan 1 baseline: canonical-label proximity extractor.

    Registered as "layoutlmv3" so the YAML schema's extractor lists resolve
    to this class.  Plan 2 will replace the body of this class with a
    fine-tuned LayoutLMv3ForTokenClassification inference pass while keeping
    the class name and registration key unchanged.
    """

    def produce_candidates(
        self,
        parsed: ParsedDocument,
        schema: DocSchema,
    ) -> list[Candidate]:
        candidates: list[Candidate] = []
        for field in schema.fields:
            if "layoutlmv3" not in field.extractors:
                continue
            candidates.extend(self._candidates_for_field(parsed, field))
        return candidates

    # ---------------------------------------------------------------------- #
    # Internal helpers                                                         #
    # ---------------------------------------------------------------------- #

    def _candidates_for_field(
        self,
        parsed: ParsedDocument,
        field: FieldSpec,
    ) -> list[Candidate]:
        out: list[Candidate] = []
        for label in field.canonical_labels:
            for page in parsed.pages:
                for label_token, label_score in self._find_label_tokens(page.tokens, label):
                    value_token = self._next_value_token(page.tokens, label_token)
                    if value_token is None:
                        continue
                    evidence = value_token.text.strip()
                    # Substring guard — Docling's markdown export HTML-escapes
                    # certain characters (e.g. & → &amp;) so a token's literal
                    # text may not appear in full_text verbatim.  Skip such
                    # tokens rather than emit an ungrounded candidate.
                    if evidence not in parsed.full_text:
                        continue
                    # Confidence: normalised match score × baseline discount.
                    # Capped below 1.0 to signal Plan 1 provenance.
                    confidence = min(0.95, max(0.10, (label_score / 100.0) * 0.9))
                    out.append(
                        Candidate(
                            field=field.name,
                            value=evidence,
                            page=value_token.page,
                            bbox=value_token.bbox,
                            evidence_text=evidence,
                            model="layoutlmv3",
                            confidence=confidence,
                        )
                    )
        return out

    def _find_label_tokens(
        self,
        tokens: list[Token],
        label: str,
    ) -> list[tuple[Token, float]]:
        """Return tokens whose text is a fuzzy match for *label*, with scores."""
        matches: list[tuple[Token, float]] = []
        label_lower = label.lower().strip()
        for tok in tokens:
            tok_text = tok.text.lower().strip().rstrip(":").rstrip(".")
            score = fuzz.ratio(tok_text, label_lower)
            if score >= LABEL_MATCH_MIN_SCORE:
                matches.append((tok, float(score)))
            elif label_lower in tok_text or tok_text in label_lower:
                # Contains-check covers multi-word labels that happen to be
                # embedded in a larger token (e.g. "Invoice Number:" in a
                # single OCR span) or labels split across tokens.
                matches.append((tok, max(float(score), 80.0)))
        return matches

    def _next_value_token(
        self,
        tokens: list[Token],
        label_tok: Token,
    ) -> Token | None:
        """Return the value token geometrically nearest to *label_tok*.

        Search order (lower sort key = preferred):
          1. Same line, immediately to the right  (sort key = dx)
          2. Below the label, similar x-range     (sort key = dy + 1000)

        The 1000 offset ensures right-of-label always beats below-label when
        both are present — matching the reading-order convention for key:value
        pairs in invoices and purchase orders.
        """
        lx0, ly0, lx1, ly1 = label_tok.bbox
        label_height = max(ly1 - ly0, 1.0)  # guard against zero-height bbox

        ranked: list[tuple[float, Token]] = []
        for tok in tokens:
            if tok is label_tok:
                continue
            tx0, ty0, tx1, ty1 = tok.bbox

            # --- same-line heuristic ---
            same_line = abs(ty0 - ly0) < label_height * SAME_LINE_TOLERANCE
            dx = tx0 - lx1
            if same_line and 0.0 < dx < RIGHT_OF_LABEL_MAX_DX:
                ranked.append((dx, tok))
                continue

            # --- below-the-label heuristic ---
            dy = ty0 - ly1
            if 0.0 < dy < BELOW_LABEL_MAX_DY and abs(tx0 - lx0) < BELOW_X_TOLERANCE:
                ranked.append((dy + 1000.0, tok))

        if not ranked:
            return None
        ranked.sort(key=lambda pair: pair[0])
        return ranked[0][1]
