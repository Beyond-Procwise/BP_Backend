"""Extractive QA gap-filler using deepset/roberta-base-squad2.

Given parsed.full_text and a question derived from a schema field's first
canonical_label ("What is the {label}?"), the model returns either an
answer SPAN with confidence or "no_answer". Spans are substrings by
construction — substring guarantee holds.

Used as a fallback extractor: in the L3 orchestrator, QA-roberta candidates
only commit when no higher-priority extractor produced a candidate for the
same field.
"""
from __future__ import annotations
import threading
import torch
from transformers import pipeline
from src.services.extraction_v3.extractors.base import Extractor
from src.services.extraction_v3.schemas.candidate import Candidate
from src.services.extraction_v3.schemas.parsed_document import ParsedDocument
from src.services.extraction_v3.yaml_schema.loader import DocSchema, FieldSpec
from src.services.extraction_v3.yaml_schema.registry import register_extractor

MODEL_NAME = "deepset/roberta-base-squad2"
MIN_CONFIDENCE = 0.4

_qa = None
_lock = threading.Lock()


def _get_qa():
    global _qa
    if _qa is not None:
        return _qa
    with _lock:
        if _qa is None:
            assert torch.cuda.is_available(), "QA-roberta requires GPU per spec C2"
            _qa = pipeline(
                "question-answering",
                model=MODEL_NAME,
                tokenizer=MODEL_NAME,
                device=0,  # cuda
                handle_impossible_answer=True,  # honour SQuAD2's no-answer
            )
    return _qa


def _question_for_field(field: FieldSpec) -> str:
    """Build a natural-language question from the field's canonical labels."""
    if field.canonical_labels:
        # Use the first canonical label as the question's noun phrase
        return f"What is the {field.canonical_labels[0]}?"
    return f"What is the {field.name.replace('_', ' ')}?"


def _find_bbox_for_text(parsed: ParsedDocument, text: str):
    text_l = text.lower().strip()
    for page in parsed.pages:
        for tok in page.tokens:
            if text_l in tok.text.lower():
                return (page.index, tok.bbox)
    return None


@register_extractor("qa_roberta")
class QARobertaExtractor(Extractor):

    def produce_candidates(self, parsed: ParsedDocument, schema: DocSchema) -> list[Candidate]:
        active = [f for f in schema.fields if "qa_roberta" in f.extractors]
        if not active:
            return []
        if not parsed.full_text.strip():
            return []
        qa = _get_qa()
        candidates = []
        # Truncate context to QA model's max input (RoBERTa: 512 tokens; ~2000 chars). For longer
        # docs, run QA on the full text and let the pipeline auto-truncate (its handle_long_documents
        # behavior depends on transformers version).
        context = parsed.full_text
        for field in active:
            question = _question_for_field(field)
            try:
                result = qa(question=question, context=context)
            except Exception:
                continue
            score = float(result.get("score", 0.0))
            answer = (result.get("answer") or "").strip()
            if not answer or score < MIN_CONFIDENCE:
                continue
            # Anti-hallucination: the answer must be a substring of context (parsed.full_text).
            # Defensive check even though SQuAD model returns spans — future-proofing against
            # model contract changes.
            if answer not in parsed.full_text:
                continue
            bbox_loc = _find_bbox_for_text(parsed, answer)
            page_idx, b = bbox_loc if bbox_loc else (0, (0.0, 0.0, 0.0, 0.0))
            candidates.append(Candidate(
                field=field.name,
                value=answer,
                page=page_idx,
                bbox=b,
                evidence_text=answer,
                model="qa_roberta",
                confidence=score,
            ))
        return candidates
