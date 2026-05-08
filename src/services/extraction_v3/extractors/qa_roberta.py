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
import re
import threading
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from src.services.extraction_v3.extractors.base import Extractor
from src.services.extraction_v3.schemas.candidate import Candidate
from src.services.extraction_v3.schemas.parsed_document import ParsedDocument
from src.services.extraction_v3.yaml_schema.loader import DocSchema, FieldSpec
from src.services.extraction_v3.yaml_schema.registry import register_extractor

MODEL_NAME = "deepset/roberta-base-squad2"
MIN_CONFIDENCE = 0.4

# Garbage answers that roberta often returns when the context has no real answer.
# Markdown heading tokens (##, ###), section markers, punctuation-only strings.
_GARBAGE_RE = re.compile(r"^#+$|^[-=\*]{1,}$|^\W+$")

_tokenizer = None
_model = None
_lock = threading.Lock()


def _get_qa():
    """Return (tokenizer, model) tuple. Lazy-loaded on GPU."""
    global _tokenizer, _model
    if _tokenizer is not None and _model is not None:
        return _tokenizer, _model
    with _lock:
        if _tokenizer is None or _model is None:
            assert torch.cuda.is_available(), "QA-roberta requires GPU per spec C2"
            _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            _model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME).to("cuda").eval()
    return _tokenizer, _model


def _answer_span(question: str, context: str, max_length: int = 384) -> tuple[str, float]:
    """Run QA, return (answer, score). Returns ("", 0.0) for no-answer.

    SQuAD2 null-score guard: only reject when the model is strongly confident
    the question has no answer (null_score exceeds best span score by more
    than NULL_SCORE_DIFF_THRESHOLD).  A tight threshold of 0.0 causes false
    rejections on long, noisy contexts (markdown tables, headers) because the
    CLS logit is inflated relative to any single span.  Using a generous
    threshold (5.0) lets clearly-answerable fields pass through while still
    blocking questions the model is overwhelmingly unsure about.

    Confidence is computed as the average sigmoid of the best start/end
    logits.  Unlike softmax over the full sequence length, sigmoid is
    context-length independent and produces values in (0, 1) that are
    comparable across documents of different sizes.
    """
    tokenizer, model = _get_qa()
    # Tokenize with question + context as a sentence pair
    inputs = tokenizer(
        question, context,
        return_tensors="pt", truncation="only_second", max_length=max_length,
        return_offsets_mapping=True, padding="max_length",
    )
    offset_mapping = inputs.pop("offset_mapping")[0].cpu().numpy()
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        out = model(**inputs)

    start_logits = out.start_logits[0].cpu()
    end_logits = out.end_logits[0].cpu()

    # SQuAD2 no-answer: position 0 is the [CLS] token; if (start, end) = (0, 0) is the
    # best answer, that signals "no answer"
    null_score = float(start_logits[0] + end_logits[0])

    # Find best answer span (start <= end, end - start <= 30 tokens)
    sequence_ids = inputs["input_ids"][0].cpu().numpy()
    # The context tokens are the second sequence; restrict span search to those positions
    # (Mask out positions outside the context.)
    n = len(sequence_ids)
    # Find context bounds: tokens after the first [SEP]
    sep_token_id = tokenizer.sep_token_id
    sep_positions = [i for i, t in enumerate(sequence_ids) if t == sep_token_id]
    if len(sep_positions) >= 1:
        context_start = sep_positions[0] + 1
    else:
        context_start = 0

    best_score = -float("inf")
    best_start, best_end = 0, 0
    for start in range(context_start, n):
        if offset_mapping[start][0] == 0 and offset_mapping[start][1] == 0:
            continue  # special tokens have offset (0, 0)
        for end in range(start, min(start + 30, n)):
            if offset_mapping[end][0] == 0 and offset_mapping[end][1] == 0:
                continue
            score = float(start_logits[start] + end_logits[end])
            if score > best_score:
                best_score, best_start, best_end = score, start, end

    # Only reject as no-answer when null_score strongly dominates the best span.
    # A lenient threshold (5.0) avoids false rejections on long, markdown-heavy
    # contexts where CLS logits are systematically inflated.
    NULL_SCORE_DIFF_THRESHOLD = 5.0
    if null_score - best_score > NULL_SCORE_DIFF_THRESHOLD:
        return "", 0.0

    # Convert (start, end) token positions to character offsets in the original context
    char_start = int(offset_mapping[best_start][0])
    char_end = int(offset_mapping[best_end][1])
    answer_text = context[char_start:char_end].strip()
    # Confidence: average sigmoid of best start/end logits.
    # Sigmoid is context-length independent (unlike softmax over the full sequence)
    # and produces values in (0, 1) that are comparable across documents of all sizes.
    score = float(0.5 * (torch.sigmoid(start_logits[best_start]) + torch.sigmoid(end_logits[best_end])))
    return answer_text, score


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
        candidates = []
        context = parsed.full_text
        for field in active:
            question = _question_for_field(field)
            try:
                answer, score = _answer_span(question, context)
            except Exception:
                continue
            if not answer or score < MIN_CONFIDENCE:
                continue
            if answer not in parsed.full_text:
                continue
            # Reject garbage answers: markdown headings (#, ##), punctuation-only strings.
            # Roberta sometimes returns these when the context has no real answer span.
            if _GARBAGE_RE.match(answer.strip()):
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
