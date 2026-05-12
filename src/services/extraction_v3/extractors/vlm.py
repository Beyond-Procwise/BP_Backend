"""VLM-based extractor — one call per document, schema-aware prompt, JSON output.

The model (Qwen2.5-VL-7B-Instruct) is already loaded by judge/qwen_vl.py.
This module reuses that singleton.

Replaces the 6-extractor parallel stack with a single Qwen call per document.
The model reads the rasterized document page(s) and returns a structured JSON
object. Every extracted value is substring-checked against the parsed full_text
to enforce the hallucination-rejection contract before becoming a Candidate.

Thread safety: Qwen's visual encoder (conv3d) and CUDA cuDNN context are not
thread-safe across simultaneous inference calls. _VLM_LOCK ensures only one
Qwen inference runs at a time across all worker threads.
"""
from __future__ import annotations

import json
import logging
import re
import threading
from pathlib import Path

from src.services.extraction_v3.extractors.base import Extractor
from src.services.extraction_v3.schemas.parsed_document import ParsedDocument
from src.services.extraction_v3.schemas.candidate import Candidate
from src.services.extraction_v3.yaml_schema.loader import DocSchema
from src.services.extraction_v3.yaml_schema.registry import register_extractor
from src.services.extraction_v3.judge.qwen_vl import qwen_vl_extract

log = logging.getLogger(__name__)

# Global lock: only one Qwen inference at a time to avoid cuDNN context conflicts
# when multiple worker threads try to run GPU ops simultaneously.
_VLM_LOCK = threading.Lock()

MAX_PAGES = 3  # invoices/POs/quotes usually fit in 1-3 pages


def _build_schema_prompt(schema: DocSchema, doc_text: str | None = None) -> str:
    """Build a schema-aware extraction prompt listing every field + type + canonical labels.

    Args:
        schema: The DocSchema describing fields to extract.
        doc_text: If provided, the full document text is embedded directly in the
            prompt (text-only mode for DOCX / rasterization failures). When None
            the model reads the image visually.
    """
    fields_block = []
    for f in schema.fields:
        labels = ", ".join(f.canonical_labels[:5]) if f.canonical_labels else ""
        type_hint = f.type
        fields_block.append(f"  - {f.name} ({type_hint}): look for labels like {labels}")

    line_block = ""
    if schema.line_items:
        line_fields = ", ".join(
            f"{lf.name}: {lf.type}" for lf in schema.line_items.fields
        )
        line_block = f"\n\nline_items: array of {{{line_fields}}}"

    if doc_text is not None:
        # Text-only mode: embed document text directly (DOCX or rasterization failure)
        # Cap at 6 000 chars to stay within Qwen's context budget.
        text_snippet = doc_text[:6000].strip()
        return f"""You are extracting data from a {schema.doc_type} document.

The full document text is provided below. Extract the requested fields.

DOCUMENT TEXT:
{text_snippet}

Return ONLY a JSON object with these fields:

{chr(10).join(fields_block)}{line_block}

Rules:
- Use ONLY values that appear verbatim in the DOCUMENT TEXT above
- If a field is not present in the document text, return null for that field
- For monetary amounts: return just the number (no currency symbol), e.g. 1234.50
- For dates: return YYYY-MM-DD format if possible, otherwise as printed
- For currency: return 3-letter ISO code (USD, GBP, EUR, NZD, AUD, CAD, etc.)
- For supplier_name: return the company name exactly as it appears in the text
- For region: return the state/province code or name from the supplier address
- Do NOT invent values. Do NOT paraphrase. Values must be substrings of the text above.
- line_items should be an array of objects; return [] if no line items are present

Respond with ONLY the JSON object. No prose, no markdown code fences."""

    # Visual mode: model reads the rasterized document image
    return f"""You are extracting data from a {schema.doc_type} document.

Read the document image and return ONLY a JSON object with these fields:

{chr(10).join(fields_block)}{line_block}

Rules:
- Use ONLY values that are literally visible in the document image
- If a field is not present in the document, return null for that field
- For monetary amounts: return just the number (no currency symbol), e.g. 1234.50
- For dates: return YYYY-MM-DD format
- For currency: return 3-letter ISO code (USD, GBP, EUR, NZD, AUD, CAD, etc.)
- For supplier_name: return the company name exactly as printed (not an address or person name)
- For region: return the state/province code or name visible in the supplier address block
- Do NOT invent values. Do NOT paraphrase. Do NOT compute derived values — just read what's printed.
- line_items should be an array of objects; return [] if no line items are visible

Respond with ONLY the JSON object. No prose, no markdown code fences."""


def _rasterize_page(source_file: str, page_idx: int):
    """Render a specific page from the source doc as a PIL.Image.RGB, or None on failure."""
    p = Path(source_file)
    if not p.exists():
        # Try resolving relative paths against the project root
        for base in (Path("/home/muthu/PycharmProjects/BP_Backend"), Path.cwd()):
            cand = base / source_file
            if cand.exists():
                p = cand
                break
    if not p.exists():
        return None

    suffix = p.suffix.lower()
    try:
        if suffix == ".pdf":
            from pdf2image import convert_from_path
            from PIL import Image as _PIL
            # 96 dpi — sufficient for Qwen2.5-VL text recognition while keeping
            # image token count low enough to avoid CUDA OOM on the A10G.
            imgs = convert_from_path(str(p), dpi=96, first_page=page_idx + 1, last_page=page_idx + 1)
            if not imgs:
                return None
            img = imgs[0].convert("RGB")
            # Cap at 1024px on the long side — reduces token count significantly
            w, h = img.size
            max_side = 1024
            if max(w, h) > max_side:
                scale = max_side / max(w, h)
                img = img.resize((int(w * scale), int(h * scale)), _PIL.LANCZOS)
            return img
        if suffix in (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"):
            from PIL import Image as _PIL
            img = _PIL.open(p).convert("RGB")
            w, h = img.size
            max_side = 1024
            if max(w, h) > max_side:
                scale = max_side / max(w, h)
                img = img.resize((int(w * scale), int(h * scale)), _PIL.LANCZOS)
            return img
        # DOCX / other formats — no rasterization available; caller falls back to text-only
        return None
    except Exception as exc:
        log.warning("vlm_extractor: rasterize failed for %s page %d: %s", source_file, page_idx, exc)
        return None


def _parse_json(raw: str) -> dict | None:
    """Extract the first complete JSON object from raw LLM output."""
    if not raw:
        return None
    # Strip markdown code fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.MULTILINE)
    raw = re.sub(r"\s*```$", "", raw.strip(), flags=re.MULTILINE)

    start = raw.find("{")
    if start < 0:
        return None
    depth, end, in_str, esc = 0, -1, False, False
    for i in range(start, len(raw)):
        ch = raw[i]
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end < 0:
        return None
    try:
        return json.loads(raw[start:end])
    except json.JSONDecodeError as exc:
        log.debug("vlm_extractor: JSON parse error: %s — raw=%r", exc, raw[start:end][:200])
        return None


def _is_substring_of(value: str, full_text: str) -> bool:
    """Check if value appears in full_text. Handles whitespace/punctuation drift."""
    if not value or not full_text:
        return False
    # Exact match
    if value in full_text:
        return True
    # Whitespace-normalized match
    v_norm = re.sub(r"\s+", " ", value).strip()
    t_norm = re.sub(r"\s+", " ", full_text)
    if v_norm in t_norm:
        return True
    # For short numeric / date values (≤20 chars), allow any whitespace stripping
    if len(value) <= 20:
        v_stripped = re.sub(r"\s", "", value)
        t_stripped = re.sub(r"\s", "", full_text)
        if v_stripped and v_stripped in t_stripped:
            return True
    return False


def extract_with_vlm(
    parsed: ParsedDocument,
    schema: DocSchema,
    source_file: str | None = None,
) -> list[Candidate]:
    """Single Qwen-VL call. Parse JSON. Substring-check every value. Return candidates.

    This is the main entry point called by the pipeline. It rasterizes up to
    MAX_PAGES pages and sends the first page (primary content) to Qwen with
    the schema-aware prompt. Every returned value is substring-checked against
    parsed.full_text — values that fail are REJECTED (hallucination guard).
    """
    candidates: list[Candidate] = []

    # Rasterize up to MAX_PAGES pages; use first available image
    image = None
    if source_file:
        page_count = max(1, len(parsed.pages)) if parsed.pages else 1
        for page_idx in range(min(page_count, MAX_PAGES)):
            img = _rasterize_page(source_file, page_idx)
            if img is not None:
                image = img
                break

    text_only = image is None
    if text_only:
        # DOCX or rasterization failure — inject the parsed full text into the
        # prompt so Qwen extracts from text, not from a blank/meaningless image.
        # A minimal 32x32 white image is still required by the VL processor API.
        log.info(
            "vlm_extractor: no image for %s — using text-only mode (full_text injected into prompt)",
            source_file,
        )
        try:
            from PIL import Image as _PIL
            image = _PIL.new("RGB", (32, 32), color=(255, 255, 255))
        except Exception:
            log.error("vlm_extractor: PIL not available, cannot build placeholder")
            return []

    # Build prompt: text-only mode embeds full_text; visual mode reads the image.
    doc_text_for_prompt = parsed.full_text if text_only else None
    prompt = _build_schema_prompt(schema, doc_text=doc_text_for_prompt)

    # Serialize Qwen inference — only one call at a time to prevent cuDNN
    # context conflicts when multiple worker threads compete for the GPU.
    # max_new_tokens=2048 to cover documents with many line items (9 service
    # lines × 7 fields × ~15 tokens each ≈ 945 tokens, plus header ≈ 300,
    # total ~1250 — 2048 gives headroom for verbose field values).
    with _VLM_LOCK:
        raw_response = qwen_vl_extract(image, prompt, max_new_tokens=2048)

    if not raw_response:
        log.warning("vlm_extractor: empty response for %s", source_file)
        return []

    data = _parse_json(raw_response)
    if not data:
        log.warning("vlm_extractor: failed to parse JSON for %s — raw=%r", source_file, raw_response[:300])
        return []

    log.info("vlm_extractor: parsed %d top-level keys from Qwen response for %s", len(data), source_file)

    # Build candidates from header fields
    for f in schema.fields:
        value = data.get(f.name)
        if value is None or value == "":
            continue
        value_str = str(value).strip()
        if not value_str:
            continue

        # Substring guarantee — must be in parsed.full_text (hallucination rejection)
        if not _is_substring_of(value_str, parsed.full_text):
            log.warning(
                "vlm_extractor: %s=%r not in source text — REJECTING (hallucination guard)",
                f.name, value_str,
            )
            continue

        candidates.append(Candidate(
            field=f.name,
            value=value_str,
            page=0,
            bbox=(0.0, 0.0, 0.0, 0.0),
            evidence_text=value_str,
            model="qwen_vlm",
            confidence=0.85,
        ))
        log.debug("vlm_extractor: committed %s=%r", f.name, value_str)

    # Line items
    if schema.line_items:
        lines = data.get("line_items")
        log.debug(
            "vlm_extractor: line_items raw from Qwen=%r for %s",
            type(lines).__name__ + (f"[{len(lines)}]" if isinstance(lines, list) else ""),
            source_file,
        )
        if isinstance(lines, list):
            for idx, line in enumerate(lines):
                if not isinstance(line, dict):
                    continue
                for lf in schema.line_items.fields:
                    v = line.get(lf.name)
                    if v is None or v == "":
                        continue
                    vs = str(v).strip()
                    if not vs:
                        continue
                    if not _is_substring_of(vs, parsed.full_text):
                        log.debug(
                            "vlm_extractor: line_items[%d].%s=%r not in text — REJECTING",
                            idx, lf.name, vs,
                        )
                        continue
                    candidates.append(Candidate(
                        field=f"line_items[{idx}].{lf.name}",
                        value=vs,
                        page=0,
                        bbox=(0.0, 0.0, 0.0, 0.0),
                        evidence_text=vs,
                        model="qwen_vlm",
                        confidence=0.85,
                    ))

    log.info(
        "vlm_extractor: %d candidates accepted (substring-checked) for %s",
        len(candidates), source_file,
    )
    return candidates


@register_extractor("qwen_vlm")
class QwenVLMExtractor(Extractor):
    """L2 extractor: single Qwen2.5-VL call per document.

    Registered as 'qwen_vlm' in the extractor registry. The pipeline calls
    produce_candidates() which delegates to extract_with_vlm().
    """

    def produce_candidates(
        self,
        parsed: ParsedDocument,
        schema: DocSchema,
    ) -> list[Candidate]:
        source_file = parsed.source_path
        return extract_with_vlm(parsed, schema, source_file=source_file)
