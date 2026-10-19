"""Splitting text into embeddable chunks. bge-large-en-v1.5 truncates input at 512 tokens and English runs about 1.3 tokens a word,
so 350 words (~455 tokens) are embedded whole. Overlapping word windows over whitespace-collapsed text.

Identical to ``extraction.embed.chunk_text`` (a test pins that) so documents re-embedded here chunk exactly as they were before.
"""
from __future__ import annotations

import re
from typing import List, Optional

CHUNK_WORDS = 350
CHUNK_OVERLAP = 50
SUMMARY_CHARS = 240


def chunk_text(text: Optional[str]) -> List[str]:
    words = [w for w in re.sub(r"\s+", " ", text or "").strip().split(" ") if w]
    if not words:
        return []
    step = CHUNK_WORDS - CHUNK_OVERLAP
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + CHUNK_WORDS]))
        if start + CHUNK_WORDS >= len(words):
            break
    return chunks
