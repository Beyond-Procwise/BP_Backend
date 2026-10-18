"""Feature flags for the /ask pipeline stages.

Each stage has its own flag: ``ASK_<NAME>=1|0`` decides it outright. With no
variable set, a stage is ON in development and OFF when ``ENVIRONMENT`` is
``prod``/``production`` — so a new stage reaches production only when somebody
switches it on there. The exception is ``OPT_IN``: flags that change which documents
are retrieved stay off until set.
"""
from __future__ import annotations

import os

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}

INPUT_GUARD = "ASK_INPUT_GUARD"            # deterministic input check before retrieval
INPUT_GUARD_EMBED = "ASK_INPUT_GUARD_EMBED"  # embedding-similarity second pass of that check
OUTPUT_CHECK = "ASK_OUTPUT_APPROVAL_CHECK"   # narrow "I approve / I waive" output check
SAFETY_PROMPT = "ASK_SAFETY_PROMPT"        # boundaries appended to the system prompt
FAST_RERANK = "ASK_FAST_RERANK"            # smaller reranker, fewer and shorter candidates (read once, at start-up)
TRACE = "ASK_TRACE"                        # debug trace, shown to admins only


def is_production() -> bool:
    return os.environ.get("ENVIRONMENT", "").strip().lower() in ("prod", "production")


# Flags that change WHICH documents come back are opt-in everywhere until their effect on answers is approved.
OPT_IN = frozenset({FAST_RERANK})


def enabled(name: str) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if raw in _TRUE:
        return True
    if raw in _FALSE:
        return False
    return name not in OPT_IN and not is_production()
