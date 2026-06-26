"""Verify the model-routing consolidation: every NON-extraction agent must
resolve to the single reasoning brain (:unified); only document extraction
stays on the extraction specialist. Replicates AgentNick._build_agent_model_
registry / get_agent_model resolution against the REAL settings — no heavy boot.
"""
import sys
sys.path.insert(0, "src")

from config.settings import settings
from agents.base_agent import _AGENT_MODEL_FIELD_PREFERENCES, _slugify_agent_name


def resolve(slug: str) -> str:
    # mirror _build_agent_model_registry: walk preference fields, else terminal fallback
    fields = _AGENT_MODEL_FIELD_PREFERENCES.get(slug, ())
    for f in fields:
        v = getattr(settings, f, None)
        if isinstance(v, str) and v.strip():
            return v.strip()
    # terminal fallback (reasoning_model, then extraction_model)
    fb = getattr(settings, "reasoning_model", None) or getattr(settings, "extraction_model", None)
    return (fb or "").strip()


print(f"reasoning_model  = {settings.reasoning_model}")
print(f"extraction_model = {settings.extraction_model}")
print(f"rag_model        = {settings.rag_model}")
print(f"local_primary    = {settings.local_primary_model}")
print("-" * 60)

EXTRACTION = {"data_extraction_agent"}
rows = sorted(set(_AGENT_MODEL_FIELD_PREFERENCES) | {"some_unlisted_agent", "deal_summary"})
bad = []
for slug in rows:
    m = resolve(slug)
    role = "EXTRACTION" if slug in EXTRACTION else "non-extraction"
    ok = (":extract" in m or ":latest" in m) if slug in EXTRACTION else (":unified" in m)
    flag = "OK " if ok else "!! "
    if not ok:
        bad.append((slug, m))
    print(f"{flag}{slug:30s} [{role:14s}] -> {m}")

print("-" * 60)
print("RESULT:", "ALL CORRECT" if not bad else f"MISROUTED: {bad}")
sys.exit(0 if not bad else 1)
