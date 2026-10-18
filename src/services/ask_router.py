"""A small router for /ask, ahead of retrieval: greeting, calculation, inventory - everything else is RAG.

Rules decide first (they are exact and explainable); the embedding model already loaded for retrieval is a
second opinion for paraphrases the rules miss. Below the threshold the answer is "rag", the default, so a
router mistake costs a normal search and never a wrong canned answer. Every decision carries its score and
the threshold it was held to, for the trace and the log (test D5 asks for exactly that).

Measured with bge-large (tests/services/test_ask_router.py pins the examples): paraphrases and unrelated
procurement questions overlap in similarity, so the embedding layer is held at a high threshold and the rules
do most of the work, as in ask_input_guard.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

GREETING = "greeting"
CALCULATION = "calculation"
INVENTORY = "inventory"
RAG = "rag"

EMBED_THRESHOLD = 0.86


@dataclass(frozen=True)
class RouteDecision:
    route: str
    layer: str = "default"            # "rule" | "embedding" | "default"
    score: Optional[float] = None     # best embedding similarity seen (None when the embedder was not consulted)
    threshold: Optional[float] = None
    reason: str = ""
    best_route: Optional[str] = None  # the closest route even when below the threshold (D5: near-threshold paraphrase)

    def log_fields(self) -> Dict[str, Any]:
        return {"route": self.route, "layer": self.layer, "score": None if self.score is None else round(self.score, 4),
                "threshold": self.threshold, "best_route": self.best_route, "reason": self.reason}


_GREETING = re.compile(
    r"^\s*(?:(?:hi|hello|hey|hiya|howdy|good\s+(?:morning|afternoon|evening)|greetings)(?:\s+(?:there|team|all|everyone|joshi))?"
    r"|thanks|thank\s+you|cheers|ok(?:ay)?|great|cool)"
    r"(?:\s*[,.!-]*\s*(?:how\s+are\s+you(?:\s+today)?|how(?:'|’)?s\s+it\s+going|hope\s+you(?:'|’)?re\s+well))?\s*[.!?]*\s*$", re.I)
GREETING_REPLY = "Hello! I can answer questions about your suppliers, purchase orders, contracts, spend and procurement policies. What would you like to know?"

_AMOUNT = re.compile(r"(?:[£$€]\s?\d[\d,]*(?:\.\d+)?|\b\d[\d,]*(?:\.\d+)?\s?(?:k|m|%|percent|gbp|usd|eur|units?|pcs|days?)\b|\b\d{2,}\b)", re.I)
_CALC_VERB = re.compile(
    r"\b(?:calculate|compute|work\s+out|what(?:'|’)?s\s+the\s+(?:total|sum|cost|landed\s+cost|price)\s+(?:of|for|if)|what\s+is\s+the\s+(?:total|sum|landed)\s+cost"
    r"|landed\s+cost|how\s+much\s+(?:would|will|does|do)\b[^?]{0,60}\b(?:cost|come\s+to|total)|add\s+up|multiply|total\s+cost\s+(?:of|for))\b", re.I)

_INVENTORY = re.compile(
    r"\b(?:inventory|stock\s+(?:level|levels|on\s+hand|count|available|balance|position)|in\s+stock|units?\s+on\s+hand|on[-\s]hand\s+(?:quantity|stock)"
    r"|how\s+(?:many|much)\b[^?]{0,40}\b(?:in\s+stock|on\s+hand|in\s+the\s+warehouse|do\s+we\s+(?:have|hold))|reorder\s+(?:level|point)|safety\s+stock|stockout|warehouse\s+stock)\b", re.I)
INVENTORY_CLARIFY = (
    "I don't hold stock levels in the procurement records, only what was ordered and invoiced. "
    "Do you want the quantity ordered for that item on your purchase orders, or the stock held in your warehouse or ERP system?"
)

EXAMPLES: Dict[str, List[str]] = {
    GREETING: ["Hello", "Hi there", "Good morning", "Hey, how are you today?", "Thanks, that was helpful"],
    CALCULATION: [
        "Calculate the landed cost of 500 units at 12.40 each with 8% freight and 5% duty.",
        "What is the total cost of 200 items at 45 pounds plus 10% shipping?",
        "Work out the total if the unit price is 30 and we order 1,200.",
    ],
    INVENTORY: [
        "How many units of this part do we have in stock?",
        "What is the current stock level for this item?",
        "Do we have inventory on hand for this product?",
    ],
}

_cache: Dict[int, Any] = {}


def _examples(embedder: Any):
    import numpy as np
    key = id(embedder)
    if key not in _cache:
        labels, texts = [], []
        for route, phrases in EXAMPLES.items():
            labels += [route] * len(phrases)
            texts += phrases
        _cache[key] = (labels, np.asarray(embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False), dtype="float32"))
    return _cache[key]


def route_by_rules(query: str) -> Optional[RouteDecision]:
    text = query if isinstance(query, str) else ""
    if not text.strip():
        return None
    if _GREETING.match(text):
        return RouteDecision(GREETING, "rule", reason="greeting-only")
    if _INVENTORY.search(text):
        return RouteDecision(INVENTORY, "rule", reason="inventory-term")
    if _CALC_VERB.search(text) and len(_AMOUNT.findall(text)) >= 2:
        return RouteDecision(CALCULATION, "rule", reason="calculation-verb-with-amounts")
    return None


def decide(query: str, *, embedder: Any = None, use_embedding: bool = False, threshold: float = EMBED_THRESHOLD) -> RouteDecision:
    """The route for ``query``. Never raises; any failure is the default route."""
    try:
        hit = route_by_rules(query)
        if hit is not None:
            return hit
        if not use_embedding or embedder is None or not isinstance(query, str) or not query.strip():
            return RouteDecision(RAG, "default", reason="no rule matched")
        import numpy as np
        labels, matrix = _examples(embedder)
        vec = np.asarray(embedder.encode(query, normalize_embeddings=True, show_progress_bar=False), dtype="float32").reshape(-1)
        sims = matrix @ vec
        best = int(sims.argmax())
        score, closest = float(sims[best]), labels[best]
        if score >= threshold and closest != GREETING:      # a greeting is only ever a rule: a paraphrase must not skip retrieval
            return RouteDecision(closest, "embedding", score, threshold, "embedding above threshold", closest)
        return RouteDecision(RAG, "default", score, threshold, "below threshold", closest)
    except Exception:
        logger.exception("ask router failed; using the default route")
        return RouteDecision(RAG, "default", reason="router error")


# --- inventory data -------------------------------------------------------------------------------------------------------------

def inventory_data_exists(conn_factory: Any) -> bool:
    """Is there any inventory/stock table in the procurement schema? (There is none today.) Errors count as no."""
    try:
        with conn_factory() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM information_schema.tables WHERE table_schema = 'proc' "
                            "AND (table_name ILIKE %s OR table_name ILIKE %s) LIMIT 1", ("%inventory%", "%stock_level%"))
                return cur.fetchone() is not None
    except Exception:
        logger.debug("inventory table check failed", exc_info=True)
        return False
