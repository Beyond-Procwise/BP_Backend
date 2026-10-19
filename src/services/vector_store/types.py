"""What the store returns, shaped so code written against Qdrant's results (``.id``, ``.score``, ``.payload``) keeps working."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class VectorHit:
    id: str
    source_id: str
    chunk_idx: int
    collection: str
    score: Optional[float]
    content: Optional[str]
    payload: Dict[str, Any] = field(default_factory=dict)


class TenantRequired(ValueError):
    """Every vector operation names its tenant. There is no tenant-less call."""
