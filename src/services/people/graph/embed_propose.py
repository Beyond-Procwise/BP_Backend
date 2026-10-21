"""The embedding proposer: "this person has handled suppliers like X, and has never touched Y, which looks like them".

Meaning only ever PROPOSES. Everything here ends in ``proposed.propose`` -> proc.bp_proposed_edge, which no graph query
reads until a person confirms it (then it becomes one ACTIVE_IN edge, source=inferred, confirmed_by). Nothing in this
module writes to the graph.

What is embedded: a short description of each SUPPLIER (its name and what it is bought for, from its own PO lines).
That is business data, not what anyone said: no message text, no activity rows, no person's content ever reaches the
model. What is compared: a person's asserted history (suppliers on their approve / reject / sign decision edges, decayed
by age) against suppliers they have never decided on.

Honesty about the score: ``confidence`` on a proposal is the cosine similarity to the person's history (0..1). It ranks
and filters; it is NOT a probability that the person is the right owner, and evidence says so (score_kind).
A pair is proposed at most once: any earlier proposal for it, whether pending, confirmed or REJECTED, suppresses it.
"""
from __future__ import annotations

import hashlib
import logging
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from services.people.admin import TENANT
from services.people.graph import derive, proposed
from services.people.graph.model import NodeRef, utc
from services.people.graph.store import GraphStore

logger = logging.getLogger(__name__)

MODEL_NAME = "BAAI/bge-large-en-v1.5"
MODEL_VERSION = "bge-large-en-v1.5@1/supplier-cosine-v1"      # embedding profile + the method that used it


def enabled() -> bool:
    return os.environ.get("PEOPLE_GRAPH_PROPOSER", "off").strip().lower() in ("on", "1", "true")


@dataclass(frozen=True)
class ProposerConfig:
    window_days: int = 180
    half_life_days: float = 42.0
    min_history: int = 3              # distinct suppliers a person has decided on before we infer anything about them
    min_similarity: float = 0.6       # sanity floor only; the real bar is the per-person chance level below
    false_alarms: float = 0.05        # expected chance proposals per person per run once the corpus is pure noise
    null_samples: int = 300
    top_k: int = 3                    # proposals per person per run
    max_pending: int = 5              # open proposals per person, so the queue can't bury a reviewer
    evidence_n: int = 3
    max_candidates: int = 600         # suppliers considered, busiest first (bounds the embedding work)


@dataclass(frozen=True)
class Candidate:
    user: str
    supplier_key: str
    similarity: float
    similar_to: Tuple[Tuple[str, float], ...]
    threshold: float = 0.0            # the bar this score had to clear (chance level for a profile of this size)


def _cos(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))          # vectors are unit length (embedder normalises)


def _unit(v: Sequence[float]) -> List[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


# ------------------------------------------------------------------ pure core
def _matrix(vectors: Dict[str, List[float]]):
    import numpy as np
    keys = sorted(vectors)
    return keys, np.asarray([vectors[k] for k in keys], dtype=float)


def chance_level(vectors: Dict[str, List[float]], size: int, user: str, cfg: ProposerConfig) -> float:
    """How well CHANCE can do: the best match any candidate gets against a random profile of `size` suppliers, taken at
    the (1 - false_alarms) quantile over many random profiles. A person scoring above it is a result chance reaches for
    only `false_alarms` (5%) of people who have nothing in common with any supplier.

    Why not a fixed cut-off, or "beat a random pair"? Sentence embeddings of short texts sit in a narrow band (everything
    is "fairly similar") that shifts with how many suppliers a profile averages; and a person is compared with EVERY
    candidate, so a per-pair 99% bar still lets ~1% of N through by luck. The best-of-all-candidates statistic corrects both,
    with no distributional assumption. Deterministic per person, so a rerun gives the same bar."""
    import numpy as np
    keys, m = _matrix(vectors)
    if len(keys) - size < 5:
        return 1.0                                       # no room to tell signal from chance: propose nothing
    rng = np.random.default_rng(int(hashlib.sha1(f"{user}:{size}:{len(keys)}".encode()).hexdigest()[:8], 16))
    best = []
    for _ in range(cfg.null_samples):
        pick = rng.choice(len(keys), size=size, replace=False)
        prof = m[pick].sum(axis=0)
        prof /= (np.linalg.norm(prof) or 1.0)
        sims = m @ prof
        sims[pick] = -1.0                                # a profile's own members are not candidates
        best.append(sims.max())
    return float(np.quantile(best, 1.0 - cfg.false_alarms))


def history_by_person(decisions: Iterable[Tuple[str, str, str, datetime]], now: datetime,
                      cfg: ProposerConfig) -> Dict[str, Dict[str, float]]:
    """(user, edge_type, 'supplier:ID', at) -> {user: {supplier_key: decayed weight}}. Supplier keys only."""
    out: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for user, _type, key, at in decisions:
        if key.startswith("supplier:"):
            out[user][key] += derive.decayed(utc(at), now, cfg.half_life_days)
    return {u: dict(h) for u, h in out.items()}


def rank_candidates(vectors: Dict[str, List[float]], history: Dict[str, Dict[str, float]],
                    cfg: ProposerConfig) -> List[Candidate]:
    """For each person with enough described history: suppliers they never decided on, closest to their history first."""
    out: List[Candidate] = []
    for user, hist in sorted(history.items()):
        known = {k: w for k, w in hist.items() if k in vectors}
        if len(known) < cfg.min_history:
            continue
        profile = _unit([sum(w * vectors[k][i] for k, w in known.items()) for i in range(len(next(iter(vectors.values()))))])
        bar = max(cfg.min_similarity, chance_level(vectors, len(known), user, cfg))
        scored = []
        for key, vec in vectors.items():
            if key in hist:
                continue
            sim = _cos(profile, vec)
            if sim > bar:
                scored.append((sim, key))
        scored.sort(key=lambda t: (-t[0], t[1]))
        for sim, key in scored[:cfg.top_k]:
            near = sorted(((_cos(vectors[h], vectors[key]), h) for h in known), reverse=True)[:cfg.evidence_n]
            out.append(Candidate(user, key, round(sim, 4), tuple((h, round(s, 4)) for s, h in near), round(bar, 4)))
    return out


# ------------------------------------------------------------------ data in
DESCRIBE_SQL = """
SELECT supplier_id, name, descs FROM (
  SELECT p.supplier_id, max(p.supplier_name) AS name, count(DISTINCT p.po_id) AS pos,
         (array_agg(l.item_description ORDER BY l.n DESC, l.item_description))[1:8] AS descs
    FROM proc.bp_purchase_order_trgt p
    JOIN (SELECT po_id, item_description, count(*) AS n FROM proc.bp_po_line_items_trgt
           WHERE item_description IS NOT NULL AND btrim(item_description) <> '' GROUP BY 1, 2) l ON l.po_id = p.po_id
   WHERE p.supplier_id IS NOT NULL {where}
   GROUP BY p.supplier_id) s
 {tail}
"""


_ITEM_CODE = re.compile(r"\s*\bITM\d+\b")       # catalogue codes carry no meaning for a text model


def describe_suppliers(conn, limit: int, also: Iterable[str] = ()) -> Dict[str, str]:
    """{'supplier:ID': 'Name. Bought for: a; b; c'} for the `limit` busiest suppliers plus every id in `also`."""
    with conn.cursor() as cur:
        cur.execute(DESCRIBE_SQL.format(where="", tail="ORDER BY pos DESC, supplier_id LIMIT %s"), (limit,))
        rows = cur.fetchall()
        extra = sorted(set(also) - {r[0] for r in rows})
        if extra:
            cur.execute(DESCRIBE_SQL.format(where="AND p.supplier_id = ANY(%s)", tail=""), (extra,))
            rows += cur.fetchall()
    out = {}
    for sid, name, descs in rows:
        things = "; ".join(dict.fromkeys(c for c in (_ITEM_CODE.sub("", d).strip()[:80] for d in (descs or []) if d) if c))
        if things:
            out[f"supplier:{sid}"] = f"{(name or sid).strip()}. Bought for: {things}"[:600]
    return out


_CACHE: Dict[str, List[float]] = {}                  # sha1(text) -> unit vector; lives as long as the process
_CACHE_MAX = 5000


def embed_cached(embedder, texts: Dict[str, str]) -> Dict[str, List[float]]:
    """{key: unit vector}. A supplier whose description is unchanged is not embedded again (the CPU pass is minutes)."""
    h = {k: hashlib.sha1(t.encode()).hexdigest() for k, t in texts.items()}
    todo = sorted({h[k] for k in texts} - set(_CACHE))
    by_hash = {h[k]: texts[k] for k in texts}
    for i in range(0, len(todo), 64):
        batch = todo[i:i + 64]
        for hv, v in zip(batch, embedder.encode([by_hash[x] for x in batch])):
            _CACHE[hv] = _unit(v)
    if len(_CACHE) > _CACHE_MAX:                      # drop what this run did not use
        for hv in [x for x in _CACHE if x not in set(h.values())][: len(_CACHE) - _CACHE_MAX]:
            del _CACHE[hv]
    return {k: _CACHE[h[k]] for k in texts}


def _existing(conn) -> Tuple[Dict[str, int], Set[Tuple[str, str]]]:
    """(pending proposals per person, every (user, supplier_key) ever proposed in any state)."""
    pending: Dict[str, int] = defaultdict(int)
    seen: Set[Tuple[str, str]] = set()
    with conn.cursor() as cur:
        cur.execute("SELECT src_key, dst_key, status FROM proc.bp_proposed_edge WHERE tenant_id=%s AND edge_type='ACTIVE_IN' "
                    "AND dst_label='Entity'", (TENANT,))
        for src, dst, status in cur.fetchall():
            seen.add((src, dst))
            if status == "proposed":
                pending[src] += 1
    return pending, seen


def _active_people(conn) -> Set[str]:
    with conn.cursor() as cur:
        cur.execute("SELECT subject FROM proc.bp_person WHERE tenant_id=%s AND is_active", (TENANT,))
        return {r[0] for r in cur.fetchall()}


def run(conn, store: GraphStore, embedder, *, now: Optional[datetime] = None, cfg: ProposerConfig = ProposerConfig(),
        describe: Callable[..., Dict[str, str]] = describe_suppliers) -> Dict[str, int]:
    now = now or datetime.now(timezone.utc)
    decisions = derive.load_decisions(store, TENANT, now - timedelta(days=cfg.window_days))
    people = _active_people(conn)
    history = {u: h for u, h in history_by_person(decisions, now, cfg).items() if u in people and len(h) >= cfg.min_history}
    stats = {"people": len(history), "described": 0, "candidates": 0, "proposed": 0, "suppressed": 0}
    if not history:
        return stats                                             # nothing to infer from: don't even load the model
    needed = {k[len("supplier:"):] for h in history.values() for k in h}
    texts = describe(conn, cfg.max_candidates, also=sorted(needed))
    stats["described"] = len(texts)
    keys = sorted(texts)
    vectors = embed_cached(embedder, texts)
    pending, seen = _existing(conn)
    already = {(e.src.key, e.dst.key) for e in store.edges(TENANT, type="ACTIVE_IN", as_of=now)}
    cands = rank_candidates(vectors, history, cfg)
    stats["candidates"] = len(cands)
    for c in cands:
        if (c.user, c.supplier_key) in seen or (c.user, c.supplier_key) in already or pending[c.user] >= cfg.max_pending:
            stats["suppressed"] += 1
            continue
        pid = proposed.propose(
            conn, NodeRef("User", c.user), "ACTIVE_IN", NodeRef("Entity", c.supplier_key), confidence=c.similarity,
            model_version=MODEL_VERSION,
            evidence={"kind": "supplier_similarity", "score_kind": "cosine_similarity", "model": MODEL_NAME,
                      "chance_level": c.threshold,
                      "similar_to": [{"key": h, "similarity": s} for h, s in c.similar_to],
                      "history_suppliers": len(history[c.user])},
            props={"evidence_count": len(history[c.user])})
        if pid:
            pending[c.user] += 1
            stats["proposed"] += 1
    return stats


def default_embedder():
    from services.vector_store.embedder import SentenceTransformerEmbedder
    # CPU by default: this runs inside the daily job on a host whose GPU the models share.
    return SentenceTransformerEmbedder(MODEL_NAME, device=os.environ.get("PEOPLE_GRAPH_PROPOSER_DEVICE", "cpu"))
