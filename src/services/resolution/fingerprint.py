"""A reproducibility fingerprint for a request.

Every output carries its inputs: the exact edge set, constraint set and profile
registry version that produced it, hashed so a stored result can be checked
against a re-run.
"""
from __future__ import annotations

import hashlib
import json

from .contracts import ResolutionRequest


def canonical_inputs(request: ResolutionRequest) -> dict:
    return {
        "profile_registry_version": request.profile_registry_version,
        "edges": sorted(
            [
                {
                    "source_id": e.source_id,
                    "target_id": e.target_id,
                    "profile_id": e.profile_id,
                    "log_odds": repr(float(e.log_odds)),
                    "consumes": {k: repr(float(v)) for k, v in sorted(e.consumes.items())},
                }
                for e in request.edges
            ],
            key=lambda d: (d["source_id"], d["target_id"], d["profile_id"]),
        ),
        "capacities": sorted(
            [
                {
                    "resource_id": c.resource_id,
                    "capacity": repr(float(c.capacity)),
                    "tolerance": repr(float(c.tolerance)),
                }
                for c in request.capacities
            ],
            key=lambda d: d["resource_id"],
        ),
        "rules": sorted(
            [
                {
                    "profile_id": r.profile_id,
                    "shape": r.shape,
                    "max_sources_per_target": r.max_sources_per_target,
                    "max_targets_per_source": r.max_targets_per_source,
                }
                for r in request.rules
            ],
            key=lambda d: d["profile_id"],
        ),
    }


def inputs_hash(request: ResolutionRequest) -> str:
    blob = json.dumps(canonical_inputs(request), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
