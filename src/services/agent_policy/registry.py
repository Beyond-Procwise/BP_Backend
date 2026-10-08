"""What the orchestrator can see, at which checkpoint. The single source for every
"does the orchestrator recognise this name" and "is this input available here" question."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


@dataclass(frozen=True)
class RegistrySnapshot:
    checkpoints: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    actions: Dict[str, Set[str]] = field(default_factory=dict)
    inputs: Dict[str, Dict[str, Dict[str, Any]]] = field(default_factory=dict)

    def checkpoint_live(self, cp: Optional[str]) -> bool:
        return bool(cp) and (self.checkpoints.get(cp) or {}).get("status") == "live"

    def knows_checkpoint(self, cp: Optional[str]) -> bool:
        return bool(cp) and cp in self.checkpoints

    def knows_action(self, cp: Optional[str], name: str) -> bool:
        return name in self.actions.get(cp or "", set())

    def input_row(self, cp: Optional[str], fld: str) -> Optional[Dict[str, Any]]:
        return self.inputs.get(cp or "", {}).get(fld)

    def available(self, cp: Optional[str], fld: str) -> bool:
        row = self.input_row(cp, fld)
        return bool(row) and row.get("status") == "live" and self.checkpoint_live(cp)

    def plain(self, cp: Optional[str]) -> str:
        return (self.checkpoints.get(cp or "") or {}).get("plain") or (cp or "")

    def as_dict(self) -> Dict[str, Any]:
        return {"checkpoints": self.checkpoints,
                "actions": {k: sorted(v) for k, v in self.actions.items()},
                "inputs": self.inputs}


def snapshot_from_rows(rows: List[Dict[str, Any]]) -> RegistrySnapshot:
    cps: Dict[str, Dict[str, Any]] = {}
    acts: Dict[str, Set[str]] = {}
    ins: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for r in rows:
        if r["kind"] == "checkpoint":
            cps[r["name"]] = {"plain": r["plain"], "status": r.get("status", "live")}
        elif r["kind"] == "action":
            acts.setdefault(r["checkpoint"], set()).add(r["name"])
        elif r["kind"] == "input":
            ins.setdefault(r["checkpoint"], {})[r["name"]] = {
                "plain": r["plain"], "type": r.get("value_type"),
                "source": r.get("source"), "status": r.get("status", "live")}
    return RegistrySnapshot(cps, acts, ins)


def load_registry(conn: Any = None) -> RegistrySnapshot:
    from services.db import get_conn

    if conn is None:
        with get_conn() as own:
            return load_registry(own)
    cur = conn.cursor()
    cur.execute("SELECT kind, name, checkpoint, plain, value_type, source, status "
                "FROM proc.bp_orchestrator_registry")
    cols = ("kind", "name", "checkpoint", "plain", "value_type", "source", "status")
    return snapshot_from_rows([dict(zip(cols, row)) for row in cur.fetchall()])
