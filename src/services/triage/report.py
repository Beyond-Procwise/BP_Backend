"""The run report: the triage spec's §15 measures for one run (spec §1, criterion 4)."""
from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Optional

from .model import Outcome, Severity, money

BANK_GAP = "Payee bank details are not checked: invoices carry no bank data."


@dataclass
class RunReport:
    mode: str
    dry_run: bool = False
    deals_requested: int = 0
    run_id: Optional[str] = None
    deals_done: int = 0
    deals_without_documents: int = 0
    deals_vanished: int = 0               # had a state row, now no documents: findings closed
    failed: dict = field(default_factory=dict)
    raw_differences: int = 0
    shown: int = 0
    outcomes: Counter = field(default_factory=Counter)
    verdicts: Counter = field(default_factory=Counter)
    findings_by_rule: Counter = field(default_factory=Counter)
    findings_by_severity: Counter = field(default_factory=Counter)
    exposure_by_severity: dict = field(default_factory=dict)
    write_counts: Counter = field(default_factory=Counter)
    known_gaps: list = field(default_factory=lambda: [BANK_GAP])
    elapsed_s: float = 0.0
    _t0: float = field(default_factory=time.monotonic, repr=False)

    def fail(self, deal_id: str, why: str) -> None:
        self.failed[deal_id] = why

    def add(self, out) -> None:
        self.deals_done += 1
        for r in out.results:
            self.outcomes[r.outcome.value] += 1
            if r.outcome != Outcome.MATCH:
                self.raw_differences += 1
        for f in out.findings:
            sev = f.severity.name
            self.findings_by_rule[f.rule_id] += 1
            self.findings_by_severity[sev] += 1
            self.exposure_by_severity[sev] = (self.exposure_by_severity.get(sev, Decimal("0"))
                                              + (f.exposure_gbp or Decimal("0")))
            if f.severity >= Severity.S2:
                self.shown += 1
        self.verdicts[out.verdict.verdict] += 1

    def finish(self) -> None:
        self.elapsed_s = round(time.monotonic() - self._t0, 2)

    @property
    def noise_ratio(self) -> float:
        return self.shown / self.raw_differences if self.raw_differences else 0.0

    @property
    def deals_per_second(self) -> float:
        return self.deals_done / self.elapsed_s if self.elapsed_s else 0.0

    def to_dict(self) -> dict:
        return {
            "mode": self.mode, "dry_run": self.dry_run, "run_id": self.run_id,
            "deals_requested": self.deals_requested, "deals_done": self.deals_done,
            "deals_without_documents": self.deals_without_documents,
            "deals_vanished": self.deals_vanished,
            "failed": dict(self.failed), "elapsed_s": self.elapsed_s,
            "deals_per_second": round(self.deals_per_second, 2),
            "raw_differences": self.raw_differences, "shown": self.shown,
            "noise_ratio": round(self.noise_ratio, 4),
            "outcomes": dict(self.outcomes), "verdicts": dict(self.verdicts),
            "findings_by_rule": dict(self.findings_by_rule),
            "findings_by_severity": dict(self.findings_by_severity),
            "exposure_by_severity": {k: str(v) for k, v in self.exposure_by_severity.items()},
            "write_counts": dict(self.write_counts), "known_gaps": list(self.known_gaps),
        }

    def render(self) -> str:
        def counts(c):
            return " · ".join(f"{k} {v:,}" for k, v in c.most_common()) or "none"

        lines = [
            f"Discrepancy triage — {self.mode}{' (dry run)' if self.dry_run else ''} "
            f"· run {self.run_id or '-'}",
            f"Deals: {self.deals_done:,} checked of {self.deals_requested:,} · "
            f"{self.deals_without_documents:,} without documents · "
            f"{self.deals_vanished:,} no longer have documents · {len(self.failed):,} failed",
            f"Time: {self.elapsed_s:,.1f}s · {self.deals_per_second:,.1f} deals/s",
            f"Noise ratio: {self.shown:,} findings shown / {self.raw_differences:,} raw "
            f"differences = {self.noise_ratio:.2%}",
            f"Verdicts: {counts(self.verdicts)}",
            "Exposure by severity: " + (" · ".join(
                f"{s} {money(self.exposure_by_severity.get(s), 'GBP')}"
                for s in ("S1", "S2", "S3") if s in self.exposure_by_severity) or "none"),
            f"Findings by rule: {counts(self.findings_by_rule)}",
            f"Outcomes: {counts(self.outcomes)}",
            f"Writes: {counts(self.write_counts)}",
            "Known gaps:",
            *[f"  - {g}" for g in self.known_gaps],
        ]
        if self.failed:
            lines.append("Failed deals (first 20):")
            lines += [f"  - {d}: {why}" for d, why in list(self.failed.items())[:20]]
        return "\n".join(lines)
