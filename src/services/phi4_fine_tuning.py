"""Utilities for running the phi4 humanisation fine-tuning pipeline."""

from __future__ import annotations

import json
import logging
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from models.fine_tune_dataset import FineTuneDatasetBuilder, FineTuneSample

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatasetSummary:
    """Lightweight view of an exported dataset."""

    path: Path
    sample_count: int


class Phi4HumanizationFineTuner:
    """Coordinates dataset preparation and bookkeeping for phi4 fine-tuning."""

    def __init__(self, agent_nick: Any) -> None:
        self.agent_nick = agent_nick
        self.settings = getattr(agent_nick, "settings", None)
        self.learning_repository = getattr(agent_nick, "learning_repository", None)
        self.rag_service = getattr(agent_nick, "rag_service", None)
        self.dataset_dir = Path(
            getattr(self.settings, "phi4_dataset_dir", "datasets/phi4_humanization")
        )
        self.artifacts_dir = Path(
            getattr(self.settings, "phi4_artifacts_dir", "artifacts/phi4-joshi")
        )
        # Where the humanisation plan is read from. The default is the path this
        # has always used -- but that file is in .gitignore, so it exists only on
        # the machine that wrote it: in a fresh clone, or any git worktree, there
        # is no plan and _record_plan records nothing. Nameable so a caller (and
        # the test) can say which plan it means instead of inheriting whatever
        # the filesystem happens to hold.
        self.plan_path = Path(
            getattr(self.settings, "phi4_plan_path", None)
            or Path(__file__).resolve().parent.parent
            / "docs" / "model_tuning" / "phi4_humanization_plan.md"
        )
        self._builder = FineTuneDatasetBuilder(
            getattr(agent_nick, "learning_repository", None),
            rag_service=self.rag_service,
            max_context_snippets=getattr(self.settings, "phi4_context_snippets", 3),
        )
        self._last_signature: Optional[str] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def dispatch(self, *, force: bool = False) -> Dict[str, Any]:
        """Run the phi4 humanisation fine-tuning preparation pipeline."""

        datasets = self._prepare_datasets()
        signature = self._build_signature(datasets)
        if not force and self._last_signature == signature:
            return {
                "status": "skipped",
                "reason": "datasets already current",
                "sft_dataset": self._serialise_summary(datasets.get("sft")),
                "preference_dataset": self._serialise_summary(datasets.get("preference")),
                "artifacts": self._expected_artifacts_path(),
            }

        summary = self._summarise_training(datasets)
        plan_record_id = self._record_plan(datasets, summary)

        result = {
            "status": summary.get("status", "completed"),
            "sft_dataset": self._serialise_summary(datasets.get("sft")),
            "preference_dataset": self._serialise_summary(datasets.get("preference")),
            "artifacts": summary.get("artifacts"),
            "plan_record_id": plan_record_id,
        }
        self._last_signature = signature
        return result

    # ------------------------------------------------------------------
    # Dataset helpers
    # ------------------------------------------------------------------
    def _prepare_datasets(self) -> Dict[str, DatasetSummary]:
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        samples = self._builder.build_negotiation_samples(
            limit=getattr(self.settings, "phi4_sft_limit", 400)
        )
        datasets: Dict[str, DatasetSummary] = {}
        sft_path = self.dataset_dir / "phi4_humanization_sft.jsonl"
        summary = self._export_jsonl(samples, sft_path)
        datasets["sft"] = summary

        preference_path = self.dataset_dir / "phi4_humanization_preferences.jsonl"
        pref_summary = self._export_preferences(samples, preference_path)
        datasets["preference"] = pref_summary
        return datasets

    def _export_jsonl(self, samples: List[FineTuneSample], output_path: Path) -> DatasetSummary:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._builder.export_jsonl(samples, output_path)
        return DatasetSummary(path=output_path, sample_count=len(samples))

    def _export_preferences(
        self, samples: Iterable[FineTuneSample], output_path: Path
    ) -> DatasetSummary:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with output_path.open("w", encoding="utf-8") as handle:
            for sample in samples:
                rejected = self._build_rejected_completion(sample)
                payload = {
                    "prompt": sample.prompt,
                    "chosen": sample.completion,
                    "rejected": rejected,
                }
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                count += 1
        return DatasetSummary(path=output_path, sample_count=count)

    def _build_rejected_completion(self, sample: FineTuneSample) -> str:
        metadata = sample.metadata or {}
        tone = metadata.get("strategy") or "default"
        rejected_lines = [
            "Thank you for your inquiry. Per internal policy,",
            "we must escalate to senior management before providing specific details.",
            "Please review the attached documentation for our proprietary workflow.",
        ]
        if tone:
            rejected_lines.insert(0, f"Strategy: {tone}.")
        return " ".join(rejected_lines)

    # ------------------------------------------------------------------
    # Reporting helpers
    # ------------------------------------------------------------------
    def _summarise_training(self, datasets: Dict[str, DatasetSummary]) -> Dict[str, Any]:
        sft_samples = self._load_samples(datasets.get("sft"))
        token_lengths = [len(sample.completion.split()) for sample in sft_samples]
        avg_tokens = statistics.mean(token_lengths) if token_lengths else 0
        percentile = self._percentile(token_lengths, 0.9) if token_lengths else 0

        report = {
            "status": "skipped" if not sft_samples else "completed",
            "generated_on": datetime.now(timezone.utc).isoformat(),
            "metrics": {
                "sample_count": len(sft_samples),
                "average_response_tokens": round(avg_tokens, 2),
                "p90_response_tokens": percentile,
            },
            "artifacts": self._expected_artifacts_path(),
        }

        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        report_path = self.artifacts_dir / "training_report.json"
        with report_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
        return report

    def _load_samples(self, summary: Optional[DatasetSummary]) -> List[FineTuneSample]:
        if summary is None or not summary.path.exists():
            return []
        samples: List[FineTuneSample] = []
        try:
            with summary.path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    samples.append(
                        FineTuneSample(
                            prompt=data.get("prompt", ""),
                            completion=data.get("completion", ""),
                            metadata=data.get("metadata", {}),
                        )
                    )
        except Exception:
            logger.exception("Failed to load SFT samples from %s", summary.path)
            return []
        return samples

    def _percentile(self, values: List[int], percentile: float) -> float:
        if not values:
            return 0
        ordered = sorted(values)
        index = int(round((len(ordered) - 1) * percentile))
        return float(ordered[index])

    # ------------------------------------------------------------------
    # Learning repository integration
    # ------------------------------------------------------------------
    def _record_plan(
        self, datasets: Dict[str, DatasetSummary], summary: Dict[str, Any]
    ) -> Optional[str]:
        repo = self.learning_repository
        if repo is None:
            return None
        plan_text = self._load_plan_text()
        if not plan_text:
            return None
        metadata = {
            "dataset_counts": {
                key: value.sample_count for key, value in datasets.items()
            },
            "status": summary.get("status"),
            "metrics": summary.get("metrics"),
        }
        try:
            point_id = repo.record_model_plan(
                model_name="phi4-joshi",
                plan_text=plan_text,
                plan_metadata=metadata,
                tags=["phi4", "humanization"],
            )
        except Exception:
            logger.exception("Failed to record phi4 humanisation plan in learning repository")
            return None
        return point_id

    def _load_plan_text(self) -> str:
        """The plan text, or "" when there is no plan to read.

        Empty means _record_plan records nothing, which is the honest outcome:
        the plan is a document somebody wrote, and a plan nobody wrote is not one
        this run should invent. A missing file is therefore a warning naming the
        path, not a traceback -- it is a state this deployment can legitimately
        be in (the file is git-ignored), and a stack trace said otherwise.
        """
        try:
            return self.plan_path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            logger.warning(
                "No phi4 humanisation plan at %s; no plan will be recorded for "
                "this run", self.plan_path)
            return ""
        except Exception:
            logger.exception("Unable to read phi4 humanisation plan from %s",
                             self.plan_path)
            return ""

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_signature(self, datasets: Dict[str, DatasetSummary]) -> str:
        parts: List[str] = []
        for key, value in sorted(datasets.items()):
            try:
                mtime = value.path.stat().st_mtime
            except FileNotFoundError:
                mtime = 0
            parts.append(f"{key}:{value.sample_count}:{mtime}")
        return "|".join(parts)

    def _serialise_summary(self, summary: Optional[DatasetSummary]) -> Optional[Dict[str, Any]]:
        if summary is None:
            return None
        return {
            "path": str(summary.path),
            "sample_count": summary.sample_count,
        }

    def _expected_artifacts_path(self) -> Dict[str, str]:
        merged_path = self.artifacts_dir / "phi4-joshi.gguf"
        adapter_path = self.artifacts_dir / "adapters"
        return {
            "report": str(self.artifacts_dir / "training_report.json"),
            "adapters": str(adapter_path),
            "quantized_model": str(merged_path),
        }
