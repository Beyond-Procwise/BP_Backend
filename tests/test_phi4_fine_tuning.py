import json
from types import SimpleNamespace

from services.phi4_fine_tuning import Phi4HumanizationFineTuner


class DummyLearningRepository:
    def __init__(self, records):
        self._records = list(records)
        self.plan_calls = []

    def get_recent_learnings(self, **kwargs):
        return list(self._records)

    def record_model_plan(self, **kwargs):
        self.plan_calls.append(kwargs)
        return f"plan-{len(self.plan_calls)}"


class DummyRagService:
    def search(self, query, top_k=3, filters=None):
        return [
            SimpleNamespace(
                payload={"summary": "Supplier sentiment positive"},
                score=0.88,
            )
        ]


def _build_learning_record(strategy: str, counter_price: str) -> dict:
    return {
        "event_type": "negotiation_round",
        "summary": f"Strategy {strategy} with counter {counter_price}",
        "metadata": {
            "strategy": strategy,
            "counter_price": counter_price,
            "target_price": "890",
            "asks": ["shorter lead time"],
            "lead_time_request": "7 days",
            "awaiting_response": True,
            "supplier_reply_registered": False,
        },
    }


PLAN_TEXT = "Phi4 humanization plan, written by this test."


def _write_plan(tmp_path) -> str:
    """The plan this test means, not whichever one the filesystem holds.

    The product's default plan path is in .gitignore, so it exists only on the
    machine that wrote it: this test passed in the main checkout and failed in
    every git worktree, for reasons that had nothing to do with fine-tuning.
    """
    plan = tmp_path / "plan" / "phi4_humanization_plan.md"
    plan.parent.mkdir(parents=True, exist_ok=True)
    plan.write_text(PLAN_TEXT, encoding="utf-8")
    return str(plan)


def test_phi4_fine_tuner_exports_datasets_and_records_plan(tmp_path):
    records = [
        _build_learning_record("counter", "910"),
        _build_learning_record("hold", "905"),
    ]
    repo = DummyLearningRepository(records)
    rag_service = DummyRagService()
    settings = SimpleNamespace(
        phi4_dataset_dir=tmp_path / "datasets",
        phi4_artifacts_dir=tmp_path / "artifacts",
        phi4_sft_limit=10,
        phi4_context_snippets=2,
        phi4_plan_path=_write_plan(tmp_path),
    )
    agent = SimpleNamespace(
        settings=settings,
        learning_repository=repo,
        rag_service=rag_service,
    )

    tuner = Phi4HumanizationFineTuner(agent)
    result = tuner.dispatch(force=True)

    assert result["status"] == "completed"
    sft_info = result["sft_dataset"]
    pref_info = result["preference_dataset"]
    assert sft_info["sample_count"] == 2
    assert pref_info["sample_count"] == 2

    sft_path = tmp_path / "datasets" / "phi4_humanization_sft.jsonl"
    assert sft_path.exists()
    with sft_path.open("r", encoding="utf-8") as handle:
        lines = [json.loads(line) for line in handle if line.strip()]
    assert len(lines) == 2

    report_path = tmp_path / "artifacts" / "training_report.json"
    assert report_path.exists()
    with report_path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    assert report["metrics"]["sample_count"] == 2

    assert repo.plan_calls, "expected model plan to be recorded"
    assert repo.plan_calls[0]["plan_text"] == PLAN_TEXT, (
        "the recorded plan is not the one this test wrote — it came from the "
        "filesystem")
    metadata = repo.plan_calls[0]["plan_metadata"]
    assert metadata["dataset_counts"]["sft"] == 2
    assert metadata["status"] == "completed"


def test_no_plan_file_means_no_plan_recorded(tmp_path, caplog):
    """A plan is a document somebody wrote. With none to read, this run records
    none and says so -- it does not invent one, and it does not raise."""
    import logging

    repo = DummyLearningRepository([
        _build_learning_record("counter", "910"),
        _build_learning_record("hold", "905"),
    ])
    settings = SimpleNamespace(
        phi4_dataset_dir=tmp_path / "datasets",
        phi4_artifacts_dir=tmp_path / "artifacts",
        phi4_sft_limit=10,
        phi4_context_snippets=2,
        phi4_plan_path=str(tmp_path / "absent" / "phi4_humanization_plan.md"),
    )
    agent = SimpleNamespace(settings=settings, learning_repository=repo,
                            rag_service=DummyRagService())

    with caplog.at_level(logging.WARNING):
        result = Phi4HumanizationFineTuner(agent).dispatch(force=True)

    assert result["status"] == "completed", "a missing plan must not fail the run"
    assert repo.plan_calls == [], "a plan was recorded with no plan to record"
    assert any("No phi4 humanisation plan" in r.getMessage() for r in caplog.records), (
        "the missing plan passed without a word")


def test_phi4_fine_tuner_handles_missing_samples(tmp_path):
    repo = DummyLearningRepository([])
    rag_service = DummyRagService()
    settings = SimpleNamespace(
        phi4_dataset_dir=tmp_path / "datasets",
        phi4_artifacts_dir=tmp_path / "artifacts",
        phi4_sft_limit=5,
    )
    agent = SimpleNamespace(
        settings=settings,
        learning_repository=repo,
        rag_service=rag_service,
    )

    tuner = Phi4HumanizationFineTuner(agent)
    result = tuner.dispatch(force=True)

    assert result["status"] == "skipped"
    sft_info = result["sft_dataset"]
    assert sft_info["sample_count"] == 0
    report_path = tmp_path / "artifacts" / "training_report.json"
    with report_path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    assert report["status"] == "skipped"
