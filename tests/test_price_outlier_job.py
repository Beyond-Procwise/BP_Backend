from src.services.backend_scheduler import (
    BackendScheduler, price_outlier_enabled, price_outlier_interval_minutes,
)


def test_job_is_off_by_default_until_the_first_run_is_reviewed(monkeypatch):
    """190,000 seeded lines could bury the queue. The job stays off until a
    human has looked at a dry run."""
    monkeypatch.delenv("PRICE_OUTLIER_ENABLED", raising=False)
    assert price_outlier_enabled() is False


def test_job_turns_on_only_when_explicitly_enabled(monkeypatch):
    for value, expected in (("1", True), ("true", True), ("True", True),
                            ("0", False), ("", False), ("yes", False)):
        monkeypatch.setenv("PRICE_OUTLIER_ENABLED", value)
        assert price_outlier_enabled() is expected, value


def test_interval_defaults_to_an_hour_and_survives_rubbish(monkeypatch):
    monkeypatch.delenv("PRICE_OUTLIER_INTERVAL_MINUTES", raising=False)
    assert price_outlier_interval_minutes() == 60
    monkeypatch.setenv("PRICE_OUTLIER_INTERVAL_MINUTES", "not-a-number")
    assert price_outlier_interval_minutes() == 60
    monkeypatch.setenv("PRICE_OUTLIER_INTERVAL_MINUTES", "0")
    assert price_outlier_interval_minutes() == 1


def test_the_job_has_a_name():
    assert BackendScheduler.PRICE_OUTLIER_JOB_NAME == "price-outlier-scan"
