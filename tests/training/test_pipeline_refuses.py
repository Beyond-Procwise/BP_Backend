import pytest

from src.training.pipeline import _merge_adapters, _train_model


def test_training_refuses_rather_than_returning_an_empty_adapter(tmp_path):
    """It logged "Training model", made a directory and returned it, having
    trained nothing. A caller could ship that as a fine-tuned model."""
    dataset = tmp_path / "d.jsonl"
    dataset.write_text('{"a": 1}\n')

    class Cfg:
        use_unsloth = False
        base_model = "AgentNick:extract"
        output_dir = tmp_path / "out"

    with pytest.raises(NotImplementedError, match="not implemented"):
        _train_model(Cfg(), dataset)


def test_merging_adapters_refuses_too(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()

    class Cfg:
        adapter_path = adapter
        output_dir = tmp_path / "merged"

    with pytest.raises(NotImplementedError, match="not implemented"):
        _merge_adapters(Cfg())
