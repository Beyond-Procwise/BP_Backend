import json

from src.services.truth.build_set import build


def _corpus(tmp_path, rows):
    p = tmp_path / "corpus.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    return str(p)


def test_examples_without_source_are_recovered_when_possible(tmp_path):
    rows = [{"doc_type": "Invoice", "pk": 1, "file_path": "documents/a.pdf",
             "source_text": "", "extracted": {"header": {"invoice_id": "INV-1"}}}]
    out = tmp_path / "out.jsonl"
    summary = build(_corpus(tmp_path, rows), str(out),
                    recover=lambda fp: "Invoice No: INV-1")
    assert summary["recovered"] == 1
    assert summary["counts"]["verified"] == 1


def test_an_unrecoverable_example_is_kept_as_unverifiable_not_dropped(tmp_path):
    rows = [{"doc_type": "Invoice", "pk": 2, "file_path": "documents/b.pdf",
             "source_text": "", "extracted": {"header": {"invoice_id": "INV-2"}}}]
    out = tmp_path / "out.jsonl"
    summary = build(_corpus(tmp_path, rows), str(out), recover=lambda fp: None)
    assert summary["examples"] == 1, "an example must never be silently dropped"
    assert summary["unrecoverable"] == 1
    assert summary["counts"]["unverifiable"] == 1


def test_a_malformed_row_is_reported_and_skipped(tmp_path):
    p = tmp_path / "corpus.jsonl"
    p.write_text('{"broken": ')
    out = tmp_path / "out.jsonl"
    summary = build(str(p), str(out), recover=lambda fp: None)
    assert summary["malformed"] == 1
    assert summary["examples"] == 0


def test_every_labelled_row_carries_its_provenance(tmp_path):
    rows = [{"doc_type": "Invoice", "pk": 3, "file_path": "documents/c.pdf",
             "source_text": "Invoice No: INV-3",
             "extracted": {"header": {"invoice_id": "INV-3"}}}]
    out = tmp_path / "out.jsonl"
    build(_corpus(tmp_path, rows), str(out), recover=lambda fp: None)
    row = json.loads(out.read_text().strip())
    assert row["pk"] == 3
    assert row["fields"]["invoice_id"]["rule"] == "text-match"
    assert row["source"] == "corpus"


def test_default_recover_reads_full_text_not_text(monkeypatch):
    """ParsedDocument exposes `full_text`. Reading `text` returned None for every
    document, so recovery silently recovered nothing while the S3 download and
    the PDF conversion both succeeded."""
    from src.services.truth import build_set

    class FakeParsed:
        full_text = "Invoice No: INV-7"

    class FakeParser:
        @staticmethod
        def parse(path):
            return FakeParsed()

    import sys
    import types
    module = types.ModuleType("src.services.extraction.parser")
    module.parse = FakeParser.parse
    monkeypatch.setitem(sys.modules, "src.services.extraction.parser", module)
    pkg = types.ModuleType("src.services.extraction")
    pkg.parser = module
    monkeypatch.setitem(sys.modules, "src.services.extraction", pkg)

    assert build_set._default_recover("documents/x.pdf") == "Invoice No: INV-7"
