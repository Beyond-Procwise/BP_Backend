import pytest
from src.services.extraction_v3.extractors.base import Extractor
from src.services.extraction_v3.yaml_schema.registry import (
    register_extractor, known_extractors, get_extractor
)
from src.services.extraction_v3.schemas.candidate import Candidate
from src.services.extraction_v3.schemas.parsed_document import ParsedDocument, Page
from src.services.extraction_v3.yaml_schema.loader import DocSchema, load_doc_schema


def _empty_parsed():
    return ParsedDocument(
        source_path="/tmp/x.pdf", file_format="pdf-native",
        pages=[Page(index=0, width=1, height=1, rotation=0,
                    regions=[], tables=[], tokens=[])],
        full_text="", parser_backend="docling", parser_confidence=1.0,
    )


def test_extractor_is_abstract():
    """Cannot instantiate the base directly."""
    with pytest.raises(TypeError):
        Extractor()


def test_register_extractor_decorator():
    """The @register_extractor decorator inserts into the runtime registry."""
    @register_extractor("test_extractor_unique_xyz")
    class _Mock(Extractor):
        def produce_candidates(self, parsed, schema): return []
    assert "test_extractor_unique_xyz" in known_extractors()
    cls = get_extractor("test_extractor_unique_xyz")
    inst = cls()
    parsed = _empty_parsed()
    schema = load_doc_schema("invoice")
    assert inst.produce_candidates(parsed, schema) == []


def test_subclass_must_implement_produce_candidates():
    """Forgetting produce_candidates → TypeError on instantiation."""
    class _Bad(Extractor): pass
    with pytest.raises(TypeError):
        _Bad()
