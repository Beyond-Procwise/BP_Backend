import pytest

from services.people import messaging as m
from services.people.admin import PeopleError


def test_direct_key_is_order_independent():
    assert m.direct_key("b", "a") == m.direct_key("a", "b") == "a|b"


@pytest.mark.parametrize("body", ["", "   ", None, 5])
def test_empty_body_refused(body):
    with pytest.raises(PeopleError):
        m.clean_body(body)


def test_body_limit_and_trim():
    assert m.clean_body("  hi \n") == "hi"
    with pytest.raises(PeopleError):
        m.clean_body("x" * (m.MAX_BODY + 1))


def test_mentions_from_markup_and_list_deduped_in_order():
    body = "ping @[Jane](u-1) and @[Carl](u-2) and @[Jane again](u-1)"
    assert m.parse_mentions(body, ["u-3", "u-2"]) == ["u-1", "u-2", "u-3"]
    assert m.parse_mentions("no mention, an @ alone and @[broken(u-1)") == []


def test_too_many_mentions():
    with pytest.raises(PeopleError):
        m.parse_mentions("x", [f"u{i}" for i in range(m.MAX_MENTIONS + 1)])


def test_attachments_metadata_only():
    assert m.clean_attachments([{"name": "q.pdf", "ref": "s3://x", "data": "AAAA", "size": 5}]) == [
        {"name": "q.pdf", "ref": "s3://x", "size": 5}]
    for bad in ("x", [{"ref": "r"}], [{"name": " "}], [{"name": "a"}] * (m.MAX_ATTACHMENTS + 1)):
        with pytest.raises(PeopleError):
            m.clean_attachments(bad)
