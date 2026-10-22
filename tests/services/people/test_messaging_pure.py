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


# ---- what the recipient's notification says under the title --------------------------------

def test_notification_body_is_the_preview_for_a_direct_message():
    conv = {"type": "direct", "title": None, "entity_type": None, "entity_id": None}
    assert m.notification_body(conv, "Can you confirm   the budget?\n\nThanks") == "Can you confirm the budget? Thanks"


def test_notification_body_names_the_item_a_thread_is_about():
    conv = {"type": "entity_thread", "title": "INV-0917 Acme", "entity_type": "approval", "entity_id": "123"}
    assert m.notification_body(conv, "Is this one ours?") == "Is this one ours?\nAbout approval 123: INV-0917 Acme"


def test_notification_body_names_the_group_and_renders_mentions_as_names():
    conv = {"type": "group", "title": "Q4 renewals", "entity_type": None, "entity_id": None}
    assert m.notification_body(conv, "@[Nick Geelen](sub-1) your call") == "@Nick Geelen your call\nIn Q4 renewals"


def test_notification_body_is_cut_at_the_preview_length():
    conv = {"type": "direct"}
    out = m.notification_body(conv, "word " * 100)
    assert len(out) <= m.PREVIEW_CHARS and out.endswith("…")
