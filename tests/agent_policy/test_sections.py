from services.agent_policy.sections import chunk_sections, diff_sections, split_sections

MD = "Intro words\n\n# Scope\nApplies to all.\n\n## Limits\nSee below.\n\n1. Purpose\ntext\n"
CLAUSES = "1. Aim\nx\n1.1 Sub\ny\n4.2) Approvals\nz\n\n10.3.1 Deep clause\nw\n"
PLAIN = "no markers at all\njust text\n"


def _synthetic(n=60):
    return "".join(f"{i}. Clause {i}\n" + ("body " * 20 + "\n\n") for i in range(1, n + 1))


FIXTURES = [MD, CLAUSES, PLAIN, _synthetic()]


def test_concatenation_identity():
    for text in FIXTURES + ["", "\n\n", "# only heading", "  3) indented\r\nwin\r\n"]:
        assert "".join(s["text"] for s in split_sections(text)) == text


def test_starts_are_offsets():
    for text in FIXTURES:
        for s in split_sections(text):
            assert text[s["start"]:s["start"] + len(s["text"])] == s["text"]


def test_markdown_headings_and_preamble():
    secs = split_sections(MD)
    assert [s["reference"] for s in secs] == [None, "Scope", "Limits", "1"]
    assert secs[0]["heading"] == "Preamble"
    assert secs[1]["heading"] == "# Scope"


def test_numbered_clauses():
    secs = split_sections(CLAUSES)
    assert [s["reference"] for s in secs] == ["1", "1.1", "4.2", "10.3.1"]


def test_every_section_lands_in_exactly_one_chunk():
    for text in FIXTURES:
        secs = split_sections(text)
        for max_chars in (9000, 300, 50):
            chunks = chunk_sections(secs, max_chars=max_chars)
            flat = [p for c in chunks for p in c]
            assert "".join(p["text"] for p in flat) == text
            assert [p["start"] for p in flat] == sorted(p["start"] for p in flat)
            assert len({p["start"] for p in flat}) == len(flat)
            assert all(sum(len(p["text"]) for p in c) <= max_chars for c in chunks)
    assert len(split_sections(_synthetic())) == 60
    assert len(chunk_sections(split_sections(_synthetic()), max_chars=2000)) > 1


def test_oversize_section_splits_on_paragraphs_with_part_suffix():
    paras = [f"paragraph {i} " + "w" * 80 for i in range(10)]
    text = "1. Big\n" + "\n\n".join(paras) + "\n"
    chunks = chunk_sections(split_sections(text), max_chars=300)
    parts = [p for c in chunks for p in c]
    assert len(parts) > 1
    assert [p["reference"] for p in parts] == [f"1 (part {n})" for n in range(1, len(parts) + 1)]
    assert "".join(p["text"] for p in parts) == text
    for p in parts[:-1]:
        assert p["text"].endswith("\n\n")


def test_single_unbroken_line_is_still_kept():
    text = "1. Long\n" + "x" * 1000
    parts = [p for c in chunk_sections(split_sections(text), max_chars=100) for p in c]
    assert "".join(p["text"] for p in parts) == text


def test_diff_changed_removed_added():
    old = "1. Top\nkeep\n1.1 Limit\nten\n1.2 Gone\nbye\n"
    new = "1. Top\nkeep\n1.1 Limit\ntwenty\n1.3 New\nhi\n"
    rows = {r["reference"]: r for r in diff_sections(old, new)}
    assert rows["1"]["status"] == "unchanged"
    assert rows["1.1"]["status"] == "changed"
    assert "ten" in rows["1.1"]["before"] and "twenty" in rows["1.1"]["after"]
    assert rows["1.2"]["status"] == "removed" and rows["1.2"]["after"] is None
    assert rows["1.3"]["status"] == "added" and rows["1.3"]["before"] is None
