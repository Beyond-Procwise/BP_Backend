import pytest

from src.orchestration import agentnick_control as control


@pytest.fixture
def tools(make_skill, monkeypatch):
    make_skill()
    monkeypatch.setenv("SKILLS_DIR", str(make_skill.root))
    return {t.name: t for t in control._skill_tools()}


def test_both_tools_are_in_the_agents_tool_set(make_skill, monkeypatch):
    monkeypatch.setenv("SKILLS_DIR", str(make_skill.root))
    names = [t.name for t in control.build_tools(None)]
    assert "list_skills" in names and "use_skill" in names


def test_list_skills_gives_slug_and_description(tools):
    out = tools["list_skills"].handler()
    assert out["skills"] == [{"name": "demo-skill", "description": "Does a demo job. Use when demoing."}]


def test_use_skill_returns_playbook_and_config(tools):
    out = tools["use_skill"].handler(name="demo-skill")
    assert out["found"] is True and "## Purpose" in out["playbook"] and out["config"] == {"thresholds": {}}


@pytest.mark.parametrize("name", ["nope", "../etc/passwd", "demo-skill/../x", "Demo", "", None])
def test_unknown_or_path_like_names_are_refused(tools, name):
    out = tools["use_skill"].handler(name=name)
    assert out["found"] is False and out["available"] == ["demo-skill"]


def test_a_broken_skill_is_reported_not_served(make_skill, monkeypatch):
    make_skill(name="bad-skill", config=False)
    monkeypatch.setenv("SKILLS_DIR", str(make_skill.root))
    out = {t.name: t for t in control._skill_tools()}["use_skill"].handler(name="bad-skill")
    assert out["found"] is False and "config.yaml" in out["note"]


def test_description_says_a_skill_is_advice_only(tools):
    d = tools["use_skill"].description.lower()
    assert "advice" in d and "approval" in d


def test_a_real_skill_outside_the_skills_folder_cannot_be_reached(make_skill, monkeypatch, tmp_path):
    from tests.skills.conftest import write_skill
    outside = tmp_path / "outside"
    outside.mkdir()
    write_skill(outside, name="secret-skill")
    root = tmp_path / "root"
    root.mkdir()
    write_skill(root, name="demo-skill")
    monkeypatch.setenv("SKILLS_DIR", str(root))
    out = {t.name: t for t in control._skill_tools()}["use_skill"].handler(name="../outside/secret-skill")
    assert out["found"] is False


def _long_skill(make_skill, monkeypatch):
    from tests.skills.conftest import SECTIONS, skill_text
    body = "\n\n".join(f"## {s}\n\n" + ("word " * 400) for s in SECTIONS)
    text = "---\nname: long-skill\ndescription: A long one. Use when long.\n---\n\n# Long\n\n" + body + "\n"
    make_skill(name="long-skill", text=text)
    monkeypatch.setenv("SKILLS_DIR", str(make_skill.root))
    return {t.name: t for t in control._skill_tools()}["use_skill"], text


def test_a_long_playbook_comes_in_parts_that_each_fit_the_result_cap(make_skill, monkeypatch):
    from src.services import tool_runtime
    use, _ = _long_skill(make_skill, monkeypatch)
    first = use.handler(name="long-skill")
    assert first["parts_total"] > 1 and first["part"] == 1 and "read all" in first["note"].lower()
    for n in range(1, first["parts_total"] + 1):
        out = use.handler(name="long-skill", part=n)
        assert len(tool_runtime._render_result(out)) <= tool_runtime._MAX_RESULT_CHARS, n


def test_the_parts_together_are_the_whole_playbook_in_order(make_skill, monkeypatch):
    from src.services.skills import checks
    use, text = _long_skill(make_skill, monkeypatch)
    n = use.handler(name="long-skill")["parts_total"]
    joined = "".join(use.handler(name="long-skill", part=i)["playbook"] for i in range(1, n + 1))
    assert joined == checks.split_frontmatter(text)[1]


def test_a_part_out_of_range_is_refused(make_skill, monkeypatch):
    use, _ = _long_skill(make_skill, monkeypatch)
    assert use.handler(name="long-skill", part=99)["found"] is False


def test_every_real_skill_fits_in_parts(monkeypatch):
    from pathlib import Path
    from src.services import tool_runtime
    monkeypatch.setenv("SKILLS_DIR", str(Path(__file__).resolve().parents[2] / "skills"))
    use = {t.name: t for t in control._skill_tools()}["use_skill"]
    for s in use_names(control):
        first = use.handler(name=s)
        for n in range(1, first["parts_total"] + 1):
            assert len(tool_runtime._render_result(use.handler(name=s, part=n))) <= tool_runtime._MAX_RESULT_CHARS


def use_names(control):
    return [x["name"] for x in {t.name: t for t in control._skill_tools()}["list_skills"].handler()["skills"]]


def test_the_system_prompt_routes_policy_questions_and_big_jobs():
    s = control._SYSTEM
    assert "check_policy" in s and "list_skills" in s and "use_skill" in s
    assert s.index("check_policy") < s.index("fetch it with get_policy")  # the specific route is read first
