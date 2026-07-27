import pytest

from scripts.testdata.build import main, parse_args


def test_defaults_point_at_the_test_databases():
    args = parse_args([])
    assert args.target == "bp_testdb"
    assert args.uicanvas_target == "uicanvas_test"
    assert args.seed == 42
    assert args.drop_first is False


def test_seed_and_target_are_overridable():
    args = parse_args(["--target", "scratch_db", "--seed", "7", "--drop-first"])
    assert args.target == "scratch_db"
    assert args.seed == 7
    assert args.drop_first is True


@pytest.mark.parametrize("target", ["bp_sqldb", "uicanvas", "ses", "postgres"])
def test_main_refuses_live_targets_with_exit_code_2(target, capsys):
    assert main(["--target", target]) == 2
    assert "refuses" in capsys.readouterr().err


@pytest.mark.parametrize("target", ["bp_sqldb", "uicanvas"])
def test_main_refuses_live_uicanvas_targets(target, capsys):
    assert main(["--uicanvas-target", target]) == 2
    assert "refuses" in capsys.readouterr().err


def test_drop_first_does_not_bypass_the_guard(capsys):
    assert main(["--target", "bp_sqldb", "--drop-first"]) == 2
    assert "refuses" in capsys.readouterr().err


def test_deal_assignment_failure_does_not_abort_the_build(monkeypatch):
    """A grouping failure is a product finding to report, not a reason to throw
    away a good 190,000-row build."""
    from scripts.testdata import build

    def boom(*args, **kwargs):
        raise RuntimeError("linking engine exploded")

    monkeypatch.setattr(build, "_assign_deals_impl", boom)
    result = build.assign_deals_on("bp_testdb")
    assert "error" in result
    assert "linking engine exploded" in result["error"]
