import pytest

from scripts.testdata import snapshot
from scripts.testdata.guards import UnsafeTargetError


@pytest.mark.parametrize("label", ["demo-base", "v1", "a.b_c-1", "x" * 64])
def test_sensible_labels_are_accepted(label):
    assert snapshot._check_label(label) == label


@pytest.mark.parametrize(
    "label",
    ["", "../escape", "with/slash", "UPPER", "x" * 65, "-leading", None],
)
def test_a_label_cannot_wander_out_of_the_snapshot_directory(label):
    with pytest.raises(snapshot.SnapshotError, match="invalid snapshot label"):
        snapshot._check_label(label)


def test_path_stays_inside_the_snapshot_directory():
    path = snapshot.path_for("demo-base", "bp_testdb")
    assert path.parent == snapshot.SNAPSHOT_DIR
    assert path.name == "demo-base.bp_testdb.dump"


def test_a_snapshot_covers_both_databases():
    """Restoring one without the other leaves the crosswalk pointing at
    suppliers that are not there."""
    assert [database for database, _ in snapshot.PAIRS] == [
        "bp_testdb", "uicanvas_test",
    ]


def test_restore_refuses_a_live_database(monkeypatch, tmp_path):
    monkeypatch.setattr(snapshot, "SNAPSHOT_DIR", tmp_path)
    for database in ("bp_sqldb", "uicanvas"):
        (tmp_path / f"live.{database}.dump").write_bytes(b"x")

    with pytest.raises(UnsafeTargetError):
        snapshot.restore("live", pairs=(("bp_sqldb", "bp"), ("uicanvas", "ui")))


def test_restore_reports_a_missing_snapshot_before_touching_anything(monkeypatch, tmp_path):
    monkeypatch.setattr(snapshot, "SNAPSHOT_DIR", tmp_path)

    def must_not_run(*args, **kwargs):
        raise AssertionError("a database was dropped for a snapshot that is not there")

    monkeypatch.setattr(snapshot, "create_database", must_not_run)

    with pytest.raises(snapshot.SnapshotError, match="no such snapshot"):
        snapshot.restore("never-taken")


def test_listing_an_empty_directory_is_not_an_error(monkeypatch, tmp_path):
    monkeypatch.setattr(snapshot, "SNAPSHOT_DIR", tmp_path)
    assert snapshot.available() == []


def test_listing_reports_label_database_and_size(monkeypatch, tmp_path):
    monkeypatch.setattr(snapshot, "SNAPSHOT_DIR", tmp_path)
    (tmp_path / "demo-base.bp_testdb.dump").write_bytes(b"0" * 2048)
    (tmp_path / "demo-base.uicanvas_test.dump").write_bytes(b"0" * 1024)

    found = snapshot.available()
    assert {s.database for s in found} == {"bp_testdb", "uicanvas_test"}
    assert all(s.label == "demo-base" for s in found)
    assert sum(s.size_bytes for s in found) == 3072


def test_cli_list_on_an_empty_directory_succeeds(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(snapshot, "SNAPSHOT_DIR", tmp_path)
    assert snapshot.main(["list"]) == 0
    assert "no snapshots" in capsys.readouterr().out


def test_cli_reports_a_bad_label_without_a_traceback(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(snapshot, "SNAPSHOT_DIR", tmp_path)
    assert snapshot.main(["restore", "--label", "../escape"]) == 1
    assert "invalid snapshot label" in capsys.readouterr().err


def test_cli_defaults_to_the_demo_base_label(monkeypatch, tmp_path):
    monkeypatch.setattr(snapshot, "SNAPSHOT_DIR", tmp_path)
    captured = {}

    def fake_save(label):
        captured["label"] = label
        return []

    monkeypatch.setattr(snapshot, "save", fake_save)
    assert snapshot.main(["save"]) == 0
    assert captured["label"] == "demo-base"
