"""data_curate.py (a local collection) and data_dl.py get (tab URLs), offline."""

import csv
import importlib.util
import shutil
import sys
from pathlib import Path

import guitarpro as gp
import pytest

ROOT = Path(__file__).parents[1]
FIXTURE = Path(__file__).parent / "fixtures" / "progmetal.gp3"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"data_{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


cur = _load("curate")
dl = _load("dl")


@pytest.fixture(autouse=True)
def _restore_converter():
    """curate.py switches the converter to its 8-string mode; other test files expect the defaults."""
    yield
    cv = sys.modules.get("convert")
    if cv is not None:
        cv._init(False, False, "high", 0)
        cv.MAX_STRINGS = 7
    cur._CV = None


def test_collection_is_curated_into_out_and_originals_stay(tmp_path):
    src = tmp_path / "mine"
    (src / "sub").mkdir(parents=True)
    shutil.copy(FIXTURE, src / "Vai, Steve - Riff (Rock).gp3")
    shutil.copy(FIXTURE, src / "sub" / "Steve Vai - Riff.gp3")
    shutil.copy(FIXTURE, src / "noname.gp3")
    (src / "bad.gp5").write_bytes(b"<html>nope</html>" + b" " * 600)
    out = tmp_path / "out"
    assert cur.main([str(src), "--out", str(out)]) == 1                  # one file failed
    names = sorted(p.name for p in out.glob("*.gp5"))
    assert names == ["Opeth - Bleak.gp5", "Steve Vai - Riff.gp5"]      # named from the tab; the copy is a duplicate
    assert all(len(t.strings) == 7 for t in gp.parse(str(out / names[1])).tracks if t.name.startswith("Guitar"))
    assert len(list((out / "__removed").glob("*.gp5"))) == 1
    assert len(list(src.rglob("*.gp*"))) == 4                           # nothing touched without --move
    rows = list(csv.DictReader(open(out / "curate_report.csv", encoding="utf-8")))
    assert sorted(r["status"] for r in rows) == ["duplicate", "failed", "ok", "ok"]


def test_move_and_genre_folders(tmp_path):
    src = tmp_path / "mine"
    src.mkdir()
    shutil.copy(FIXTURE, src / "Band - Song.gp3")
    out = tmp_path / "out"
    (out / "tabs - jazz").mkdir(parents=True)
    gp.write(gp.parse(str(FIXTURE)), str(out / "tabs - jazz" / "Band - Old.gp5"), version=(5, 1, 0))
    assert cur.main([str(src), "--out", str(out), "--genre-folders", "--genre", "Metal", "--move"]) == 0
    assert (out / "tabs - jazz" / "Band - Song.gp5").is_file() and not (src / "Band - Song.gp3").exists()


def test_out_inside_a_source_is_refused(tmp_path):
    with pytest.raises(SystemExit):
        cur.main([str(tmp_path), "--out", str(tmp_path / "out")])


def test_tab_row_from_url():
    r = dl.tab_row("https://tabs.ultimate-guitar.com/tab/a-perfect-circle/3-libras-guitar-pro-214593?app=1")
    assert (r["id"], r["artist"], r["song"]) == (214593, "A Perfect Circle", "3 Libras")
    assert r["tab_url"].endswith("-214593")
    with pytest.raises(ValueError):
        dl.tab_row("https://www.ultimate-guitar.com/artist/meshuggah_3907")
