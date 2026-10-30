"""data/dl.py, offline: parsing an artist page, the tab list, sorting downloads."""

import html
import importlib.util
import json
import sys
from pathlib import Path

# data/dl.py is a stand-alone script, not part of the package
_spec = importlib.util.spec_from_file_location("dl", Path(__file__).parents[1] / "data" / "dl.py")
ug = importlib.util.module_from_spec(_spec)
sys.modules["dl"] = ug
_spec.loader.exec_module(ug)


def _page(tabs):
    data = {"store": {"page": {"data": {"other_tabs": tabs, "album_tabs": [], "pagination": {"pages": []}}}}}
    return f'<div class="js-store" data-content="{html.escape(json.dumps(data))}"></div>'


def _tab(i, song, version, rating, votes, kind="Pro", access="public"):
    return {"id": i, "song_name": song, "artist_name": "Band", "type": kind, "version": version, "rating": rating,
            "votes": votes, "tab_access_type": access, "tuning": "Drop D",
            "tab_url": f"https://tabs.ultimate-guitar.com/tab/band/x-guitar-pro-{i}"}


def test_shortlist_keeps_free_community_tabs_above_rating():
    tabs = [_tab(1, "Riff", 1, 4.5, 10), _tab(2, "Riff", 2, 4.8, 3), _tab(3, "Other", 1, 3.9, 50),
            _tab(4, "Paid", 1, 4.9, 99, kind="TabPro"), _tab(5, "Official", 1, 4.9, 99, kind="official")]
    found = ug.community_gp_tabs(ug.page_data(_page(tabs)))
    assert [t["id"] for t in found] == [1, 2, 3]
    assert [r["id"] for r in ug.shortlist(found, 4.0, 1, False)] == [1, 2]
    assert [r["id"] for r in ug.shortlist(found, 4.0, 1, True)] == [2]


def test_sort_names_versions_as_song_family(tmp_path):
    rows = ug.shortlist(ug.community_gp_tabs(ug.page_data(_page([_tab(1, "Riff", 1, 4.5, 10),
                                                                   _tab(2, "Riff", 2, 4.8, 3)]))), 4.0, 1, False)
    rows[0]["genre"] = "Metal"  # a column only another tool writes
    lst = tmp_path / "list.csv"
    ug.write_list(rows, lst)
    dl = tmp_path / "dl"
    dl.mkdir()
    (dl / "Band - Riff (ver 2).gp5").write_bytes(b"x")
    ug.main(["sort", "--list", str(lst), "--downloads", str(dl), "--dest", str(tmp_path / "album")])
    assert (tmp_path / "album" / "Band" / "Band - Riff__v2.gp5").is_file()
    after = ug.read_list(lst)
    assert {r["id"]: r["status"] for r in after} == {1: "to download", 2: "sorted"}
    assert after[0]["genre"] == "Metal"
    assert sorted(tmp_path.iterdir()) == sorted([lst, dl, tmp_path / "album"])  # no .json / .html copies


def test_sort_uses_the_download_manifest(tmp_path):
    rows = ug.shortlist(ug.community_gp_tabs(ug.page_data(_page([_tab(7, "Riff", 1, 4.5, 10),
                                                                   _tab(8, "Riff", 1, 4.6, 10)]))), 4.0, 1, False)
    lst = tmp_path / "list.csv"
    ug.write_list(rows, lst)
    dl = tmp_path / "dl"
    dl.mkdir()
    (dl / "Band - Riff__id8.gp5").write_bytes(b"x")
    state = tmp_path / "state.json"
    state.write_text(json.dumps({rows[1]["tab_url"]: {"status": "ok", "path": str(dl / "Band - Riff__id8.gp5")}}))
    ug.main(["sort", "--list", str(lst), "--downloads", str(dl), "--dest", str(tmp_path / "album"),
             "--manifest", str(state)])
    assert {r["id"]: r["status"] for r in ug.read_list(lst)} == {7: "to download", 8: "sorted"}


# ---------------------------------------------------------------------------------------------------------
# curation of new downloads

import guitarpro as gp  # noqa: E402
import pytest  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "progmetal.gp3"


@pytest.fixture(autouse=True)
def _restore_converter():
    """dl.py switches the converter to its 8-string mode; other test files expect the defaults."""
    yield
    cv = sys.modules.get("convert")
    if cv is not None:
        cv._init(False, False, "high", 0)
        cv.MAX_STRINGS = 7
    ug.cur._CV = None


@pytest.fixture
def work(tmp_path, monkeypatch):
    """A tiny library: data/ with tabs - rock holding one tab of 'Band' and an empty landing folder."""
    w = tmp_path / "data"
    (w / "tabs - rock").mkdir(parents=True)
    (w / "tabs - generic").mkdir()
    for name, value in (("WORK", w), ("ERROR_DIR", w / "errors")):
        monkeypatch.setattr(ug, name, value)
    gp.write(gp.parse(str(FIXTURE)), str(w / "tabs - rock" / "Band - Old Song.gp5"), version=(5, 1, 0))
    return w


def _notes(song):
    return sum(len(b.notes) for t in song.tracks for m in t.measures for v in m.voices for b in v.beats)


def _tab_with_change(path, instrument=29):
    """The fixture's first guitar changes to ``instrument`` halfway; written to ``path`` as GP5."""
    song = gp.parse(str(FIXTURE))
    beats = [b for m in song.tracks[0].measures for v in m.voices for b in v.beats if b.notes]
    beats[len(beats) // 2].effect.mixTableChange = gp.models.MixTableChange(instrument=gp.models.MixTableItem(instrument))
    gp.write(song, str(path), version=(5, 1, 0))
    return song


def test_download_is_split_by_instrument_made_seven_string_and_filed_by_artist(work):
    src = work / "tabs - generic" / "Band - Riff__v2.gp5"
    _tab_with_change(src)
    before = _notes(gp.parse(str(src)))
    lib = ug.Library(work)
    status, final, note = ug.curate_file(src, "Jazz", lib)           # the artist is known: its folder wins over the genre
    assert status == "ok", note
    final = Path(final)
    assert final.parent.name == "tabs - rock" and final.name == "Band - Riff.gp5"      # no __v2
    out = gp.parse(str(final))
    assert len(out.tracks) == 6 and out.tracks[1].name == "Guitar (Overdriven Guitar)"
    assert [t.channel.instrument for t in out.tracks[:2]] == [30, 29]
    assert _notes(out) == before
    assert not any(b.effect.mixTableChange is not None and b.effect.mixTableChange.instrument is not None
                   for t in out.tracks for m in t.measures for v in m.voices for b in v.beats)
    channels = [t.channel.channel for t in out.tracks]
    assert len(set(channels)) == len(channels)
    assert all(len(t.strings) == 7 for t in out.tracks if t.name.startswith(("Guitar", "Acoustic")))
    assert [len(t.strings) for t in out.tracks if t.name == "Bass"] == [5]      # basses are left alone


def test_no_free_channel_keeps_the_first_instrument(work, monkeypatch):
    def full(song):
        raise ug.NoChannel
    monkeypatch.setattr(ug.cur, "_free_channel", full)
    src = work / "tabs - generic" / "Band - Riff.gp5"
    _tab_with_change(src)
    before = _notes(gp.parse(str(src)))
    status, final, note = ug.curate_file(src, "", ug.Library(work))
    out = gp.parse(final)
    assert status == "ok" and "first instrument kept" in note
    assert len(out.tracks) == 5 and out.tracks[0].channel.instrument == 30 and _notes(out) == before


def test_unknown_artist_goes_by_genre_else_to_metal_generic(work):
    lib = ug.Library(work)
    for i, (genre, folder) in enumerate((("Jazz, Instrumental jazz", "tabs - jazz"), ("", "tabs - metal generic"),
                                         ("Something Odd", "tabs - metal generic"), ("Alternative", "tabs - rock"))):
        src = work / "tabs - generic" / f"Nobody{i} - Song.gp5"
        gp.write(gp.parse(str(FIXTURE)), str(src), version=(5, 1, 0))
        status, final, note = ug.curate_file(src, genre, lib)
        assert status == "ok" and Path(final).parent.name == folder, (genre, final)


def test_same_file_is_dropped_and_one_version_of_a_song_stays(work):
    lib = ug.Library(work)
    old = work / "tabs - rock" / "Band - Old Song.gp5"
    src = work / "tabs - generic" / "Band - Old Song (Rock).gp5"       # genre tag in the name is dropped
    gp.write(gp.parse(str(FIXTURE)), str(src), version=(5, 1, 0))
    status, final, note = ug.curate_file(src, "", lib)
    assert status in ("ok", "duplicate")
    if status == "duplicate":
        assert (work / "__removed").is_dir() and old.is_file()
    assert len(list((work / "tabs - rock").glob("Band - Old Song*"))) == 1


def test_only_the_largest_version_of_a_song_stays(work):
    lib = ug.Library(work)
    sizes = []
    for i, keep_tracks in enumerate((3, 5, 1)):
        song = gp.parse(str(FIXTURE))
        song.tracks = song.tracks[:keep_tracks]
        src = work / "tabs - generic" / f"Band - Hit__v{i + 1}.gp5"
        gp.write(song, str(src), version=(5, 1, 0))
        sizes.append(src.stat().st_size)
        status, final, note = ug.curate_file(src, "", lib)
        assert status in ("ok", "duplicate"), note
    left = list((work / "tabs - rock").glob("Band - Hit*"))
    assert len(left) == 1 and left[0].stat().st_size > min(sizes)        # the 5-track version
    assert len(list((work / "__removed" / "tabs - rock").glob("Band - Hit*"))) == 2


def test_a_tab_the_training_would_skip_is_rejected(work):
    song = gp.parse(str(FIXTURE))
    song.tracks = [t for t in song.tracks if "guitar" not in t.name.lower() and not 24 <= t.channel.instrument <= 31]
    src = work / "tabs - generic" / "Band - No Guitar.gp5"
    gp.write(song, str(src), version=(5, 1, 0))
    status, final, note = ug.curate_download(src, ug.Target(url="u", tab_id="9"), ug.Library(work))
    assert status == "rejected" and "no guitar track" in note and not src.exists()
    assert (work / "__removed" / "rejected" / "Band - No Guitar.gp5").is_file()


def test_tidy_artist_swaps_only_when_sure(work):
    lib = ug.Library(work)
    (work / "tabs - rock" / "Steve Vai - A.gp5").write_bytes(b"x")
    lib = ug.Library(work)
    assert lib.tidy_artist("Vai, Steve") == "Steve Vai"             # "Steve Vai" exists in the library
    assert lib.tidy_artist("Beatles, The") == "The Beatles"
    assert lib.tidy_artist("Moore, Vinnie") == "Moore, Vinnie"      # nothing says which order is right
    assert lib.tidy_artist("Earth, Wind and Fire") == "Earth, Wind and Fire"


def test_unhealthy_download_goes_to_errors(work):
    src = work / "tabs - generic" / "Band - Broken.gp5"
    src.write_bytes(b"<html>nope</html>" + b" " * 600)
    t = ug.Target(url="https://x/y-123", tab_id="123")
    status, final, note = ug.curate_download(src, t, ug.Library(work))
    assert status == "failed" and note.startswith("curate:")
    assert (work / "errors" / "123.gp5.bad").is_file() and not src.exists()


def test_eight_string_track_is_cut_with_the_cheapest_method():
    cv = ug.convert_module()
    song = gp.parse(str(FIXTURE))
    t = song.tracks[0]
    t.strings += [gp.GuitarString(7, 35), gp.GuitarString(8, 30)]      # two extra low strings nobody plays
    assert cv.squeeze(song) == 0
    assert len(t.strings) == 7 and [s.value for s in t.strings[:6]] == [64, 59, 55, 50, 45, 40]


def test_best_version_is_fetched_first_by_rating_then_votes():
    a = ug.Target(url="a", rating=4.5, votes=10, version=1)
    b = ug.Target(url="b", rating=4.5, votes=90, version=2)
    c = ug.Target(url="c", rating=4.9, votes=1, version=1)
    assert [t.url for t in sorted([a, b, c], key=lambda t: (-t.rating, -t.votes, -t.version))] == ["c", "b", "a"]


def test_download_with_an_eight_string_guitar_is_cut_to_seven_strings(work, monkeypatch):
    monkeypatch.setattr(ug, "TRAIN_FILTERS", False)       # the cut keeps the low F# string: riff.py calls that a bass
    spec = importlib.util.spec_from_file_location("test_convert_helpers", Path(__file__).parent / "test_convert.py")
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    src = work / "tabs - generic" / "Band - Eight.gp"
    helpers._write_gp(src, helpers._eight("electricGuitar"))
    status, final, note = ug.curate_file(src, "Metal", ug.Library(work))
    assert status == "ok", note
    song = gp.parse(final)
    assert Path(final).parent.name == "tabs - rock" and len(song.tracks[0].strings) == 7     # the artist "Band" lives there
    assert "gpif -> gp5" in note


def test_download_with_an_eight_entry_piano_tuning_is_cut_too(work, monkeypatch):
    monkeypatch.setattr(ug, "TRAIN_FILTERS", False)       # a piano-only tab: the training would skip it
    spec = importlib.util.spec_from_file_location("test_convert_helpers", Path(__file__).parent / "test_convert.py")
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    src = work / "tabs - generic" / "Band - Piano.gp"
    helpers._write_gp(src, helpers._eight("acousticPiano"))       # not a guitar, yet read with all 8 strings
    status, final, note = ug.curate_file(src, "Metal", ug.Library(work))
    assert status == "ok", note
    assert all(len(t.strings) <= 7 for t in gp.parse(final).tracks)
