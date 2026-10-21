import copy

from riffgen.check import plagiarism as plag
from riffgen.check import playability as play
from riffgen.dataset.song import Event
from riffgen.dataset.store import Store, song_from_view
from riffgen.utils.config import load_config
from riffgen import CONFIG_DIR


def _index(tmp_path, album_store):
    cfg = load_config(CONFIG_DIR / "plagiarism.yaml")
    st = Store(album_store)
    v = st.load_vocab()
    idx = plag.Index(tmp_path / "i.sqlite")
    songs = [song_from_view(st.view(i), v) for i in range(len(st))]
    for s, meta in zip(songs, st.songs):
        idx.add_song(s, own=meta.get("set") == "own", cfg=cfg)
    idx.finalize(cfg)
    return idx, songs, st


def test_copy_is_red_and_own_copy_is_self_quote(tmp_path, album_store):
    idx, songs, st = _index(tmp_path, album_store)
    fav = next(i for i, m in enumerate(st.songs) if m.get("set") != "own")
    m = plag.check_song(idx, songs[fav], range(2, 10))
    assert m.verdict == "RED" and not m.self_quote
    own = next(i for i, mm in enumerate(st.songs) if mm.get("set") == "own")
    m2 = plag.check_song(idx, songs[own], range(2, 10))
    assert m2.verdict == "RED" and m2.self_quote


def test_transposed_copy_is_caught_by_l2(tmp_path, album_store):
    idx, songs, st = _index(tmp_path, album_store)
    s = copy.deepcopy(songs[next(i for i, m in enumerate(st.songs) if m.get("set") != "own")])
    for bar in s.bars:   # move everything up 2 frets: same riff, another key
        for ev in bar.events:
            ev.tokens = [t.rsplit(":f", 1)[0] + f":f{int(t.rsplit(':f', 1)[1]) + 2}" if ":note:" in t else t
                         for t in ev.tokens]
    m = plag.check_song(idx, s, range(2, 10))
    assert m.verdict == "RED" and m.level == "L2"


def test_playability_flags_and_fix(album_store):
    st = Store(album_store)
    v = st.load_vocab()
    s = song_from_view(st.view(0), v)
    # an impossible chord: frets 1 and 12 at the same time (playable as s7 f8 + s1 f12)
    bar = s.bars[0]
    evs = [e for e in bar.events if not (e.track == 0 and e.tick == 0)]
    evs.append(Event(0, 0, ["distorted0:note:s1:f12", "distorted0:note:s6:f1"]))
    bar.events = sorted(evs, key=lambda e: (e.tick, e.track))
    rep = play.check_song(s, range(0, 1))
    assert any(i.kind == "span" and i.severity == "error" for i in rep.issues)
    play.fix_song(s, range(0, 1))
    rep2 = play.check_song(s, range(0, 1))
    assert not any(i.kind == "span" and i.severity == "error" for i in rep2.issues)
