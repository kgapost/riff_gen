"""Preprocessing: DadaGP tokens, own GP files (tuning bypass), unrolling, filters, round trips."""

import guitarpro as gp

from riffgen.dataset import pipeline as pl
from riffgen.dataset.song import Bar, parse_dadagp, raw_to_song
from riffgen.dataset.unroll import unroll
from riffgen.gp import dadagp_bridge, writer
from riffgen.tokens import tunings as T

from conftest import FIXTURES


def _events(song):
    return [[(e.tick, e.track, e.tokens) for e in b.events] for b in song.bars]


def test_dadagp_encoder_rejects_drop_a7():
    song = gp.parse(str(FIXTURES / "progmetal.gp3"))
    song.tracks[0].strings = [gp.GuitarString(number=i + 1, value=v) for i, v in enumerate(T.DROP_A7)]
    try:
        dadagp_bridge.encode_song(song)
        raised = False
    except AssertionError:
        raised = True
    assert raised, "if DadaGP ever accepts Drop A, the tuning bypass is no longer needed"


def test_own_path_equals_dadagp_path(tmp_path):
    """The tuning bypass gives exactly the DadaGP events, with drop_a7 tunings."""
    song = gp.parse(str(FIXTURES / "progmetal.gp3"))
    for t in song.tracks:
        if not t.isPercussionTrack and 24 <= t.channel.instrument <= 31:
            t.strings = [gp.GuitarString(number=i + 1, value=v) for i, v in enumerate(T.DROP_A7)]
    gp.write(song, str(tmp_path / "dropA.gp5"))
    (tmp_path / "d" / "P").mkdir(parents=True)
    import shutil
    shutil.copy(FIXTURES / "progmetal.gp3.tokens.txt", tmp_path / "d" / "P" / "progmetal.gp3.tokens.txt")
    a = pl.process_dadagp_file(tmp_path / "d" / "P" / "progmetal.gp3.tokens.txt", tmp_path / "d",
                               pl.PipelineConfig(analyze=False, use_gp_files=False))
    b = pl.process_gp_file(tmp_path / "dropA.gp5", tmp_path, pl.PipelineConfig(require_tuning="drop_a7", analyze=False))
    assert {t.tuning_name for t in b.tracks.values()} == {"drop_a7"}
    assert _events(a) == _events(b)
    assert len(a.bars) == 99 and a.stats["written_bars"] == 56     # repeats unrolled


def test_gp5_round_trip_is_exact(tmp_path):
    song = gp.parse(str(FIXTURES / "progmetal.gp3"))
    for t in song.tracks:
        if not t.isPercussionTrack and 24 <= t.channel.instrument <= 31:
            t.strings = [gp.GuitarString(number=i + 1, value=v) for i, v in enumerate(T.DROP_A7)]
    gp.write(song, str(tmp_path / "dropA.gp5"))
    cfg = pl.PipelineConfig(require_tuning="drop_a7", analyze=False)
    o = pl.process_gp_file(tmp_path / "dropA.gp5", tmp_path, cfg)
    writer.write_bars(tmp_path / "rt.gp5", o.bars, o.tracks, o.tempo)
    r = pl.process_gp_file(tmp_path / "rt.gp5", tmp_path, cfg)
    assert _events(o) == _events(r)
    assert [b.ticks for b in o.bars] == [b.ticks for b in r.bars]


def test_drop_tuning_frets_become_physical():
    toks = ["artist:x", "downtune:0", "tempo:120", "start", "new_measure", "distorted0:note:s6:f-2", "nfx:palm_mute",
            "wait:960", "distorted0:note:s6:f3", "distorted0:note:s5:f5", "wait:2880", "end"]
    s = raw_to_song(parse_dadagp(toks), "x")
    assert s.tracks[0].tuning_name == "drop_d6"
    assert s.bars[0].events[0].tokens[0] == "distorted0:note:s6:f0"
    assert s.bars[0].events[1].tokens == ["distorted0:note:s5:f5", "distorted0:note:s6:f5"]   # sorted by string


def test_unroll_alternate_endings():
    def bar(i, *tokens):
        return Bar(ticks=3840, measure_tokens=list(tokens), orig=i)
    bars = [bar(0, "measure:repeat_open"), bar(1, "measure:repeat_alternative:1", "measure:repeat_close:1"),
            bar(2, "measure:repeat_alternative:2"), bar(3)]
    assert [b.orig for b in unroll(bars)] == [0, 1, 0, 2, 3]
    bars = [bar(0, "measure:repeat_open"), bar(1, "measure:repeat_close:3"), bar(2)]
    assert [b.orig for b in unroll(bars)] == [0, 1, 0, 1, 0, 1, 0, 1, 2]


def test_filters_on_synthetic_corpus(dadagp_store):
    import csv
    import json
    rows = list(csv.DictReader(open(dadagp_store / "filters.csv")))
    assert rows and all(r["reason"] == "simple riff" for r in rows)
    stats = json.loads((dadagp_store / "stats.json").read_text())
    assert stats["context_bars"] > 0          # simple chug riffs -> context
    assert stats["gap_bars"] > 0              # empty guitar stretches -> gap placeholders
