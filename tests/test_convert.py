"""The GP -> GP5 converter (data_convert.py): format sniffing, GP3/GP4/.gtp rewriting, the GP6/GP7 reader,
what is kept, and moving unreadable files to <folder>/__erroneous."""

import importlib.util
import shutil
import struct
import sys
import zipfile
from pathlib import Path

import guitarpro as gp

# data_convert.py is a stand-alone script at the repo root, not part of the package
_spec = importlib.util.spec_from_file_location("convert", Path(__file__).parents[1] / "data_convert.py")
_mod = importlib.util.module_from_spec(_spec)
sys.modules["convert"] = _mod
_spec.loader.exec_module(_mod)
convert, sniff, main = _mod.convert, _mod.sniff, _mod.main

FIXTURES = Path(__file__).parent / "fixtures"
FIXTURE = FIXTURES / "progmetal.gp3"

GPIF = """<?xml version="1.0" encoding="utf-8"?>
<GPIF><Score><Title>Tiny</Title><Artist>Nobody</Artist><Album>Small</Album></Score>
<MasterTrack><Tracks>0</Tracks><Automations><Automation><Type>Tempo</Type><Bar>0</Bar><Position>0</Position><Value>96 2</Value></Automation></Automations></MasterTrack>
<Tracks><Track id="0"><Name>Guitar</Name><GeneralMidi><Program>29</Program></GeneralMidi>
<Staves><Staff id="0"><Properties><Property name="Tuning"><Pitches>38 45 50 55 59 64</Pitches></Property></Properties></Staff></Staves></Track></Tracks>
<MasterBars><MasterBar><Time>4/4</Time><Bars>0</Bars></MasterBar><MasterBar><Time>3/4</Time><Bars>1</Bars></MasterBar></MasterBars>
<Bars><Bar id="0"><Voices>0 -1 -1 -1</Voices></Bar><Bar id="1"><Voices>1 -1 -1 -1</Voices></Bar></Bars>
<Voices><Voice id="0"><Beats>0 1 2</Beats></Voice><Voice id="1"><Beats>3</Beats></Voice></Voices>
<Beats><Beat id="0"><Rhythm ref="0"/><Notes>0 1</Notes></Beat><Beat id="1"><Rhythm ref="0"/></Beat><Beat id="2"><Rhythm ref="1"/><Notes>2</Notes></Beat><Beat id="3"><Rhythm ref="2"/><Notes>3</Notes></Beat></Beats>
<Notes>
<Note id="0"><Properties><Property name="String"><String>0</String></Property><Property name="Fret"><Fret>0</Fret></Property></Properties></Note>
<Note id="1"><Properties><Property name="String"><String>1</String></Property><Property name="Fret"><Fret>2</Fret></Property></Properties></Note>
<Note id="2"><Properties><Property name="String"><String>5</String></Property><Property name="Fret"><Fret>5</Fret></Property><Property name="PalmMuted"><Enable/></Property></Properties></Note>
<Note id="3"><Properties><Property name="String"><String>2</String></Property><Property name="Fret"><Fret>7</Fret></Property></Properties></Note>
</Notes>
<Rhythms><Rhythm id="0"><NoteValue>Quarter</NoteValue></Rhythm><Rhythm id="1"><NoteValue>Half</NoteValue></Rhythm><Rhythm id="2"><NoteValue>Half</NoteValue><AugmentationDot count="1"/></Rhythm></Rhythms>
</GPIF>"""


def _note(i, string, fret, extra=""):
    return (f'<Note id="{i}"><Properties><Property name="String"><String>{string}</String></Property>'
            f'<Property name="Fret"><Fret>{fret}</Fret></Property>{extra}</Properties></Note>')


# GP6 style: no <Staves>; the piano ("-gs" = grand staff) owns two of the three bars of a master bar,
# the 7-string guitar keeps its tuning in the track's own properties, and the drum kit has two hits at once
GPIF6 = f"""<?xml version="1.0" encoding="utf-8"?>
<GPIF><Score><Title>Six</Title></Score><MasterTrack><Automations/></MasterTrack>
<Tracks>
<Track id="0"><Name>Piano</Name><Instrument ref="a-piano-gs"/></Track>
<Track id="1"><Name>Guitar 7</Name><Instrument ref="e-gtr7"/><Properties><Property name="Tuning"><Pitches>35 40 45 50 55 59 64</Pitches></Property></Properties></Track>
<Track id="2"><Name>Drums</Name><Instrument ref="drmkt"/><InstrumentSet><Type>drumKit</Type></InstrumentSet></Track>
</Tracks>
<MasterBars><MasterBar><Time>4/4</Time><Bars>0 1 2 3</Bars></MasterBar></MasterBars>
<Bars><Bar id="0"><Voices>0 -1 -1 -1</Voices></Bar><Bar id="1"><Voices>1 -1 -1 -1</Voices></Bar>
<Bar id="2"><Voices>2 -1 -1 -1</Voices></Bar><Bar id="3"><Voices>3 -1 -1 -1</Voices></Bar></Bars>
<Voices><Voice id="0"><Beats>0</Beats></Voice><Voice id="1"><Beats>1</Beats></Voice><Voice id="2"><Beats>2</Beats></Voice><Voice id="3"><Beats>3</Beats></Voice></Voices>
<Beats><Beat id="0"><Rhythm ref="0"/></Beat><Beat id="1"><Rhythm ref="0"/></Beat><Beat id="2"><Rhythm ref="0"/><Notes>0</Notes></Beat><Beat id="3"><Rhythm ref="0"/><Notes>1 2</Notes></Beat></Beats>
<Notes>{_note(0, 0, 3)}
{_note(1, 0, 36, '<Property name="Midi"><Number>36</Number></Property>')}
{_note(2, 1, 42, '<Property name="Midi"><Number>42</Number></Property>')}</Notes>
<Rhythms><Rhythm id="0"><NoteValue>Whole</NoteValue></Rhythm></Rhythms></GPIF>"""


# ---------------------------------------------------------------- helpers: fake GP6 containers

def _bcfs(name: str, payload: bytes) -> bytes:
    """An uncompressed GP6 file system ("BCFS") holding one file."""
    blocks = [payload[i:i + 0x1000] for i in range(0, len(payload), 0x1000)] or [b""]
    body = bytearray(0x1000 * (2 + len(blocks)))                # sector 0: header, sector 1: directory entry
    struct.pack_into("<i", body, 0x1000, 2)
    body[0x1004:0x1004 + len(name)] = name.encode()
    struct.pack_into("<i", body, 0x1000 + 0x8C, len(payload))
    for k, block in enumerate(blocks):
        struct.pack_into("<i", body, 0x1000 + 0x94 + 4 * k, 2 + k)
        body[0x1000 * (2 + k):0x1000 * (2 + k) + len(block)] = block
    return b"BCFS" + bytes(body)


def _bcfz(data: bytes) -> bytes:
    """The BCFZ wrapper with literal runs only (valid, just not small)."""
    bits: list[int] = []
    for i in range(0, len(data), 3):
        chunk = data[i:i + 3]
        bits += [0, len(chunk) & 1, len(chunk) >> 1]               # flag "literal", 2-bit size (low bit first)
        for byte in chunk:
            bits += [(byte >> (7 - k)) & 1 for k in range(8)]
    bits += [0] * (-len(bits) % 8)
    packed = bytes(int("".join(map(str, bits[k:k + 8])), 2) for k in range(0, len(bits), 8))
    return b"BCFZ" + struct.pack("<i", len(data)) + packed


def _write_gp(path: Path, gpif: str = GPIF) -> Path:
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("Content/score.gpif", gpif)
    return path


def _write_gpx(path: Path, gpif: str = GPIF) -> Path:
    path.write_bytes(_bcfz(_bcfs("score.gpif", gpif.encode())))
    return path


def _signature(song):
    """Everything about a song that conversion must keep, as plain data."""
    return {
        "meta": [song.title, song.subtitle, song.artist, song.album, song.words, song.music, song.copyright, song.tab, song.tempo, song.key],
        "bars": [(h.timeSignature.numerator, h.timeSignature.denominator.value, h.isRepeatOpen, h.repeatClose, h.repeatAlternative,
                  h.marker.title if h.marker else None) for h in song.measureHeaders],
        "tracks": [(t.name, [(s.number, s.value) for s in t.strings], t.offset, t.isPercussionTrack, t.channel.instrument,
                    [[[(b.start, b.duration.value, b.duration.isDotted, b.status.value,
                        sorted((n.string, n.value, n.type.value, n.velocity, n.effect.palmMute, n.effect.hammer, n.effect.vibrato,
                                n.effect.letRing, n.effect.staccato, n.effect.ghostNote, bool(n.effect.bend), bool(n.effect.harmonic),
                                tuple(x.value for x in n.effect.slides)) for n in b.notes))
                       for b in v.beats] for v in m.voices] for m in t.measures]) for t in song.tracks],
    }


# ---------------------------------------------------------------- format detection

def test_sniff_finds_the_format_from_the_bytes(tmp_path):
    gp3 = tmp_path / "really_gp3.gp5"          # wrong extension on purpose
    gp3.write_bytes(FIXTURE.read_bytes())
    assert sniff(gp3) == "gp3"
    html = tmp_path / "page.gp4"
    html.write_text("<!DOCTYPE HTML><html>not found</html>")
    assert sniff(html) == "html"
    old = tmp_path / "old.gtp"
    old.write_bytes(bytes([25]) + b"FICHIER GUITARE PRO v1.04" + b"\0" * 40)
    assert sniff(old) == "gp12"
    assert sniff(_write_gp(tmp_path / "a.gp")) == "gpif"
    assert sniff(_write_gpx(tmp_path / "a.gpx")) == "gpif"
    (tmp_path / "empty.gp3").write_bytes(b"")
    assert sniff(tmp_path / "empty.gp3") == "empty"


# ---------------------------------------------------------------- GP3 / GP4 / .gtp: nothing may change

def test_gp3_becomes_gp5_and_keeps_everything(tmp_path):
    dst = tmp_path / "out.gp5"
    status, note = convert(FIXTURE, dst, "gp3")
    assert status == "ok", note
    old, new = gp.parse(str(FIXTURE)), gp.parse(str(dst))
    assert new.versionTuple[0] == 5
    assert _signature(new) == _signature(old)


def test_gp4_becomes_gp5_and_keeps_everything(tmp_path):
    src = tmp_path / "song.gp4"
    gp.write(gp.parse(str(FIXTURE)), str(src), version=(4, 0, 0))
    assert sniff(src) == "gp4"
    dst = tmp_path / "song.gp5"
    status, note = convert(src, dst, "gp4")
    assert status == "ok", note
    assert gp.parse(str(dst)).versionTuple[0] == 5
    assert _signature(gp.parse(str(dst))) == _signature(gp.parse(str(src)))


def test_gtp_with_gp3_content_is_converted_by_its_content(tmp_path):
    src = tmp_path / "old_name.gtp"
    shutil.copy(FIXTURE, src)
    kind = sniff(src)
    assert kind == "gp3"
    dst = tmp_path / "old_name.gp5"
    assert convert(src, dst, kind)[0] == "ok"
    assert _signature(gp.parse(str(dst))) == _signature(gp.parse(str(FIXTURE)))


# ---------------------------------------------------------------- GP6 / GP7

def test_gp7_file_becomes_gp5(tmp_path):
    src = _write_gp(tmp_path / "tiny.gp")
    dst = tmp_path / "tiny.gp5"
    status, note = convert(src, dst, "gpif")
    assert status == "ok", note
    song = gp.parse(str(dst))
    assert (song.title, song.artist, song.album, song.tempo) == ("Tiny", "Nobody", "Small", 96)
    assert [h.timeSignature.numerator for h in song.measureHeaders] == [4, 3]
    assert [s.value for s in song.tracks[0].strings] == [64, 59, 55, 50, 45, 38]     # drop D, high string first
    assert song.tracks[0].channel.instrument == 29
    first = song.tracks[0].measures[0].voices[0].beats
    assert sorted((n.string, n.value) for n in first[0].notes) == [(5, 2), (6, 0)]          # GPIF string 0 is the lowest
    assert first[1].status == gp.BeatStatus.rest and first[1].notes == []
    assert first[2].notes[0].effect.palmMute and first[2].duration.value == 2
    last = song.tracks[0].measures[1].voices[0].beats[0]
    assert last.duration.value == 2 and last.duration.isDotted and last.notes[0].value == 7


def test_gpx_file_becomes_gp5(tmp_path):
    """A GP6 file: BCFZ-compressed file system around the same XML."""
    src = _write_gpx(tmp_path / "tiny.gpx")
    assert sniff(src) == "gpif"
    dst = tmp_path / "tiny.gp5"
    status, note = convert(src, dst, "gpif")
    assert status == "ok", note
    via_gp = tmp_path / "same.gp5"
    assert convert(_write_gp(tmp_path / "same.gp"), via_gp, "gpif")[0] == "ok"
    assert _signature(gp.parse(str(dst))) == _signature(gp.parse(str(via_gp)))          # GP6 and GP7 give the same song
    assert gp.parse(str(dst)).title == "Tiny"


def test_gp6_grand_staff_seven_strings_and_drums(tmp_path):
    src = _write_gpx(tmp_path / "six.gpx", GPIF6)
    dst = tmp_path / "six.gp5"
    status, note = convert(src, dst, "gpif")
    assert status == "ok", note
    song = gp.parse(str(dst))
    names = [t.name for t in song.tracks]
    assert names == ["Piano", "Piano (2)", "Guitar 7", "Drums"]                  # the grand staff takes two bars per master bar
    guitar = song.tracks[2]
    assert [s.value for s in guitar.strings] == [64, 59, 55, 50, 45, 40, 35]    # tuning read from the track, 7 strings
    (note_,) = guitar.measures[0].voices[0].beats[0].notes
    assert (note_.string, note_.value) == (7, 3)                                # lowest string, fret 3: not shifted by the piano
    drums = song.tracks[3]
    assert drums.isPercussionTrack
    hits = drums.measures[0].voices[0].beats[0].notes
    assert sorted(n.value for n in hits) == [36, 42] and len({n.string for n in hits}) == 2    # both hits kept


# ---------------------------------------------------------------- unreadable files go to __erroneous

def _library(root: Path) -> Path:
    folder = root / "tabs - acoustic"
    folder.mkdir(parents=True)
    shutil.copy(FIXTURE, folder / "Good - Song.gp3")
    _write_gp(folder / "Good - Seven.gp")
    (folder / "Bad - Web page.gp4").write_text("<!DOCTYPE HTML><html>404</html>")
    (folder / "Bad - Empty.gp3").write_bytes(b"")
    (folder / "Bad - Cut off.gp3").write_bytes(FIXTURE.read_bytes()[:300])
    (folder / "Bad - Broken zip.gp").write_bytes(b"PK\x03\x04 this is not a zip")
    (folder / "Old - Guitar Pro 1.gtp").write_bytes(bytes([25]) + b"FICHIER GUITARE PRO v1.04" + b"\0" * 400)
    return folder


def test_unreadable_files_are_moved_to_erroneous_next_to_where_they_were(tmp_path):
    folder = _library(tmp_path / "lib")
    out = tmp_path / "out"
    assert main([str(tmp_path / "lib"), "--out", str(out), "--workers", "1"]) == 0
    bad = folder / "__erroneous"
    assert sorted(p.name for p in bad.iterdir()) == ["Bad - Broken zip.gp", "Bad - Cut off.gp3", "Bad - Empty.gp3",
                                                    "Bad - Web page.gp4", "Old - Guitar Pro 1.gtp"]
    assert sorted(p.name for p in folder.iterdir() if p.is_file()) == ["Good - Seven.gp", "Good - Song.gp3"]     # originals untouched
    converted = sorted(p.name for p in (out / "tabs - acoustic").iterdir())
    assert converted == ["Good - Seven.gp5", "Good - Song.gp5"]
    for p in (out / "tabs - acoustic").iterdir():
        assert gp.parse(str(p)).tracks                                                  # every output reads back
    # a second run finds nothing left to do and does not look inside __erroneous
    assert main([str(tmp_path / "lib"), "--out", str(out), "--workers", "1"]) == 0
    assert len(list(bad.iterdir())) == 5


def test_options_leave_files_in_place(tmp_path):
    folder = _library(tmp_path / "lib")
    assert main([str(tmp_path / "lib"), "--out", str(tmp_path / "out"), "--workers", "1", "--keep-old-gtp"]) == 0
    assert (folder / "Old - Guitar Pro 1.gtp").exists()                               # valid file we cannot read: stays
    assert not (folder / "__erroneous" / "Old - Guitar Pro 1.gtp").exists() and (folder / "__erroneous" / "Bad - Empty.gp3").exists()
    folder2 = _library(tmp_path / "lib2")
    assert main([str(tmp_path / "lib2"), "--out", str(tmp_path / "out2"), "--workers", "1", "--no-quarantine"]) == 0
    assert not (folder2 / "__erroneous").exists() and (folder2 / "Bad - Empty.gp3").exists()


# ---------------------------------------------------------------- 8-string guitars go to __8string

def _eight(kind: str) -> str:
    """The tiny song with an 8-string tuning on an instrument of the given family (electricGuitar, acousticPiano ...)."""
    return (GPIF.replace("<Pitches>38 45 50 55 59 64</Pitches>", "<Pitches>30 35 40 45 50 55 59 64</Pitches>")
                .replace("<Name>Guitar</Name>", f"<Name>Guitar</Name><InstrumentSet><Type>{kind}</Type></InstrumentSet>"))


def test_eight_string_guitar_files_are_moved_to_their_own_folder(tmp_path):
    folder = tmp_path / "lib" / "tabs - metal"
    folder.mkdir(parents=True)
    _write_gp(folder / "Band - Eight String.gp", _eight("electricGuitar"))
    _write_gp(folder / "Band - Eight Bass.gp", _eight("electricBass"))
    _write_gp(folder / "Band - Piano.gp", _eight("acousticPiano"))           # a piano's default tuning has 8 entries: not a guitar
    _write_gp(folder / "Band - Synth Bass.gp", _eight("bassSynthesizer"))
    _write_gp(folder / "Band - Seven.gp")                                     # 6 strings
    assert main([str(tmp_path / "lib"), "--out", str(tmp_path / "out"), "--workers", "1"]) == 0
    assert sorted(p.name for p in (folder / "__8string").iterdir()) == ["Band - Eight Bass.gp", "Band - Eight String.gp"]
    assert sorted(p.name for p in folder.iterdir() if p.is_file()) == ["Band - Piano.gp", "Band - Seven.gp", "Band - Synth Bass.gp"]
    assert sorted(p.name for p in (tmp_path / "out" / "tabs - metal").iterdir()) == ["Band - Piano.gp5", "Band - Seven.gp5", "Band - Synth Bass.gp5"]
    assert not (folder / "__erroneous").exists()
    # the moved files are not looked at again
    assert main([str(tmp_path / "lib"), "--out", str(tmp_path / "out"), "--workers", "1"]) == 0
    assert len(list((folder / "__8string").iterdir())) == 2


def test_convert_8string_option_converts_them_anyway(tmp_path):
    folder = tmp_path / "lib" / "tabs - metal"
    folder.mkdir(parents=True)
    _write_gp(folder / "Band - Eight String.gp", _eight("electricGuitar"))
    assert main([str(tmp_path / "lib"), "--out", str(tmp_path / "out"), "--workers", "1", "--convert-8string"]) == 0
    assert not (folder / "__8string").exists() and (folder / "Band - Eight String.gp").exists()
    song = gp.parse(str(tmp_path / "out" / "tabs - metal" / "Band - Eight String.gp5"))
    assert len(song.tracks[0].strings) == 7                                         # GP5 holds seven: the lowest string is dropped
