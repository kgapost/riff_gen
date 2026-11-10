"""Convert Guitar Pro files of any version to Guitar Pro 5 (the format the RiffGen pipeline reads).

One self-contained file: it needs only Python 3.10+ and PyGuitarPro (``pip install PyGuitarPro``).
Put it next to the tab folders (data/) and run it from anywhere:

    python data_convert.py --dry-run          # count what is in the tabs* folders next to this file
    python data_convert.py --limit 200        # a first try
    python data_convert.py                    # data/tabs* -> data/gp5/<same folders>/<name>.gp5
    python data_convert.py some/folder --out elsewhere

What it reads (the format is found from the file's first bytes, never from its extension, so a
.gp5 that is really a .gpx, or a .gtp that is really GP3, is handled):

* GP3, GP4, GP5 (also named .gtp/.gp): read and rewritten with PyGuitarPro.
* GP6 (.gpx) and GP7/GP8 (.gp): read by the GPIF reader below (PyGuitarPro cannot) and written as GP5.
  A .gp file is a zip with Content/score.gpif; a .gpx file is a BCFZ-compressed "BCFS" mini file
  system holding the same XML document. What is carried over: title/artist/album, tempo and tempo
  changes, time and key signatures, repeats and alternative endings, section markers, tracks with
  name, tuning, capo and MIDI program, every note with string, fret, velocity, ties, dead notes, palm
  mutes, hammer-ons, slides, harmonics, bends, vibrato, let-ring, staccato/accents, ghost notes, brush
  strokes, slap/pop/tap and grace notes. Drum tracks keep their MIDI notes. Dropped: chord diagrams,
  lyrics, tremolo bar, trills, tremolo picking, fingerings, voices 3 and 4.
  Limits: an 8-string track keeps its 7 highest strings (a note on the lowest one moves to the lowest kept
  string when the pitch allows, otherwise it is lost); notes of a drum track beyond 7 at once are lost.
* GP1/GP2 (``FICHIER GUITARE PRO v1.x``, ``FICHIER GUITAR PRO v2.xx``): valid files, but this tool cannot read the format.
* HTML pages saved as tabs, empty or damaged files, GP7/GP8 files locked for editing (their score is encrypted):
  "not a Guitar Pro file" / "source cannot be read".

Files that cannot be read are moved to ``__erroneous`` inside the folder they were in
(``tabs - acoustic/x.gp3`` -> ``tabs - acoustic/__erroneous/x.gp3``) and are not looked at again.
``--no-quarantine`` leaves them where they are; ``--keep-old-gtp`` leaves only the GP1/GP2 files
(they are fine, just unreadable here).

Songs with a guitar or bass track of 8 or more strings (GP6/GP7 files only; a GP5 file holds 7 strings) are
moved to ``__8string`` inside their folder instead of being converted. A piano or synth track that merely
carries an 8-entry default tuning does not count. ``--convert-8string`` converts them anyway (the lowest
string is dropped). Found while converting, so ``--dry-run`` does not list them.

The originals are never changed (apart from those moves). Every output is read back and must have the
same number of tracks and bars as the source, otherwise it is deleted and the file is reported as failed.
Output goes to ``<out>/<folder of the source>/<name>.gp5``; two sources with one name get ``(from gpx)`` etc.
Existing outputs are skipped, so a stopped run continues. ``<out>/conversion.tsv`` lists every file.
Already-GP5 files are skipped unless ``--copy-gp5`` is given (then they are copied unchanged).
Several processes convert at once (``--workers``, default: all cores but one).
"""

from __future__ import annotations

import argparse
import io
import multiprocessing as mp
import os
import re
import shutil
import struct
import sys
import time
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from pathlib import Path

import guitarpro as gp

HERE = Path(__file__).resolve().parent / "data"          # the collection: data/tabs - */

# =========================================================================== GP6 / GP7 reader

NOTE_VALUES = {"long": 1, "whole": 1, "half": 2, "quarter": 4, "eighth": 8, "16th": 16, "32nd": 32,
               "64th": 64, "128th": 64, "256th": 64}
DYNAMICS = {"ppp": 15, "pp": 31, "p": 47, "mp": 63, "mf": 79, "f": 95, "ff": 111, "fff": 127}


_BAD_XML = re.compile("[" + "".join(chr(c) for c in (*range(0, 9), 11, 12, *range(14, 32))) + "]")   # control characters XML forbids


class GpifError(Exception):
    pass


# --------------------------------------------------------------------------- containers

class _Bits:
    """Bit reader of the BCFZ stream: ``bits`` reads most significant bit first, ``bits_le`` the other way round."""

    def __init__(self, data: bytes, pos: int):
        self.data, self.pos, self.bit = data, pos, 8

    def one(self) -> int:
        if self.bit == 8:
            if self.pos >= len(self.data):
                raise EOFError
            self.cur, self.pos, self.bit = self.data[self.pos], self.pos + 1, 0
        v = (self.cur >> (7 - self.bit)) & 1
        self.bit += 1
        return v

    def bits(self, n: int) -> int:
        v = 0
        for _ in range(n):
            v = (v << 1) | self.one()
        return v

    def bits_le(self, n: int) -> int:
        v = 0
        for i in range(n):
            v |= self.one() << i
        return v


def _bcfz(data: bytes) -> bytes:
    expected = struct.unpack_from("<i", data, 4)[0]
    src = _Bits(data, 8)
    out = bytearray()
    try:
        while len(out) < expected:
            if src.one():
                width = src.bits(4)
                offset = src.bits_le(width)
                size = src.bits_le(width)
                start = len(out) - offset
                if start < 0:
                    raise GpifError("corrupt BCFZ stream")
                out += out[start:start + min(offset, size)]
            else:
                for _ in range(src.bits_le(2)):
                    out.append(src.bits(8))
    except EOFError:
        pass
    return bytes(out)


def _bcfs(fs: bytes) -> dict[str, bytes]:
    fs = fs[4:]                       # sectors are counted from after the "BCFS" tag
    files: dict[str, bytes] = {}
    offset = 0x1000
    while offset + 0x98 <= len(fs):
        if struct.unpack_from("<i", fs, offset)[0] == 2:
            name = fs[offset + 4:offset + 4 + 127].split(b"\0", 1)[0].decode("latin-1")
            size = struct.unpack_from("<i", fs, offset + 0x8C)[0]
            blocks, i = [], 0
            while True:
                block = struct.unpack_from("<i", fs, offset + 0x94 + 4 * i)[0]
                if block == 0 or 0x94 + 4 * i >= 0x1000 - 4:
                    break
                blocks.append(fs[block * 0x1000:(block + 1) * 0x1000])
                i += 1
            files[name] = b"".join(blocks)[:size]
        offset += 0x1000
    return files


def read_gpif(path: str | Path) -> bytes:
    """The score.gpif document of a .gp or .gpx file."""
    data = Path(path).read_bytes()
    if data[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            name = next((n for n in z.namelist() if n.lower().endswith("score.gpif")), None)
            if name is None:
                raise GpifError("no score.gpif in the zip")
            return z.read(name)
    if data[:4] == b"BCFZ":
        data = _bcfz(data)
    if data[:4] != b"BCFS":
        raise GpifError(f"not a GP6/GP7 file (starts with {data[:4]!r})")
    files = _bcfs(data)
    if "score.gpif" not in files:
        raise GpifError(f"no score.gpif inside the GP6 container (has {sorted(files)})")
    return files["score.gpif"]


# --------------------------------------------------------------------------- the XML

def _text(el, tag: str, default: str = "") -> str:
    child = el.find(tag)
    return (child.text or "").strip() if child is not None else default


def _ints(el, tag: str) -> list[int]:
    return [int(x) for x in _text(el, tag).split()]


def _props(el) -> dict[str, ET.Element]:
    return {p.get("name"): p for p in el.findall("Properties/Property")}


def _flag(props: dict, name: str) -> bool:
    p = props.get(name)
    return p is not None and (p.find("Enable") is not None or _text(p, "Flags") not in ("", "0"))


def load_gpif(path: str | Path) -> gp.Song:
    """Parse a .gp/.gpx file into a PyGuitarPro song."""
    raw = read_gpif(path)
    # some files carry stray control characters or broken UTF-8 in a text field
    raw = _BAD_XML.sub("", raw.decode("utf-8", "replace")).encode("utf-8")
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as e:
        raise GpifError(f"score.gpif is not valid XML: {e}") from e
    return _Builder(root).song()


class _Builder:
    def __init__(self, root: ET.Element):
        self.root = root
        index = lambda tag, path: {e.get("id"): e for e in root.findall(path)}  # noqa: E731
        self.bars = index("Bar", "Bars/Bar")
        self.voices = index("Voice", "Voices/Voice")
        self.beats = index("Beat", "Beats/Beat")
        self.notes = index("Note", "Notes/Note")
        self.rhythms = index("Rhythm", "Rhythms/Rhythm")
        self.fixes = 0   # things that could not be represented exactly

    # ---- song ----
    def song(self) -> gp.Song:
        root = self.root
        score = root.find("Score")
        song = gp.Song()
        song.measureHeaders, song.tracks = [], []
        if score is not None:
            for field, tag in (("title", "Title"), ("subtitle", "SubTitle"), ("artist", "Artist"), ("album", "Album"),
                               ("words", "Words"), ("music", "Music"), ("copyright", "Copyright"),
                               ("tab", "Tabber"), ("instructions", "Instructions")):
                setattr(song, field, _text(score, tag))
            notices = _text(score, "Notices")
            song.notice = notices.splitlines() if notices else []

        tempos = self._tempos()
        master_bars = root.findall("MasterBars/MasterBar")
        if not master_bars:
            raise GpifError("the file has no bars")
        song.tempo = int(tempos.get(0, 120))
        song.hideTempo = False

        start, prev_tempo = gp.Duration.quarterTime, None
        for number, mb in enumerate(master_bars, 1):
            h = gp.MeasureHeader(number=number, start=start)
            num, _, den = _text(mb, "Time", "4/4").partition("/")
            h.timeSignature = gp.TimeSignature(int(num), gp.Duration(value=int(den or 4)))
            key = mb.find("Key")
            if key is not None:
                count = int(_text(key, "AccidentalCount", "0") or 0)
                minor = _text(key, "Mode").lower() == "minor"
                try:
                    h.keySignature = gp.KeySignature((max(-7, min(7, count)), int(minor)))
                except ValueError:
                    pass
            rep = mb.find("Repeat")
            if rep is not None:
                h.isRepeatOpen = rep.get("start") == "true"
                if rep.get("end") == "true":
                    h.repeatClose = max(int(rep.get("count") or 2) - 1, 1)
            endings = _ints(mb, "AlternateEndings")
            if endings:
                h.repeatAlternative = sum(1 << (e - 1) for e in endings if 1 <= e <= 8)
            section = mb.find("Section")
            if section is not None:
                title = _text(section, "Text") or _text(section, "Letter")
                if title:
                    h.marker = gp.Marker(title=title)
            h.hasDoubleBar = _text(mb, "DoubleBar") == "true"
            tempo = tempos.get(number - 1, prev_tempo if prev_tempo is not None else song.tempo)
            h.tempo = gp.Tempo(int(round(tempo)))
            prev_tempo = tempo
            song.addMeasureHeader(h)
            start += h.length

        # one PyGuitarPro track per staff (a GP7 piano track has two)
        slots: list[tuple[ET.Element, ET.Element]] = []
        for track in root.findall("Tracks/Track"):
            staves = track.findall("Staves/Staff")
            if not staves:       # GP6 lists no staves: a grand-staff instrument ("a-piano-gs") has two, every other one
                ref = track.find("Instrument").get("ref") if track.find("Instrument") is not None else ""
                staves = [ET.Element("Staff") for _ in range(2 if (ref or "").endswith("-gs") else 1)]
            for staff in staves:
                slots.append((track, staff))
        for n, (track, staff) in enumerate(slots, 1):
            song.tracks.append(self._track(song, track, staff, n, sum(1 for t, _ in slots[:n - 1] if t is track)))

        for m, mb in enumerate(master_bars):
            ids = _text(mb, "Bars").split()
            for t, trk in enumerate(song.tracks):
                measure = trk.measures[m]
                bar = self.bars.get(ids[t]) if t < len(ids) else None
                if bar is not None:
                    self._fill_measure(measure, bar, trk)

        # a tempo change becomes a mix-table change on the first beat of its bar (what a GP5 file has)
        last = song.tempo
        for header in song.measureHeaders:
            if header.tempo.value != last and song.tracks:
                beats = song.tracks[0].measures[header.number - 1].voices[0].beats
                if beats:
                    beats[0].effect.mixTableChange = gp.MixTableChange(tempo=gp.MixTableItem(header.tempo.value, 0, True),
                                                                      tempoName="", hideTempo=True)
            last = header.tempo.value
        song.eight_string = [t.name for t in song.tracks if t._eight]
        return song

    def _tempos(self) -> dict[int, float]:
        out: dict[int, float] = {}
        for a in self.root.findall("MasterTrack/Automations/Automation"):
            if _text(a, "Type").lower() != "tempo":
                continue
            value = _text(a, "Value").split()
            if not value:
                continue
            bpm = float(value[0])
            unit = int(value[1]) if len(value) > 1 else 2
            bpm *= {1: 0.5, 2: 1, 3: 2, 4: 1.5, 5: 3}.get(unit, 1)   # eighth, quarter, half, dotted quarter, dotted half
            out.setdefault(int(_text(a, "Bar", "0") or 0), bpm)
        return out

    # ---- tracks ----
    def _track(self, song: gp.Song, el: ET.Element, staff: ET.Element, number: int, nth: int) -> gp.Track:
        name = _text(el, "Name") or f"Track {number}"
        if nth:
            name += f" ({nth + 1})"
        props = {**_props(el), **_props(staff)}       # GP7 keeps tuning and capo on the staff, GP6 on the track
        pitches: list[int] = []
        tuning = props.get("Tuning")
        if tuning is not None:
            pitches = [int(x) for x in _text(tuning, "Pitches").split()]
        drums = _text(el, "InstrumentSet/Type").lower() in ("drumkit", "drums")
        if drums:
            pitches = [0] * 7                          # a GP5 drum track: fret = MIDI note, up to 7 hits at once
        elif not pitches:
            pitches = [40, 45, 50, 55, 59, 64]
        dropped = max(0, len(pitches) - MAX_STRINGS)             # GP5 has at most 7 strings: the lowest ones of an 8-string go
        kept = pitches[dropped:]
        n = len(kept)
        strings = [gp.GuitarString(k + 1, kept[n - 1 - k]) for k in range(n)]    # string 1 = last in the GPIF list
        trk = gp.Track(song, number=number, name=name[:40], strings=strings, measures=[])
        trk.measures = [gp.Measure(trk, h) for h in song.measureHeaders]
        trk.isPercussionTrack = drums
        capo = props.get("CapoFret")
        if capo is not None:
            trk.offset = int(_text(capo, "Fret", "0") or 0)
        midi = el.find("GeneralMidi")
        program = int(_text(midi, "Program", "25") or 25) if midi is not None else 25
        channel = 9 if drums else (2 * (number - 1)) % 16
        if channel in (9, 8) and not drums:
            channel = (channel + 2) % 16
        trk.channel = gp.MidiChannel(channel=channel, effectChannel=channel + 1 if channel < 15 else 14,
                                     instrument=program)
        trk._elements = self._drum_elements(el) if drums else {}
        kind = f"{_text(el, 'InstrumentSet/Type')} {el.find('Instrument').get('ref') if el.find('Instrument') is not None else ''}".lower()
        string_instrument = (not drums and any(w in kind for w in ("guitar", "gtr", "bass")) and not any(w in kind for w in ("synth", "snt")))
        trk._eight = string_instrument and len(pitches) >= 8     # (a piano or synth track often carries an 8-string default tuning)
        trk._pitches, trk._dropped = pitches, dropped
        return trk

    @staticmethod
    def _drum_elements(el: ET.Element) -> dict[tuple[int, int], int]:
        out = {}
        for e, element in enumerate(el.findall("InstrumentSet/Elements/Element")):
            for v, art in enumerate(element.findall("Articulations/Articulation")):
                num = _text(art, "OutputMidiNumber")
                if num:
                    out[(e, v)] = int(num)
        return out

    # ---- bars, beats, notes ----
    def _fill_measure(self, measure: gp.Measure, bar: ET.Element, trk: gp.Track) -> None:
        clef = _text(bar, "Clef")
        if clef.startswith("F"):
            measure.clef = gp.MeasureClef.bass
        voice_ids = [v for v in _text(bar, "Voices").split() if v != "-1"][:gp.Measure.maxVoices]
        carry_grace: list[dict] = []
        for v, vid in enumerate(voice_ids):
            voice = measure.voices[v]
            start = measure.header.start
            for bid in _text(self.voices[vid], "Beats").split():
                el = self.beats[bid]
                beat = self._beat(voice, el, trk, start, carry_grace)
                if beat is None:      # a grace beat: its note moved onto the next one
                    continue
                voice.beats.append(beat)
                start += beat.duration.time

    def _duration(self, el: ET.Element) -> gp.Duration:
        rhythm = self.rhythms[el.find("Rhythm").get("ref")]
        value = NOTE_VALUES.get(_text(rhythm, "NoteValue").lower(), 4)
        dots = rhythm.find("AugmentationDot")
        tup = rhythm.find("PrimaryTuplet")
        tuplet = gp.Tuplet(int(tup.get("num")), int(tup.get("den"))) if tup is not None else gp.Tuplet()
        if not tuplet.isSupported():
            self.fixes += 1
            tuplet = gp.Tuplet()
        return gp.Duration(value=value, isDotted=dots is not None and int(dots.get("count", 1)) >= 1, tuplet=tuplet)

    def _beat(self, voice: gp.Voice, el: ET.Element, trk: gp.Track, start: int, carry: list[dict]) -> gp.Beat | None:
        beat = gp.Beat(voice, duration=self._duration(el), start=start)
        beat.status = gp.BeatStatus.rest
        props = _props(el)
        velocity = DYNAMICS.get(_text(el, "Dynamic").lower(), gp.Velocities.default)
        note_ids = _text(el, "Notes").split()
        for nid in note_ids:
            note = self._note(beat, self.notes[nid], trk, velocity)
            if note is not None:
                beat.notes.append(note)
        if trk.isPercussionTrack:
            # GP5 stores a drum hit as a fret = MIDI note on some string; simultaneous hits need different strings
            seen_midi: set[int] = set()
            beat.notes = [n for n in beat.notes if not (n.value in seen_midi or seen_midi.add(n.value))][:len(trk.strings)]
            for k, n in enumerate(beat.notes):
                n.string = k + 1
        else:
            seen: set[int] = set()   # a GP5 beat holds one note per string
            beat.notes = [n for n in beat.notes if not (n.string in seen or seen.add(n.string))]
        if beat.notes:
            beat.status = gp.BeatStatus.normal
        if el.find("GraceNotes") is not None:      # becomes a grace effect of the next beat's first note
            if beat.notes:
                carry.append({"fret": beat.notes[0].value, "dead": beat.notes[0].type == gp.NoteType.dead,
                              "on_beat": _text(el, "GraceNotes") == "OnBeat"})
            return None
        if carry and beat.notes:
            g = carry.pop()
            carry.clear()
            beat.notes[0].effect.grace = gp.GraceEffect(fret=max(g["fret"], 0), isDead=g["dead"], isOnBeat=g["on_beat"],
                                                        duration=32, velocity=velocity)
        brush = props.get("Brush")
        if brush is not None:
            beat.effect.stroke = gp.BeatStroke(
                gp.BeatStrokeDirection.up if _text(brush, "Direction") == "Up" else gp.BeatStrokeDirection.down, 8)
        if _flag(props, "Slapped"):
            beat.effect.slapEffect = gp.SlapEffect.slapping
        elif _flag(props, "Popped"):
            beat.effect.slapEffect = gp.SlapEffect.popping
        if _text(el, "Fadding").lower() == "fadein":
            beat.effect.fadeIn = True
        text = _text(el, "FreeText")
        if text:
            beat.text = gp.models.BeatText(text)
        return beat

    def _note(self, beat: gp.Beat, el: ET.Element, trk: gp.Track, velocity: int) -> gp.Note | None:
        props = _props(el)
        note = gp.Note(beat, velocity=velocity)
        note.type = gp.NoteType.normal
        n_strings = len(trk.strings)
        if trk.isPercussionTrack:
            midi = None
            midi_prop = props.get("Midi")                       # GP7: the sounding MIDI note is stored on the note
            if midi_prop is not None and _text(midi_prop, "Number").lstrip("-").isdigit():
                midi = int(_text(midi_prop, "Number"))
            element, variation = props.get("Element"), props.get("Variation")
            if midi is None and element is not None:             # GP6: element + variation index into the kit
                midi = trk._elements.get((int(_text(element, "Element", "0") or 0),
                                          int(_text(variation, "Variation", "0") or 0) if variation is not None else 0))
            if midi is None or midi < 0:
                return None
            note.value = midi
            note.string = 1          # (the next free string is given in _beat: simultaneous drums need distinct strings)
            return note
        string = props.get("String")
        fret = props.get("Fret")
        if string is None or fret is None:
            return None
        s = int(_text(string, "String", "0") or 0)
        fret_no = max(int(_text(fret, "Fret", "0") or 0), 0)
        if s < trk._dropped:                         # a note on a string GP5 cannot hold: same pitch on the lowest kept string
            fret_no = trk._pitches[s] + fret_no - trk._pitches[trk._dropped]
            if fret_no < 0:
                self.fixes += 1
                return None
            s = trk._dropped
        s -= trk._dropped
        note.string = n_strings - s if 0 <= s < n_strings else 1
        note.value = fret_no
        tie = el.find("Tie")
        if tie is not None and tie.get("destination") == "true":
            note.type = gp.NoteType.tie
        if _flag(props, "Muted"):
            note.type = gp.NoteType.dead
        eff = note.effect
        eff.palmMute = _flag(props, "PalmMuted")
        eff.hammer = _flag(props, "HopoOrigin")
        eff.letRing = el.find("LetRing") is not None
        eff.ghostNote = _text(el, "AntiAccent") == "Normal"
        vib = _text(el, "Vibrato")
        eff.vibrato = vib in ("Slight", "Wide")
        accent = int(_text(el, "Accent", "0") or 0)
        eff.staccato = bool(accent & 1)
        eff.heavyAccentuatedNote = bool(accent & 4)
        eff.accentuatedNote = bool(accent & 8)
        slide = props.get("Slide")
        if slide is not None:
            flags = int(_text(slide, "Flags", "0") or 0)
            for bit, kind in ((1, gp.SlideType.shiftSlideTo), (2, gp.SlideType.legatoSlideTo), (4, gp.SlideType.outDownwards),
                              (8, gp.SlideType.outUpwards), (16, gp.SlideType.intoFromBelow), (32, gp.SlideType.intoFromAbove)):
                if flags & bit:
                    eff.slides.append(kind)
        if _flag(props, "Tapped"):
            beat.effect.slapEffect = gp.SlapEffect.tapping
        htype = props.get("HarmonicType")
        if htype is not None:
            kind = _text(htype, "HType")
            if kind == "Natural":
                eff.harmonic = gp.NaturalHarmonic()
            elif kind == "Tap":
                hfret = props.get("HarmonicFret")
                eff.harmonic = gp.TappedHarmonic(fret=int(float(_text(hfret, "HFret", "0") or 0)) if hfret is not None else 0)
            elif kind == "Semi":
                eff.harmonic = gp.SemiHarmonic()
            elif kind:
                eff.harmonic = gp.PinchHarmonic()
        if _flag(props, "Bended"):
            eff.bend = self._bend(props)
        return note

    @staticmethod
    def _bend(props: dict) -> gp.BendEffect | None:
        def val(name: str, default: float | None = None) -> float | None:
            p = props.get(name)
            return float(_text(p, "Float")) if p is not None and _text(p, "Float") else default

        origin = val("BendOriginValue", 0.0)
        dest = val("BendDestinationValue", 0.0)
        middle = val("BendMiddleValue")
        pts = [(val("BendOriginOffset", 0.0), origin)]
        if middle is not None:
            pts.append((val("BendMiddleOffset1", 50.0), middle))
            if val("BendMiddleOffset2") is not None and val("BendMiddleOffset2") != val("BendMiddleOffset1"):
                pts.append((val("BendMiddleOffset2"), middle))
        pts.append((val("BendDestinationOffset", 100.0), dest))
        points = [gp.BendPoint(position=round(max(0.0, min(100.0, off)) * gp.BendEffect.maxPosition / 100),
                               value=round(v / 25)) for off, v in pts]
        peak = max(p.value for p in points)
        if peak <= 0 and all(p.value == 0 for p in points):
            return None
        if middle is not None and middle > max(origin, dest):
            kind = gp.BendType.bendRelease
        elif origin > 0 and dest >= origin:
            kind = gp.BendType.prebend
        elif origin > 0 and dest < origin:
            kind = gp.BendType.prebendRelease
        elif dest < origin:
            kind = gp.BendType.releaseUp
        else:
            kind = gp.BendType.bend
        bend = gp.BendEffect(type=kind, value=int(peak * 25))
        bend.points = points
        return bend


# =========================================================================== converter

GP_EXTS = {".gp3", ".gp4", ".gp5", ".gpx", ".gp", ".gtp"}
ERRONEOUS = "__erroneous"      # unreadable files are moved to <their folder>/__erroneous
EIGHT_STRING = "__8string"     # files with an 8-string (or longer) guitar/bass track go to <their folder>/__8string
ORDER = {".gp5": 0, ".gpx": 1, ".gp": 2, ".gp4": 3, ".gp3": 4, ".gtp": 5}   # who keeps the plain name on a clash


def sniff(path: Path) -> str:
    """'gp3' / 'gp4' / 'gp5' / 'gpif' / 'gp12' / 'html' / 'empty' / 'unknown'"""
    with open(path, "rb") as fh:
        head = fh.read(64)
    if not head:
        return "empty"
    n = head[0]
    if n < 40 and head[1:1 + n].startswith((b"FICHIER GUITAR PRO v", b"FICHIER GUITARE PRO v")):
        version = head[1:1 + n].decode("latin-1")
        major = version.rsplit("v", 1)[1][:1]
        return {"3": "gp3", "4": "gp4", "5": "gp5"}.get(major, "gp12")
    if head[:2] == b"PK" or head[:4] in (b"BCFZ", b"BCFS"):
        return "gpif"
    if head.lstrip()[:1] == b"<":
        return "html"
    return "unknown"


def _fit(s: str, limit: int = 250) -> str:
    return s.encode("cp1252", "replace").decode("cp1252")[:limit]


def sanitize(song: gp.Song) -> None:
    """Make the song writable as GP5: strings of 8-bit text and at most 250 characters, grace frets in range."""
    for field in ("title", "subtitle", "artist", "album", "words", "music", "copyright", "tab", "instructions", "tempoName"):
        setattr(song, field, _fit(getattr(song, field) or ""))
    song.notice = [_fit(line) for line in (song.notice or [])]
    for h in song.measureHeaders:
        if h.marker is not None:
            h.marker.title = _fit(h.marker.title)
        h.repeatAlternative &= 0xFF       # GP5 keeps the alternative endings as one byte of flags
        h.repeatClose = max(-1, min(127, h.repeatClose))
    for t in song.tracks:
        t.name = _fit(t.name, 40)
        for m in t.measures:
            for v in m.voices:
                for b in v.beats:
                    if b.text is not None:
                        b.text.value = _fit(b.text.value)
                    for n in b.notes:
                        g = n.effect.grace
                        if g is not None and g.fret < 0:      # GP3/GP4 store "dead" as -1, the GP5 writer needs 0..255
                            g.fret, g.isDead = 0, True
                        if not 0 <= n.value <= 255:
                            n.value = max(0, min(255, n.value))


def _degrade(song: gp.Song, level: int) -> None:
    for t in song.tracks:
        for m in t.measures:
            for v in m.voices:
                for b in v.beats:
                    for n in b.notes:
                        if level == 1:
                            n.effect.grace = None
                        else:
                            n.effect = gp.models.NoteEffect()
                    if level >= 2:
                        b.effect = gp.models.BeatEffect()


MAX_STRINGS = 7          # what a GP5 track can hold


class EightString(Exception):
    """The song has a guitar or bass track with 8 or more strings, which a GP5 file cannot hold."""


class Unreadable(Exception):
    """The source file itself cannot be read (damaged, truncated)."""


def _read(path: Path, kind: str) -> gp.Song:
    try:
        if kind == "gpif":
            return load_gpif(path)
        try:
            return gp.parse(str(path))
        except UnicodeDecodeError:
            return gp.parse(str(path), encoding="latin-1")
    except Exception as e:                           # noqa: BLE001
        raise Unreadable(f"source cannot be read ({type(e).__name__}: {str(e)[:70]})") from e


ALLOW_8STRING = False        # set per worker process by _init
SQUEEZE = False
DROP_SIDE = "high"
MAX_LOST = 0


def _init(allow_8string: bool, squeeze: bool = False, drop_side: str = "high", max_lost: int = 0) -> None:
    global ALLOW_8STRING, SQUEEZE, MAX_STRINGS, DROP_SIDE, MAX_LOST
    ALLOW_8STRING, SQUEEZE, DROP_SIDE, MAX_LOST = allow_8string, squeeze, drop_side, max_lost
    if squeeze:
        MAX_STRINGS = 99         # read every string, squeeze() then removes the highest ones


def _squeeze_track(t: gp.Track, lowest: bool, apply: bool, fret_limit: int = 24, target: int = 7) -> int:
    """Notes lost when ``t`` is cut to 7 strings by removing its highest (or lowest) strings; with ``apply`` the
    track is changed. A note on a removed string moves, with the same pitch, to the nearest remaining string that is
    free in that beat (fret 0..fret_limit)."""
    n = len(t.strings)
    k = n - target
    open_ = {st.number: st.value for st in t.strings}
    # string 1 is the highest; the kept strings are k+1..n, or 1..n-k when the lowest go
    kept = list(range(1, n - k + 1)) if lowest else list(range(k + 1, n + 1))
    gone = set(range(1, n + 1)) - set(kept)
    lost = 0
    for m in t.measures:
        for v in m.voices:
            for b in v.beats:
                used = {nt.string for nt in b.notes if nt.string in kept}
                keep = []
                for nt in sorted(b.notes, key=lambda x: x.string):
                    if nt.string in gone:
                        pitch = open_[nt.string] + nt.value
                        for ns in sorted(kept, key=lambda c: abs(c - nt.string)):
                            f = pitch - open_[ns]
                            if 0 <= f <= fret_limit and ns not in used:
                                used.add(ns)
                                if apply:
                                    nt.string, nt.value = ns, f
                                break
                        else:
                            lost += 1
                            continue
                    keep.append(nt)
                if apply:
                    b.notes = keep
                    for nt in keep:
                        nt.string = kept.index(nt.string) + 1
    if apply:
        t.strings = [gp.GuitarString(i + 1, open_[c]) for i, c in enumerate(kept)]
    return lost


def _octave_track(t: gp.Track, apply: bool, fret_limit: int = 24) -> int:
    """Notes lost when ``t`` is cut to 7 strings by moving every note on string 6 and lower (string 1 is the highest)
    one octave up, onto the nearest free higher string (else the nearest free string), and then removing the strings
    beyond the 7th. Returns the notes that did not fit; with ``apply`` the track is changed."""
    n = len(t.strings)
    open_ = {st.number: st.value for st in t.strings}
    lost = 0
    for m in t.measures:
        for v in m.voices:
            for b in v.beats:
                used = {nt.string for nt in b.notes if nt.string <= 5}
                keep = [nt for nt in b.notes if nt.string <= 5]
                for nt in sorted((x for x in b.notes if x.string > 5), key=lambda x: x.string):
                    pitch = open_[nt.string] + nt.value + 12
                    order = list(range(nt.string - 1, 0, -1)) + list(range(nt.string, 8))
                    for ns in order:
                        f = pitch - open_[ns]
                        if ns <= 7 and ns not in used and 0 <= f <= fret_limit:
                            used.add(ns)
                            if apply:
                                nt.string, nt.value = ns, f
                            keep.append(nt)
                            break
                    else:
                        lost += 1
                if apply:
                    b.notes = keep
    if apply:
        t.strings = t.strings[:7]
    return lost


def squeeze(song: gp.Song) -> int:
    """Cut every track of more than 7 strings down to 7, per track removing the highest or the lowest strings,
    whichever loses fewer notes (``DROP_SIDE``: "high", "low" or "best"). Returns the number of notes lost."""
    lost = 0
    for t in song.tracks:
        if len(t.strings) <= 7:
            continue
        if DROP_SIDE == "octave":
            lost += _octave_track(t, True)
            continue
        if DROP_SIDE == "any":      # per track the cheapest of: drop the highest, drop the lowest, octave shift
            tries = [(_squeeze_track(t, False, False), 0), (_squeeze_track(t, True, False), 1)]
            if len(t.strings) == 8:
                tries.append((_octave_track(t, False), 2))
            n, how = min(tries)
            lost += _octave_track(t, True) if how == 2 else _squeeze_track(t, how == 1, True)
            continue
        if DROP_SIDE == "best":
            lowest = _squeeze_track(t, True, False) < _squeeze_track(t, False, False)
        else:
            lowest = DROP_SIDE == "low"
        lost += _squeeze_track(t, lowest, True)
    return lost


def convert(src: Path, dst: Path, kind: str) -> tuple[str, str]:
    """Write ``dst`` as GP5. Returns (status, note); status is ok / degraded / failed."""
    song = _read(src, kind)
    eight = getattr(song, "eight_string", None)
    if SQUEEZE and any(len(t.strings) > 7 for t in song.tracks):  # also a piano or synth with an 8-entry tuning:
        lost = squeeze(song)                                        # read with every string, so it is cut here too
        if eight and lost > MAX_LOST:
            raise EightString(f"{lost} notes would be lost on the 7 strings: {', '.join(eight)}")
    elif eight and not SQUEEZE and not ALLOW_8STRING:
        raise EightString(", ".join(eight))
    sanitize(song)
    n_tracks, n_bars = len(song.tracks), len(song.measureHeaders)
    last = ""
    dst.parent.mkdir(parents=True, exist_ok=True)
    for level in (0, 1, 2):
        if level:
            _degrade(song, level)
        try:
            try:
                gp.write(song, str(dst), version=(5, 1, 0))
            except UnicodeEncodeError:               # a stray control character from a Latin-1 read
                gp.write(song, str(dst), version=(5, 1, 0), encoding="latin-1")
            check = gp.parse(str(dst))
            if len(check.tracks) == n_tracks and len(check.measureHeaders) == n_bars:
                return ("ok" if level == 0 else "degraded"), ("" if level == 0 else
                        ("grace notes dropped" if level == 1 else "note and beat effects dropped"))
            last = "written file does not read back with the same tracks and bars"
        except Exception as e:                       # noqa: BLE001 - any writer error: try with fewer effects
            last = f"{type(e).__name__}: {str(e)[:80]}"
    dst.unlink(missing_ok=True)
    return "failed", last


def _job(args: tuple[str, str, str]) -> tuple[str, str, str, str, float]:
    src, dst, kind = args
    t0 = time.time()
    try:
        status, note = convert(Path(src), Path(dst), kind)
    except EightString as e:
        status, note = "8string", f"8-string or longer track: {e}"
    except Unreadable as e:
        status, note = "unreadable", str(e)
    except Exception as e:                           # noqa: BLE001
        Path(dst).unlink(missing_ok=True)
        status, note = "failed", f"{type(e).__name__}: {str(e)[:100]}"
    return src, dst, status, note, time.time() - t0


def quarantine(path: Path, folder: str = ERRONEOUS) -> Path:
    """Move a file to ``<its folder>/<folder>/`` (default __erroneous), never overwriting anything."""
    dest = path.parent / folder
    dest.mkdir(exist_ok=True)
    target, k = dest / path.name, 2
    while target.exists():
        target = dest / f"{path.stem} ({k}){path.suffix}"
        k += 1
    shutil.move(str(path), str(target))
    return target


def plan(sources: list[Path], out: Path, include_gp5: bool, in_place: bool = False) -> tuple[list[tuple[str, str, str]], Counter, list[tuple[Path, str]]]:
    """(jobs, counts by kind, files that cannot be converted)"""
    files: list[tuple[Path, Path]] = []      # (file, the source folder it was found under)
    for s in sources:
        tabs = sorted(d for d in s.iterdir() if d.is_dir() and d.name.lower().startswith("tabs"))
        roots = tabs or [s]      # a library folder: its tabs* folders; any other folder: all of it
        for r in roots:
            files += [(p, s) for p in r.rglob("*") if p.is_file() and p.suffix.lower() in GP_EXTS and out not in p.parents
                      and not {ERRONEOUS, EIGHT_STRING} & set(p.parent.relative_to(r).parts)]
    root_of = {p: s for p, s in files}
    groups: dict[tuple[str, str], list[Path]] = {}
    for p, _ in files:
        groups.setdefault((str(p.parent), p.stem.lower()), []).append(p)
    jobs, kinds, bad = [], Counter(), []
    for (_, _), group in groups.items():
        group.sort(key=lambda p: ORDER[p.suffix.lower()])
        for i, p in enumerate(group):
            kind = sniff(p)
            kinds[kind] += 1
            if kind in ("html", "empty", "unknown"):
                bad.append((p, "not a Guitar Pro file" + (" (HTML page)" if kind == "html" else " (empty)" if kind == "empty" else "")))
                continue
            if kind == "gp12":
                bad.append((p, "Guitar Pro 1/2 file (this tool cannot read the format)"))
                continue
            if kind == "gp5" and not include_gp5:
                continue
            rel = p.relative_to(root_of[p]).parent
            name = p.stem + (f" (from {p.suffix.lower()[1:]})" if i else "") + ".gp5"
            dst = p.with_name(name) if in_place else out / rel / name
            jobs.append((str(p), str(dst), "copy" if kind == "gp5" else kind))
    return jobs, kinds, bad


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sources", nargs="*", default=[str(HERE)], help="folders to scan (default data/; there only the tabs* folders)")
    ap.add_argument("--out", default=str(HERE / "gp5"), help="output folder (default data/gp5)")
    ap.add_argument("--limit", type=int, default=0, help="convert at most N files (0 = all)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--in-place", action="store_true",
                    help="write each .gp5 next to its source and delete the source once the .gp5 reads back correctly (keep a backup!)")
    ap.add_argument("--copy-gp5", action="store_true", help="also copy files that already are GP5")
    ap.add_argument("--overwrite", action="store_true", help="convert again even when the output exists")
    ap.add_argument("--dry-run", action="store_true", help="only count what is there")
    ap.add_argument("--no-quarantine", action="store_true", help="leave unreadable files where they are (default: move them to <folder>/__erroneous)")
    ap.add_argument("--convert-8string", action="store_true",
                    help="convert files with an 8-string guitar/bass track too (lossy: GP5 holds 7 strings) instead of moving them to __8string")
    ap.add_argument("--squeeze-8string", action="store_true",
                    help="convert 8-string files by removing the highest string (its notes move to lower strings); files where a note would be lost are treated as 8-string files")
    ap.add_argument("--drop", choices=("high", "low", "best", "octave", "any"), default="high",
                    help="with --squeeze-8string: remove the highest string (default), the lowest, or per track whichever loses fewer notes (best: highest vs lowest; any: also the octave shift), or 'octave': notes on string 6 and lower go one octave up and the strings beyond the 7th are removed")
    ap.add_argument("--max-lost", type=int, default=0, help="with --squeeze-8string: accept files that lose up to N notes (default 0)")
    ap.add_argument("--ignore-8string", action="store_true", help="leave 8-string files where they are, neither converted nor moved")
    ap.add_argument("--keep-old-gtp", action="store_true",
                    help="leave Guitar Pro 1/2 files in place (they are valid files this tool cannot read; default: moved like other unreadable files)")
    args = ap.parse_args(argv)

    sources = [Path(s).resolve() for s in args.sources]
    out = Path(args.out).resolve()
    jobs, kinds, bad = plan(sources, out, args.copy_gp5, args.in_place)
    print(f"found: {dict(kinds)}")
    print(f"to convert: {sum(j[2] not in ('copy',) for j in jobs)}, to copy: {sum(j[2] == 'copy' for j in jobs)}, "
          f"cannot convert: {len(bad)} {dict(Counter(b[1] for b in bad))}")
    if not args.overwrite:
        jobs = [j for j in jobs if not Path(j[1]).exists()]
        print(f"not yet done: {len(jobs)}")
    if args.limit:
        jobs = jobs[:args.limit]
    if args.dry_run:
        for p, why in bad[:15]:
            print(f"  {why}: {p}")
        return 0

    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "conversion.tsv", "a", encoding="utf-8")
    moved = Counter()
    for p, why in bad:
        if args.no_quarantine or (args.keep_old_gtp and "Guitar Pro 1/2" in why):
            log.write(f"skipped\t{p}\t\t{why}\n")
        else:
            log.write(f"quarantined\t{p}\t{quarantine(p)}\t{why}\n")
            moved[why] += 1
    if not jobs:
        log.close()
        print(f"nothing to convert; moved to {ERRONEOUS}: {dict(moved)}")
        return 0
    counts, t0, done = Counter(), time.time(), 0
    todo_copy = [j for j in jobs if j[2] == "copy" and not args.in_place]
    for src, dst, _ in todo_copy:
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        counts["copied"] += 1
        log.write(f"copied\t{src}\t{dst}\t\n")
    work = [j for j in jobs if j[2] != "copy"]
    pool = mp.Pool(args.workers, initializer=_init, initargs=(args.convert_8string, args.squeeze_8string, args.drop, args.max_lost)) if args.workers > 1 else None
    _init(args.convert_8string, args.squeeze_8string, args.drop, args.max_lost)
    results = pool.imap_unordered(_job, work, chunksize=8) if pool else map(_job, work)
    try:
        for src, dst, status, note, secs in results:
            done += 1
            if status == "unreadable" and not args.no_quarantine:     # the file cannot be read at all: move it aside
                status = "quarantined"
                dst = str(quarantine(Path(src)))
            elif status == "8string" and not args.no_quarantine and not args.ignore_8string:       # GP5 cannot hold it: keep it for later, next to where it was
                dst = str(quarantine(Path(src), EIGHT_STRING))
            elif status in ("ok", "degraded") and args.in_place and Path(src) != Path(dst):
                Path(src).unlink()
            counts[status] += 1
            log.write(f"{status}\t{src}\t{dst}\t{note}\n")
            if status in ("failed", "quarantined", "8string", "unreadable") or done % 500 == 0:
                log.flush()
                print(f"[{done}/{len(work)}] {dict(counts)}" + (f"  {status.upper()} {Path(src).name}: {note}" if status in ("failed", "quarantined", "8string", "unreadable") else ""), flush=True)
    finally:
        if pool:
            pool.close()

    log.close()
    if moved:
        print(f"moved to {ERRONEOUS} before converting: {dict(moved)}")
    print(f"\ndone in {time.time() - t0:.0f}s: {dict(counts)}; details in {out / 'conversion.tsv'}")
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    sys.exit(main())
