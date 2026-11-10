"""Curate a collection of Guitar Pro tabs on your own disk, by the rules the training set was cleaned with.

    python data_curate.py "D:/tabs/new stuff" --out "D:/tabs/curated"
    python data_curate.py a.gpx b.gp4 "some folder" --out "D:/tabs/curated" --genre-folders --genre Metal
    python data_curate.py "D:/tabs/new stuff" --out "D:/tabs/curated" --move

These are the rules every download of dl.py goes through too (dl.py imports them from here). Every tab
(.gp3/.gp4/.gp5/.gpx/.gp, folders are searched recursively):

- it must open in PyGuitarPro; anything else is reported as failed and left where it is;
- it is converted to GP5; 8-string tracks are cut to 7 strings, per track the cheapest of: drop the highest
  string, drop the lowest, shift an octave;
- instrument changes are replaced by one track per instrument (when the MIDI channels run out, the first
  instrument of each track is kept for the whole tab);
- 6-string guitars get a 7th string (A);
- the result must read back with the same tracks, bars, notes and instruments;
- it must pass the training filters (riff.py's encoder, as ``riff.py prepare`` runs it): a guitar track with
  notes, a tuning riff.py can map onto 7-string Drop A, time signatures it knows, no bar too dense to encode;
  a tab that fails is "rejected" (``--no-train-filters`` skips this; it needs numpy and torch);
- the name is cleaned: no special characters, no genre tag, no __v/__id marker; the artist in one canonical form:
  "Beatles, The" / "Beatles (The)" -> "The Beatles", "Hendrix, Jimi" -> "Jimi Hendrix" (always when the collection
  already spells it so, else when the surname is one plain word), then the collection's own spelling; a file not named
  "Artist - Song" is named after the artist and title stored in the tab, when it has them;
- a byte-identical copy anywhere in the collection is dropped, and of a song with more than ``--keep-versions``
  (1) different versions only the largest stay.

The curated tabs land in ``--out`` (flat), or with ``--genre-folders`` in a genre folder there: the folder
where most of the artist's tabs already are, else "tabs - <genre>" for ``--genre``, else "tabs - metal generic".
Anything pushed out or rejected goes to <out>/__removed, nothing is deleted. The originals are left alone unless
``--move`` (then a tab that was curated, a duplicate or rejected is deleted from where it was). A report of every
file is written to <out>/curate_report.csv. Needs PyGuitarPro, and data_convert.py next to this file.
"""

from __future__ import annotations

import argparse
import bisect
import collections
import copy
import csv
import hashlib
import os
import re
import shutil
import sys
import unicodedata
from datetime import datetime
from pathlib import Path

try:
    import guitarpro as gp
except ImportError:   # dl.py borrows the names from here; listing tabs works without PyGuitarPro, curating does not
    gp = None

GP_EXTENSIONS = (".gp3", ".gp4", ".gp5", ".gpx", ".gp")
REMOVED = "__removed"                       # copies pushed out by better ones: moved here, never deleted
FLAT = "."                                  # the output folder itself
DEFAULT_FOLDER = "tabs - metal generic"     # --genre-folders, when no genre can be inferred
NOT_GENRE_OF_ARTIST = {FLAT, "tabs - generic", "tabs - errors", "tabs - CS", "tabs - exercises"}   # bins, not a genre

_KEEP = set(" -_(),'.")
_REPLACE = {"&": " and ", "[": "(", "{": "(", "]": ")", "}": ")", "´": "'", "`": "'", "’": "'", "‘": "'", "ʼ": "'",
            "ø": "o", "Ø": "O", "æ": "ae", "Æ": "AE", "œ": "oe", "Œ": "OE", "ß": "ss", "ł": "l", "Ł": "L", "đ": "d", "Đ": "D",
            "ð": "d", "Ð": "D", "þ": "th", "Þ": "Th", "ı": "i", "ª": "a", "º": "o", "№": "No", "×": "x", "–": "-", "—": "-"}


def clean(s: str) -> str:
    """A name without special characters: accents dropped, & -> and, symbols removed, other scripts kept."""
    s = "".join(_REPLACE.get(ch, ch) for ch in s)
    out = []
    for ch in unicodedata.normalize("NFKD", s):
        if unicodedata.combining(ch):
            continue
        if ch.isascii():
            out.append(ch if ch.isalnum() or ch in _KEEP else " ")
        else:
            m = re.match(r"LATIN (SMALL|CAPITAL) LETTER ([A-Z])", unicodedata.name(ch, ""))
            if m:
                out.append(m.group(2).lower() if m.group(1) == "SMALL" else m.group(2))
            else:
                out.append(ch if unicodedata.category(ch)[0] in "LN" else " ")
    s = "".join(out)
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s+([),])", r"", s)
    s = re.sub(r"\(\s+", "(", s)
    s = re.sub(r"(?:\s-)+\s-\s", " - ", s)
    return s.strip(" .-")


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


# a tab renamed to "Artist - Song (Genre)" (Deezer genre labels) still counts as that song
GENRES = ("Rock", "Metal", "Pop", "Alternative", "Hard Rock", "Jazz", "Blues", "R&B", "Electro", "Films/Games",
          "Dance", "Film Scores", "Classical", "Indie Rock", "Rock & Roll/Rockabilly", "Folk", "Rap/Hip Hop",
          "Singer & Songwriter", "International Pop", "Latin Music", "Asian Music", "Indie Rock/Rock pop",
          "Indie Pop", "Soul & Funk", "Techno/House", "Disco", "Kids", "Reggae", "Brazilian Music", "Trance",
          "Baroque", "Bolero", "Comedy", "Instrumental jazz", "Indie Pop/Folk", "Contemporary Soul",
          "Electro Pop/Electro Rock", "Dubstep", "Acoustic", "Electronic", "Blues-Soul", "Soundtrack", "Ethnic", "Doom", "Experimental", "Funk", "Exercise", "Hymn",
          "Xtreme Metal", "Groove Metal", "Melodic Death Metal", "Thrash Metal", "Prog", "Shred", "Punk", "Nu Metal", "Grunge")
GENRE_SUFFIX = re.compile(r"\s*\((?:%s)\)\s*$" % "|".join(re.escape(g) for g in GENRES), re.I)

# genre -> genre folder (derived from where the library's artists ended up)
GENRE_FOLDER = {
    "rock": "rock", "alternative": "rock", "indie rock": "rock", "indie rock/rock pop": "rock",
    "rock & roll/rockabilly": "rock", "rap/hip hop": "rock", "metal": "metal generic", "hard rock": "metal generic",
    "pop": "pop", "international pop": "pop", "indie pop": "pop", "jazz": "jazz", "instrumental jazz": "jazz",
    "progressive rock": "prog", "prog": "prog", "blues": "blues-soul-rnb", "r&b": "blues-soul-rnb",
    "contemporary soul": "blues-soul-rnb", "reggae": "blues-soul-rnb", "soul & funk": "funk", "funk": "funk",
    "classical": "classical", "baroque": "classical", "electro": "electronic", "dance": "electronic",
    "disco": "electronic", "techno/house": "electronic", "trance": "electronic", "dubstep": "electronic",
    "electro pop/electro rock": "electronic", "film scores": "soundtracks", "films/games": "soundtracks",
    "experimental": "experimental", "folk": "acoustic", "singer & songwriter": "acoustic",
    "asian music": "acoustic", "indie pop/folk": "acoustic", "acoustic": "acoustic", "latin music": "ethnic",
    "brazilian music": "ethnic", "bolero": "ethnic", "ethnic": "ethnic", "punk": "punk", "grunge": "grunge",
}
ARTICLES = {"the", "el", "la", "los", "las", "le", "les", "die", "der", "das", "il"}
SECOND_NOT_A_FIRST_NAME = {"band", "orchestra", "ensemble", "trio", "quartet", "project", "jr", "sr", "inc", "the"}   # "Surname, Band" stays
NOT_A_NAME = re.compile(r"[0-9&+/]|\band\b", re.I)

GM = ("Acoustic Grand Piano|Bright Acoustic Piano|Electric Grand Piano|Honky-tonk Piano|Electric Piano 1|Electric Piano 2|"
      "Harpsichord|Clavinet|Celesta|Glockenspiel|Music Box|Vibraphone|Marimba|Xylophone|Tubular Bells|Dulcimer|"
      "Drawbar Organ|Percussive Organ|Rock Organ|Church Organ|Reed Organ|Accordion|Harmonica|Tango Accordion|"
      "Nylon Guitar|Steel Guitar|Jazz Guitar|Clean Guitar|Muted Guitar|Overdriven Guitar|Distortion Guitar|Guitar Harmonics|"
      "Acoustic Bass|Finger Bass|Pick Bass|Fretless Bass|Slap Bass 1|Slap Bass 2|Synth Bass 1|Synth Bass 2|"
      "Violin|Viola|Cello|Contrabass|Tremolo Strings|Pizzicato Strings|Orchestral Harp|Timpani|"
      "String Ensemble 1|String Ensemble 2|Synth Strings 1|Synth Strings 2|Choir Aahs|Voice Oohs|Synth Voice|Orchestra Hit|"
      "Trumpet|Trombone|Tuba|Muted Trumpet|French Horn|Brass Section|Synth Brass 1|Synth Brass 2|"
      "Soprano Sax|Alto Sax|Tenor Sax|Baritone Sax|Oboe|English Horn|Bassoon|Clarinet|"
      "Piccolo|Flute|Recorder|Pan Flute|Blown Bottle|Shakuhachi|Whistle|Ocarina|"
      "Square Lead|Sawtooth Lead|Calliope Lead|Chiff Lead|Charang Lead|Voice Lead|Fifths Lead|Bass+Lead|"
      "New Age Pad|Warm Pad|Polysynth Pad|Choir Pad|Bowed Pad|Metallic Pad|Halo Pad|Sweep Pad|"
      "Rain FX|Soundtrack FX|Crystal FX|Atmosphere FX|Brightness FX|Goblins FX|Echoes FX|Sci-Fi FX|"
      "Sitar|Banjo|Shamisen|Koto|Kalimba|Bagpipe|Fiddle|Shanai|"
      "Tinkle Bell|Agogo|Steel Drums|Woodblock|Taiko Drum|Melodic Tom|Synth Drum|Reverse Cymbal|"
      "Guitar Fret Noise|Breath Noise|Seashore|Bird Tweet|Telephone Ring|Helicopter|Applause|Gunshot").split("|")


def stem_key(stem: str) -> str:
    """One key for all copies of a song: version markers, "(2)" and a genre tag do not count."""
    s = re.sub(r"__id\d+|__v\d+", "", stem)
    s = re.sub(r"\s*\((?:ver|version)\s*\.?\s*\d+[^)]*\)|\s*\(\d+\)\s*$|\s+v\d+$", "", s, flags=re.I)
    s = GENRE_SUFFIX.sub("", s)  # "Artist - Song (Rock)" -> "Artist - Song"
    return re.sub(r"^the", "", re.sub(r"[^a-z0-9]+", "", clean(s).lower()))


def artist_key(a: str) -> str:
    return re.sub(r"^the", "", re.sub(r"[^a-z0-9]+", "", clean(a).lower()))


class Library:
    """Index of the output folder (and, with ``genre_folders``, its sub-folders matching ``pattern``): songs,
    artists, file names, kept current as files come and go."""

    def __init__(self, root: Path, genre_folders: bool = False, pattern: str = "*"):
        self.root, self.genre_folders = Path(root), genre_folders
        self.by_key: dict[str, list[Path]] = collections.defaultdict(list)
        self.artist_folder: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
        self.artist_forms: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
        self.names: dict[str, set[str]] = collections.defaultdict(set)
        dirs = [self.root] + (sorted(d for d in self.root.glob(pattern) if d.is_dir() and d.name != REMOVED)
                              if genre_folders and self.root.is_dir() else [])
        for d in dirs:
            if d.is_dir():
                for p in d.iterdir():
                    if p.is_file():
                        self.add(p)

    def folder_of(self, p: Path) -> str:
        return FLAT if p.parent == self.root else p.parent.name

    def add(self, p: Path) -> None:
        folder = self.folder_of(p)
        self.names[folder].add(p.name.lower())
        if p.suffix.lower() not in GP_EXTENSIONS:
            return
        self.by_key[stem_key(p.stem)].append(p)
        if " - " in p.stem:
            a = p.stem.split(" - ", 1)[0].strip()
            self.artist_forms[artist_key(a)][a] += 1
            if folder not in NOT_GENRE_OF_ARTIST:
                self.artist_folder[artist_key(a)][folder] += 1

    def remove(self, p: Path) -> None:
        self.names[self.folder_of(p)].discard(p.name.lower())
        lst = self.by_key.get(stem_key(p.stem), [])
        if p in lst:
            lst.remove(p)

    def free_name(self, folder: str, stem: str, ext: str = ".gp5") -> str:
        n, name = 2, stem + ext
        while name.lower() in self.names[folder]:
            name = f"{stem} ({n}){ext}"
            n += 1
        return name

    def tidy_artist(self, a: str) -> str:
        """The artist name in its one canonical form:

        - "Beatles, The" and "Beatles (The)" -> "The Beatles" (also El/La/Los/Las/Le/Les/Die/Der/Das/Il: "Renga (La)");
        - "Vai, Steve" -> "Steve Vai": always when the collection already spells that artist so, otherwise when
          the surname is a single word (no digits, "&"/"and", nothing like "Band, Jr");
        - then the spelling the collection uses most for that artist ("5 nizza" -> "5nizza"), if it has one."""
        a = a.strip()
        m = re.match(r"^(.*?)\s*\((%s)\)$" % "|".join(sorted(ARTICLES)), a, re.I)
        if m and m.group(1):
            a = m.group(2)[0].upper() + m.group(2)[1:].lower() + " " + m.group(1).strip()
        if a.count(",") == 1:
            x, y = (s.strip() for s in a.split(","))
            if x and y:
                if y.lower() in ARTICLES:
                    a = y[0].upper() + y[1:].lower() + " " + x
                elif not NOT_A_NAME.search(a) and 1 <= len(x.split()) <= 3 and 1 <= len(y.split()) <= 3:
                    forms = self.artist_forms.get(artist_key(f"{y} {x}"))
                    if forms:
                        a = forms.most_common(1)[0][0]
                    elif (len(x.split()) == 1 and y.lower() not in SECOND_NOT_A_FIRST_NAME and not y.split()[0][0].isdigit()
                      and x.lower() not in y.lower().split() and not y.lower().startswith("the ")):
                        a = f"{y} {x}"
        forms = self.artist_forms.get(artist_key(a))
        if forms and a not in forms:
            return sorted(forms, key=lambda v: (-forms[v], v.islower() or v.isupper(), v))[0]
        return a

    def folder_for(self, artist: str, genre: str) -> str:
        """Flat: the output folder. Genre folders: where the artist's tabs already are; else the genre's folder;
        else tabs - metal generic."""
        if not self.genre_folders:
            return FLAT
        seen = self.artist_folder.get(artist_key(artist))
        if seen:
            return seen.most_common(1)[0][0]
        for g in (x.strip().lower() for x in genre.split(",")):
            if g in GENRE_FOLDER:
                return "tabs - " + GENRE_FOLDER[g]
        return DEFAULT_FOLDER

    def path(self, folder: str) -> Path:
        return self.root if folder == FLAT else self.root / folder

    def removed(self, folder: str, name: str) -> Path:
        """A free path for ``name`` in __removed/<folder> (an earlier copy there is never overwritten)."""
        d = self.root / REMOVED / ("" if folder == FLAT else folder)
        d.mkdir(parents=True, exist_ok=True)
        stem, ext = os.path.splitext(name)
        target, k = d / name, 2
        while target.exists():
            target, k = d / f"{stem} ({k}){ext}", k + 1
        return target


# --- instrument changes: one track per instrument ----------------------------------------------------------

class NoChannel(Exception):
    """No free MIDI channel (port 1 has 15) for another copy of a track."""


def _tbeats(track) -> list:
    return [(b.start, vi, b) for m in track.measures for vi, v in enumerate(m.voices) for b in v.beats]


def _is_perc(track) -> bool:
    return bool(track.isPercussionTrack or track.channel.isPercussionChannel)


def _timeline(track) -> tuple[int, list[int], list[int], int]:
    """(first instrument, [start of each change], [instrument from there on], number of events)."""
    ev = sorted((st, vi, b.effect.mixTableChange.instrument.value) for st, vi, b in _tbeats(track)
                if b.effect.mixTableChange is not None and b.effect.mixTableChange.instrument is not None)
    starts, vals = [], []
    for st, _, v in ev:
        if starts and starts[-1] == st:
            vals[-1] = v
        else:
            starts.append(st)
            vals.append(v)
    return track.channel.instrument, starts, vals, len(ev)


def _instruments(track) -> tuple[list[int], list[int]]:
    """(instruments that play at least one note, in order of their first note; the instrument of every beat)."""
    init, starts, vals, _ = _timeline(track)
    per_beat = []
    for st, _, b in _tbeats(track):
        i = bisect.bisect_right(starts, st)
        per_beat.append(vals[i - 1] if i else init)
    order = []
    for (_, _, b), ins in zip(_tbeats(track), per_beat):
        if b.notes and ins not in order:
            order.append(ins)
    return order, per_beat


def _strip_events(track) -> None:
    for _, _, b in _tbeats(track):
        mt = b.effect.mixTableChange
        if mt is not None and mt.instrument is not None:
            mt.instrument = None
            if all(getattr(mt, a) is None for a in ("volume", "balance", "chorus", "reverb", "phaser", "tremolo", "tempo", "wah")):
                b.effect.mixTableChange = None


def _blank(b) -> None:
    if b.notes:
        b.notes = []
        b.status = gp.models.BeatStatus.rest
    b.effect = gp.models.BeatEffect()
    b.text = None


def _free_channel(song) -> tuple[int, int]:
    """A MIDI channel nobody uses: first with the usual channel/effect-channel pairs, then (every effect channel
    folded onto its own channel) with single channels. NoChannel when port 1 is full."""
    for attempt in (0, 1):
        used = {x for t in song.tracks for x in (t.channel.channel, t.channel.effectChannel)}
        for c in range(16):
            if c == 9 or c in used:
                continue
            e = c + 1 if (c + 1 < 16 and c + 1 != 9 and c + 1 not in used) else c
            return c, e
        if attempt == 0:
            for t in song.tracks:
                if not t.channel.isPercussionChannel:
                    t.channel.effectChannel = t.channel.channel
    raise NoChannel


def split_instruments(song) -> int:
    """Replace the instrument changes of every non-drum track by one track per instrument (the original keeps
    the first, copies named "<track> (<instrument>)" follow it); each copy plays only its instrument's notes and
    rests elsewhere, so bars and timing stay. Returns the number of tracks added. NoChannel if they do not fit."""
    memo = {id(song): song, **{id(h): h for h in song.measureHeaders}}
    lyr = song.lyrics.trackChoice if song.lyrics is not None else -1
    lyr_track = song.tracks[lyr] if 0 <= lyr < len(song.tracks) else None
    added, ti = 0, 0
    while ti < len(song.tracks):
        t = song.tracks[ti]
        if _is_perc(t) or not _timeline(t)[3]:
            ti += 1
            continue
        order, per_beat = _instruments(t)
        if len(order) <= 1:
            if order:
                t.channel.instrument = order[0]
            _strip_events(t)
            ti += 1
            continue
        for k, ins in enumerate(order[1:], 1):
            c = copy.deepcopy(t, dict(memo))
            for (_, _, b), pi in zip(_tbeats(c), per_beat):
                if pi != ins:
                    _blank(b)
            c.channel.channel, c.channel.effectChannel = _free_channel(song)
            c.channel.instrument = ins
            c.name = (t.name + f" ({GM[ins] if 0 <= ins < 128 else ins})")[:40]
            _strip_events(c)
            song.tracks.insert(ti + k, c)       # visible to the next _free_channel
        for (_, _, b), pi in zip(_tbeats(t), per_beat):
            if pi != order[0]:
                _blank(b)
        t.channel.instrument = order[0]
        _strip_events(t)
        added += len(order) - 1
        ti += len(order)
    for i, t in enumerate(song.tracks, 1):
        t.number = i
    if lyr_track is not None:
        song.lyrics.trackChoice = song.tracks.index(lyr_track)
    return added


def drop_instrument_changes(song):
    """-> (song, how): copies per instrument, or, when that does not fit, the first instrument of every track."""
    if not any(not _is_perc(t) and _timeline(t)[3] for t in song.tracks):
        return song, ""
    snapshot = copy.deepcopy(song)
    try:
        return song, f"instrument changes -> {split_instruments(song)} tracks added"
    except NoChannel:
        for t in snapshot.tracks:
            if not _is_perc(t):
                _strip_events(t)
        return snapshot, "instrument changes removed (first instrument kept)"


# --- 7-string guitars ----------------------------------------------------------------------------------------

def _is_bass(t) -> bool:
    return 32 <= t.channel.instrument <= 39 or "bass" in t.name.lower()


def _is_guitar(t) -> bool:
    if t.isPercussionTrack or _is_bass(t):
        return False
    return 24 <= t.channel.instrument <= 31 or "guitar" in t.name.lower() or "gtr" in t.name.lower()


def to_seven_strings(song) -> int:
    """A 6-string guitar gets a 7th, lowest string tuned A1 (the next A below the lowest string when that is lower)."""
    n = 0
    for t in song.tracks:
        if len(t.strings) == 6 and _is_guitar(t):
            low, a = min(s.value for s in t.strings), 33
            while a >= low:
                a -= 12
            t.strings.append(gp.GuitarString(7, a))
            n += 1
    return n


# --- the pipeline ------------------------------------------------------------------------------------------

_CV = None


def convert_module():
    """data_convert.py, set up to cut 8-string tracks to 7 strings with the cheapest method and to accept any loss."""
    global _CV
    if _CV is None:
        here = str(Path(__file__).resolve().parent)
        if here not in sys.path:
            sys.path.insert(0, here)
        import data_convert as convert
        convert._init(True, True, "any", 10 ** 9)
        _CV = convert
    return _CV


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _parse(path: Path):
    try:
        return gp.parse(str(path))
    except UnicodeDecodeError:
        return gp.parse(str(path), encoding="latin-1")


def _removed(path: Path, lib: Library, why: str) -> None:
    target = lib.removed(lib.folder_of(path), path.name)
    shutil.move(str(path), str(target))
    log(f"    moved to {target}: {why}")


_RIFF = None


def training_check(song) -> str:
    """Why ``riff.py prepare`` would skip this song ("" when it would not): riff.py's own encoder decides."""
    global _RIFF
    if _RIFF is None:
        import importlib.util
        path = Path(__file__).resolve().parent / "riff.py"
        spec = importlib.util.spec_from_file_location("riff", path)
        riff = importlib.util.module_from_spec(spec)
        sys.modules.setdefault("riff", riff)
        spec.loader.exec_module(riff)
        _RIFF = riff, riff.Vocab(riff.make_tokens())
    riff, vocab = _RIFF
    try:
        riff.encode_song(song, vocab)
    except riff.Skip as e:
        return str(e)
    return ""


def curate_file(src: Path, genre: str, lib: Library, keep_versions: int = 1,
                train_filters: bool = True) -> tuple[str, str, str]:
    """Clean one tab and put the result into the library's folder; ``src`` itself is not touched. -> (status,
    final path, note); status is ok / duplicate (not kept: same file, or not among the best versions; the curated
    copy is in __removed) / rejected (fails the training filters; the curated copy is in __removed/rejected) /
    failed."""
    cv = convert_module()
    lib.root.mkdir(parents=True, exist_ok=True)
    tmp = lib.root / f".curate_{os.getpid()}_a.gp5"
    tmp2 = lib.root / f".curate_{os.getpid()}_b.gp5"
    steps = []
    try:
        kind = cv.sniff(src)
        if kind in ("empty", "html", "unknown", "gp12"):
            return "failed", "", f"not a readable Guitar Pro file ({kind})"
        try:
            status, note = cv.convert(src, tmp, kind)      # any format -> GP5; 8-string tracks cut to 7
        except cv.Unreadable as e:
            return "failed", "", str(e)
        if status == "failed":
            return "failed", "", f"conversion failed: {note}"
        if kind != "gp5":
            steps.append(f"{kind} -> gp5")
        if status == "degraded":
            steps.append(note)
        song = _parse(tmp)                                  # a clean GP5 song: beat starts, channels, ...
        if any(len(t.strings) > 7 for t in song.tracks):
            return "failed", "", "a track still has more than 7 strings"
        n_bars, notes = len(song.measureHeaders), sum(len(b.notes) for t in song.tracks for _, _, b in _tbeats(t))
        song, how = drop_instrument_changes(song)
        if how:
            steps.append(how)
        n7 = to_seven_strings(song)
        if n7:
            steps.append(f"{n7} guitar track(s) -> 7 strings")
        try:
            gp.write(song, str(tmp2), version=(5, 1, 0))
        except UnicodeEncodeError:
            cv.sanitize(song)
            gp.write(song, str(tmp2), version=(5, 1, 0))
        chk = _parse(tmp2)                                  # health: it must open, and say the same as the song
        if (len(chk.tracks) != len(song.tracks) or len(chk.measureHeaders) != n_bars
                or sum(len(b.notes) for t in chk.tracks for _, _, b in _tbeats(t)) != notes
                or any(not _is_perc(t) and _timeline(t)[3] for t in chk.tracks)
                or [t.channel.instrument for t in chk.tracks] != [t.channel.instrument for t in song.tracks]):
            return "failed", "", "the curated file does not read back the same"

        # name, folder
        stem = GENRE_SUFFIX.sub("", re.sub(r"__(?:id|v)\d+", "", src.stem)).strip()
        if " - " not in stem and (song.artist or "").strip() and (song.title or "").strip():
            stem = f"{song.artist.strip()} - {song.title.strip()}"          # the names stored in the tab
        if " - " in stem:
            artist, rest = stem.split(" - ", 1)
            stem = f"{clean(lib.tidy_artist(artist.strip()))} - {clean(rest)}"
        else:
            stem = clean(stem) or "untitled"
        why = training_check(chk) if train_filters else ""
        if why:
            os.replace(tmp2, lib.removed("rejected", f"{stem}.gp5"))
            return "rejected", "", f"training filter: {why}"
        artist = stem.split(" - ", 1)[0] if " - " in stem else ""
        folder = lib.folder_for(artist, genre)
        lib.path(folder).mkdir(parents=True, exist_ok=True)

        # exact copy anywhere in the collection, and the best versions of the song
        digest, size = sha256(tmp2), tmp2.stat().st_size
        same = [p for p in lib.by_key[stem_key(stem)] if p.exists() and p.resolve() != src.resolve()]   # not itself
        for p in same:
            if p.stat().st_size == size and sha256(p) == digest:
                os.replace(tmp2, lib.removed(folder, f"{stem} (same as {p.name}).gp5"))
                return "duplicate", "", f"identical to {p}"
        versions: dict[str, list] = {digest: [size, [None]]}      # content -> [size, files]
        for p in same:
            h = sha256(p)
            versions.setdefault(h, [p.stat().st_size, []])[1].append(p)
        if len(versions) > keep_versions:
            ranked = sorted(versions, key=lambda h: (-versions[h][0], h != digest))
            for h in ranked[keep_versions:]:
                if h == digest:
                    os.replace(tmp2, lib.removed(folder, f"{stem}.gp5"))
                    return "duplicate", "", f"not among the best {keep_versions} versions of the song"
                for p in versions[h][1]:
                    _removed(p, lib, f"worse than a new version of the song (kept: the {keep_versions} largest)")
                    lib.remove(p)
        final = lib.path(folder) / lib.free_name(folder, stem)
        os.replace(tmp2, final)
        lib.add(final)
        if folder != FLAT:
            steps.append(f"-> {folder}")
        return "ok", str(final), "; ".join(steps)
    except Exception as e:                                  # noqa: BLE001 - one bad file must not stop a run
        return "failed", "", f"curation error: {type(e).__name__}: {str(e)[:150]}"
    finally:
        tmp.unlink(missing_ok=True)
        tmp2.unlink(missing_ok=True)


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):     # tab names contain anything; a redirected console must not choke
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="Guitar Pro files, or folders (searched recursively)")
    ap.add_argument("--out", required=True, help="folder the curated tabs go to (may already hold a collection)")
    ap.add_argument("--genre-folders", action="store_true",
                    help='file each tab into a genre folder of --out ("tabs - rock", ...), as data/ is organised')
    ap.add_argument("--genre", default="", help="genre for artists not yet in --out (with --genre-folders)")
    ap.add_argument("--keep-versions", type=int, default=1, help="of one song, the largest versions that stay")
    ap.add_argument("--no-train-filters", action="store_true",
                    help="keep tabs riff.py prepare would skip (no guitar, odd tuning or time signature, too dense)")
    ap.add_argument("--move", action="store_true",
                    help="delete each original once it is curated (or a duplicate, or rejected)")
    args = ap.parse_args(argv)
    if not args.no_train_filters:
        try:
            training_check(gp.Song())          # loads riff.py (numpy, torch) once, up front
        except ImportError as e:
            sys.exit(f"the training filters need riff.py's packages ({e}); install them or pass --no-train-filters")
        except Exception:                      # noqa: BLE001 - an empty song may upset the encoder; riff.py is loaded
            pass

    out = Path(args.out).resolve()
    files = []
    for s in args.paths:
        p = Path(s).resolve()
        if not p.exists():
            sys.exit(f"no such file or folder: {p}")
        if out == p or out in p.parents or (p.is_dir() and p in out.parents):
            sys.exit(f"--out must not be inside {p} or contain it: pick a separate folder")
        files += sorted(f for f in p.rglob("*") if f.is_file() and f.suffix.lower() in GP_EXTENSIONS) if p.is_dir() else [p]
    lib, count, report = Library(out, args.genre_folders), collections.Counter(), []
    log(f"{len(files)} tabs to curate into {out}")
    for i, f in enumerate(files, 1):
        status, final, note = curate_file(f, args.genre, lib, args.keep_versions, not args.no_train_filters)
        count[status] += 1
        report.append({"source": str(f), "status": status, "file": final, "note": note})
        log(f"[{i}/{len(files)}] {status}: {f.name} :: {note}")
        if args.move and status != "failed":
            f.unlink(missing_ok=True)
    with open(out / "curate_report.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["source", "status", "file", "note"])
        w.writeheader()
        w.writerows(report)
    log(f"{dict(count)}; report: {out / 'curate_report.csv'}")
    return 0 if not count["failed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
