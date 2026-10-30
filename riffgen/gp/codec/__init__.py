"""Guitar Pro to tokens: the encoder half of the codec, kept in-tree.

Adapted from DadaGP (MIT licence, Dadabots — see ``LICENSE-dadagp.txt`` in this folder). The
encoder is used as it came; decoding lives in ``riffgen.gp.writer``, because the original
decoder rounds durations it cannot represent (bars change length) and silently drops tremolo
picking.

``gp_tokens`` reads ``blank.gp5`` (a template song it clones) at import time and prints a lot,
so it is imported once, from this folder, with stdout silenced.
"""

from __future__ import annotations

import contextlib
import io
from functools import lru_cache
from pathlib import Path

BLANK_GP5 = Path(__file__).with_name("blank.gp5")


@lru_cache(maxsize=1)
def _codec():
    with contextlib.redirect_stdout(io.StringIO()):
        from . import gp_tokens  # noqa: PLC0415
    return gp_tokens


def instrument_group(track) -> str:
    """The codec's instrument group of a track: distorted, clean, bass, leads, pads, drums."""
    if track.isPercussionTrack:
        return "drums"
    return _codec().instrument_groups.get(track.channel.instrument, "leads")


def encode_song(song, artist: str = "artist:unknown") -> list[str]:
    """Encode an in-memory PyGuitarPro song. Raises on songs the encoder cannot represent."""
    with contextlib.redirect_stdout(io.StringIO()):
        return _codec().guitarpro2tokens(song, artist, verbose=False)


def tokens_to_note_effect(note, nfx_tokens):
    return _codec().tokens_to_note_effect(note, nfx_tokens)


def tokens_to_beat_effect(effect, bfx_tokens):
    return _codec().tokens_to_beat_effect(effect, bfx_tokens)


def blank_song():
    """A fresh copy of the template song the encoder clones for its output."""
    import guitarpro  # noqa: PLC0415

    return guitarpro.parse(str(BLANK_GP5))
