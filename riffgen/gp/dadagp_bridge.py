"""Access to the vendored DadaGP encoder (third_party/dadagp, MIT licence, Dadabots).

dadagp.py parses ``blank.gp5`` from the *current directory* when it is imported and prints
a lot; we import it from its own folder with stdout silenced. We only use its encoder
(``guitarpro2tokens``). Decoding is done by riffgen.gp.writer, because the DadaGP decoder
rounds durations it cannot represent (bars change length) and silently drops tremolo
picking (the encoder writes ``nfx:tremoloPicking``, the decoder looks for
``tremolo_picking``).
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
from functools import lru_cache

from riffgen import THIRD_PARTY_DIR

DADAGP_DIR = THIRD_PARTY_DIR / "dadagp"


@lru_cache(maxsize=1)
def dadagp():
    if str(DADAGP_DIR) not in sys.path:
        sys.path.insert(0, str(DADAGP_DIR))
    old = os.getcwd()
    try:
        os.chdir(DADAGP_DIR)
        with contextlib.redirect_stdout(io.StringIO()):
            import dadagp as module  # noqa: PLC0415
    finally:
        os.chdir(old)
    return module


def instrument_group(track) -> str:
    """DadaGP's instrument group of a track: distorted, clean, bass, leads, pads, drums, remove."""
    if track.isPercussionTrack:
        return "drums"
    return dadagp().instrument_groups.get(track.channel.instrument, "leads")


def encode_song(song, artist: str = "artist:unknown") -> list[str]:
    """Run DadaGP's encoder on an in-memory PyGuitarPro song. Raises on unsupported songs."""
    with contextlib.redirect_stdout(io.StringIO()):
        return dadagp().guitarpro2tokens(song, artist, verbose=False)
