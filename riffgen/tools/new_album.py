"""Create an album config for a folder in data/own.

    python -m riffgen.tools.new_album "Compound Structure"
    python -m riffgen.tools.new_album --all          # one config per folder in data/own

Expected layout (any of the sub-folders may be missing):

    data/own/<Album>/own/          your songs of this album (finished)
    data/own/<Album>/favourites/   songs you like most, with __v2/__v3 copies (D9)
    data/own/<Album>/influences/   songs of similar bands
    data/own/<Album>/songs/        unfinished songs with gaps (only used for generation)

If an album folder has GP files directly inside (no sub-folders), they are treated as "own".
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from riffgen import REPO_DIR

TEMPLATE = """# Album config: {title}
inherit: train/finetune.yaml
album_id: {album_id}
title: "{title}"
model: big                      # baseline | big | huge (which pretrained base to fine-tune)
# base_model: runs/pretrain_big/best.pt
tuning: drop_a7
min_gap_bars: 4

data:
  root: data/own/{folder}
  own: {own}
  others: {others}
  holdout: []                   # D32: 1-2 of your own songs, e.g. [own/song_07.gp5]. Never make copies of them.
  songs: songs                  # unfinished songs with gaps

filters:
  simple_riff_action: context   # keep | context | drop
  force_keep: []                # - {{file: own/song_03.gp5, bars: 49-56}}
  force_drop: []

dadagp_mix:                     # §3.9: similar DadaGP songs mixed into fine-tuning
  enabled: true
  share: 0.15
  method: features              # features | embeddings
  k_nearest: 5
  top_riffs_per_song: 3
  max_songs: 1000
  max_per_artist: 20
  prefer_tunings: [drop_a7, dropdbl_a7, a_std7, b_std7]
  always_include_artists: [Gojira, Opeth, Meshuggah, Periphery]
  exclude_artists: []

generation:
  candidates: 32
  keep: 10
  temperature: 1.0
  top_p: 0.95
  min_complexity: 0.25
  max_neighbour_similarity: 0.95
"""


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def write_config(folder: Path, force: bool = False) -> Path:
    album_id = slug(folder.name)
    out = REPO_DIR / "configs" / "albums" / f"{album_id}.yaml"
    if out.exists() and not force:
        print(f"exists, not overwritten: {out}")
        return out
    subs = {p.name for p in folder.iterdir() if p.is_dir()} if folder.is_dir() else set()
    own = "own" if "own" in subs or not subs else "."
    if own == "." and any(folder.glob("*.gp[345]")):
        own = "."
    others = [s for s in ("favourites", "influences") if s in subs or not subs]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(TEMPLATE.format(title=folder.name, album_id=album_id, folder=folder.name, own=own,
                                   others="[" + ", ".join(others) + "]"), encoding="utf-8")
    print(f"wrote {out}")
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", nargs="?")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    base = REPO_DIR / "data" / "own"
    if args.all:
        for d in sorted(p for p in base.iterdir() if p.is_dir()):
            write_config(d, args.force)
    elif args.name:
        write_config(base / args.name, args.force)
    else:
        ap.error("give an album folder name or --all")


if __name__ == "__main__":
    main()
