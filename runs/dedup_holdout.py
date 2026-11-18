#!/usr/bin/env python
"""Leave each holdout tab in holdout/ ONLY, removing the copy that cancels it out.

    .venv/bin/python runs/dedup_holdout.py            # dry run
    .venv/bin/python runs/dedup_holdout.py --apply

Why this matters, and why it is invisible without checking: prepare keys "dup" on the file's CONTENT
digest, and sorts candidates by (is album folder, path) so the first one seen becomes the original. For a
tab sitting in both places:

    original -> tabs - own - inner - own/10. BRIKES TO DIKOSOU!! SKATA.gp5
    dup      -> tabs - own - inner - own/holdout/10. BRIKES TO DIKOSOU!! SKATA.gp5

the plain own/ copy sorts first, so the HOLDOUT copy is the one flagged dup - and fixed_gaps skips every
dup, so that holdout tab yields zero validation gaps. Copying instead of moving therefore looks like it
worked and measures nothing. inner had all nine of its holdout tabs in that state.

Two cases are handled:
  copy in own/              the same content in own/ and own/holdout/ -> the own/ copy goes
  duplicate inside holdout/ the same content twice under holdout/ under different names -> the one with
                            the longer name goes, since the extra words are usually a variant prefix
                            ("no fx ", "older - "), and both names are printed so the choice can be
                            reversed

Removed files are MOVED to data/_holdout_dups_<date>/, not unlinked: these are the user's own
compositions, and prepare already ignores any directory not starting with "tabs - " (riff.py:544), so
quarantining is enough to take them out of the corpus.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import shutil
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import riff  # noqa: E402

DATA = riff.ROOT / "data"
EXT = {".gp3", ".gp4", ".gp5"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    dig: dict[str, list[Path]] = collections.defaultdict(list)
    for d in DATA.iterdir():
        if d.is_dir() and d.name.lower().startswith("tabs - "):
            for p in d.rglob("*"):
                if p.suffix.lower() in EXT:
                    dig[hashlib.md5(p.read_bytes()).hexdigest()].append(p)

    victims: list[tuple[Path, Path, str]] = []          # (remove, keep, reason)
    for ps in dig.values():
        hold = [p for p in ps if "holdout" in p.parts]
        if not hold:
            continue
        outside = [p for p in ps if "holdout" not in p.parts]
        if outside:                                      # keep the holdout copy, drop the others
            keep = sorted(hold, key=lambda p: (len(p.name), p.name))[0]
            for p in outside:
                victims.append((p, keep, "copy in own/"))
            for p in hold:
                if p != keep:
                    victims.append((p, keep, "duplicate inside holdout/"))
        elif len(hold) > 1:                              # same content twice inside holdout/
            keep = sorted(hold, key=lambda p: (len(p.name), p.name))[0]
            for p in hold:
                if p != keep:
                    victims.append((p, keep, "duplicate inside holdout/"))

    if not victims:
        print("nothing to do: every holdout tab is already unique")
        return 0
    # relative_to(DATA) first: DATA is absolute, so parts[1] was "home"
    by_album = collections.Counter(v[0].relative_to(DATA).parts[0]
                                   .replace("tabs - own - ", "").replace(" - own", "")
                                   for v in victims)
    print(f"{len(victims)} files to remove, leaving each tab only in holdout/")
    print("  by album: " + ", ".join(f"{k}={v}" for k, v in by_album.most_common()))
    print()
    for rm, keep, why in sorted(victims, key=lambda v: str(v[0]))[:40]:
        print(f"  [{why:26s}] remove {str(rm.relative_to(DATA))[:62]}")
        print(f"   {'':28s}   keep   {str(keep.relative_to(DATA))[:62]}")
    if not a.apply:
        print("\ndry run; pass --apply")
        return 0

    quar = DATA / f"_holdout_dups_{date.today():%Y%m%d}"
    n = 0
    for rm, _keep, _why in victims:
        if not rm.is_file():
            continue
        dst = quar / rm.relative_to(DATA)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            continue
        shutil.move(str(rm), str(dst))
        n += 1
    print(f"\nmoved {n} files to {quar}")
    print("they are out of the corpus; `rm -rf` that folder to make it permanent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
