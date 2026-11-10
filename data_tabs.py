"""Back up and restore the tab collection (data/tabs - */, about 57k Guitar Pro files). The tabs are not in git: they are other
people's transcriptions. They live as a few zip files in a (private) Google Drive folder instead.

    python data_tabs.py pack                  # data/tabs - */  ->  data_encoded/tabs_backup/tabs_part001.zip, 002, ... (~66 MB each) + tabs_manifest.json
    python data_tabs.py fetch                 # Drive folder -> data_encoded/tabs_backup/, unzipped into data/ (skips files that exist)
    python data_tabs.py fetch --from DIR      # the same from zip files you already downloaded by hand into DIR
    python data_tabs.py verify                # the file counts per folder against tabs_manifest.json

``fetch`` needs ``pip install gdown`` and a Drive folder shared as "anyone with the link" (DRIVE_FOLDER below, or
--url). Upload the files ``pack`` wrote (every zip and tabs_manifest.json) to that folder after packing.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
BACKUP = ROOT / "data_encoded" / "tabs_backup"
MANIFEST = "tabs_manifest.json"
DRIVE_FOLDER = "https://drive.google.com/drive/folders/12636du_4adjeTp9BEyaGzSNEqL9PIedN?usp=sharing"
EXT = (".gp3", ".gp4", ".gp5", ".gpx", ".gp")


def folders() -> list[Path]:
    """Every tab folder: the genre folders and the own-album folders ('tabs - own - <album> - own|insp')."""
    return sorted(d for d in DATA.iterdir() if d.is_dir() and d.name.lower().startswith("tabs - "))


def tab_files(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in EXT)


def counts() -> dict[str, int]:
    return {d.name: len(tab_files(d)) for d in folders()}


def pack(max_mb: float) -> None:
    """Zip files of about ``max_mb`` MB each, splitting anywhere."""
    BACKUP.mkdir(parents=True, exist_ok=True)
    for old in BACKUP.glob("tabs_part*.zip"):
        old.unlink()
    limit = int(max_mb * 1e6)
    names: list[str] = []
    z = None
    for d in folders():
        for p in tab_files(d):
            if z is None or z.fp.tell() > limit:
                if z is not None:
                    z.close()
                names.append(f"tabs_part{len(names) + 1:03d}.zip")
                z = zipfile.ZipFile(BACKUP / names[-1], "w", zipfile.ZIP_DEFLATED, compresslevel=6)
            z.write(p, p.relative_to(DATA).as_posix())
    if z is not None:
        z.close()
    total = sum((BACKUP / n).stat().st_size for n in names)
    (BACKUP / MANIFEST).write_text(json.dumps({"parts": len(names), "counts": counts()}, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{sum(counts().values())} tabs in {len(names)} zips, {total / 1e6:.0f} MB, largest {max((BACKUP / n).stat().st_size for n in names) / 1e6:.1f} MB -> {BACKUP}")
    print(f"upload every tabs_part*.zip and {MANIFEST} to the Drive folder")


def unzip_all(src: Path) -> int:
    n = 0
    for zp in sorted(src.glob("tabs_part*.zip")):
        with zipfile.ZipFile(zp) as z:
            for info in z.infolist():
                dst = (DATA / info.filename).resolve()
                if DATA.resolve() not in dst.parents:               # never write outside data/
                    raise SystemExit(f"unsafe path in {zp.name}: {info.filename}")
                if dst.exists():
                    continue
                dst.parent.mkdir(parents=True, exist_ok=True)
                with z.open(info) as s, open(dst, "wb") as o:
                    o.write(s.read())
                n += 1
        print(f"{zp.name}: done")
    return n


def verify() -> bool:
    mp = BACKUP / MANIFEST
    if not mp.is_file():
        print(f"no {MANIFEST} in {BACKUP}"); return False
    want = json.loads(mp.read_text(encoding="utf-8"))["counts"]
    have = counts()
    bad = {k: (want[k], have.get(k, 0)) for k in want if have.get(k, 0) < want[k]}
    print(f"{sum(have.values())} tabs in data/ (backup says {sum(want.values())})")
    for k, (w, h) in bad.items():
        print(f"  missing in {k}: {w - h} (backup {w}, here {h})")
    return not bad


def fetch(url: str, src: str | None) -> None:
    if src is None:
        try:
            import gdown
        except ImportError:
            raise SystemExit("pip install gdown  (or download the zips by hand and use: fetch --from DIR)")
        BACKUP.mkdir(parents=True, exist_ok=True)
        gdown.download_folder(url=url, output=str(BACKUP), quiet=False, use_cookies=False)
        folder = BACKUP
    else:
        folder = Path(src)
        if folder.resolve() != BACKUP.resolve():
            BACKUP.mkdir(parents=True, exist_ok=True)
            if (folder / MANIFEST).is_file():
                (BACKUP / MANIFEST).write_bytes((folder / MANIFEST).read_bytes())
    DATA.mkdir(exist_ok=True)
    print(f"unzipped {unzip_all(folder)} new files into {DATA}")
    verify()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pack"); p.add_argument("--max-mb", type=float, default=66.0, help="size limit of one zip in MB")
    p = sub.add_parser("fetch"); p.add_argument("--url", default=DRIVE_FOLDER); p.add_argument("--from", dest="src", default=None)
    sub.add_parser("verify")
    a = ap.parse_args()
    if a.cmd == "pack":
        pack(a.max_mb)
    elif a.cmd == "fetch":
        fetch(a.url, a.src)
    else:
        sys.exit(0 if verify() else 1)


if __name__ == "__main__":
    main()
