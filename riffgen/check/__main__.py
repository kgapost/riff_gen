"""Plagiarism and playability checks on any folder of GP5 files (D42).

    python -m riffgen.check --album album_01 --folder outputs/album_01/2026-10-18_song_03_gap_33-40 \
        --plagiarism --playability --fix --mark
    python -m riffgen.check --album album_01 --build-index [--extra-reference PATH]
    python -m riffgen.check --build-dadagp-index        (optional, all of DadaGP, D40)

In a generation folder (with report.json) only the generated bars are checked; elsewhere
the whole file. Writes check_report.md and check_report.csv into the folder.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import guitarpro

from riffgen import CONFIG_DIR, REPO_DIR
from riffgen.check import plagiarism as plag
from riffgen.check import playability as play
from riffgen.check.report import write_reports
from riffgen.dataset import pipeline as pl
from riffgen.gp import writer
from riffgen.gp.reader import SkipSong
from riffgen.utils.config import load_config
from riffgen.utils.log import get_logger

log = get_logger("riffgen.check")


def album_cfg(album: str):
    p = Path(album)
    return load_config(p if p.suffix else REPO_DIR / "configs" / "albums" / f"{album}.yaml")


def checked_bars(folder: Path, song) -> range | None:
    rep = folder / "report.json"
    if rep.is_file():
        a, b = json.loads(rep.read_text())["gap_bars_unrolled"]
        if b <= len(song.bars):
            return range(a, b)
    return None


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--album", default=None)
    ap.add_argument("--folder", default=None)
    ap.add_argument("--plagiarism", action="store_true")
    ap.add_argument("--playability", action="store_true")
    ap.add_argument("--fix", action="store_true", help="write <file>.fixed.gp5 with a better fingering")
    ap.add_argument("--mark", action="store_true", help="write <file>.marked.gp5 with markers on matching bars")
    ap.add_argument("--reference", default="album", help="album | dadagp")
    ap.add_argument("--extra-reference", action="append", default=[])
    ap.add_argument("--build-index", action="store_true")
    ap.add_argument("--build-dadagp-index", action="store_true")
    ap.add_argument("--profile", default=None, help="playability profile yaml")
    args = ap.parse_args(argv)

    if args.build_dadagp_index:
        build_dadagp_index()
        return
    acfg = album_cfg(args.album) if args.album else None
    if args.build_index:
        if acfg is None:
            raise SystemExit("--build-index needs --album")
        out = REPO_DIR / "runs" / acfg.album_id / "plagiarism_index.sqlite"
        log.info("built %s", plag.build_album_index(acfg, out, args.extra_reference))
        return
    if not args.folder:
        raise SystemExit("--folder is required")
    if not (args.plagiarism or args.playability):
        args.plagiarism = args.playability = True
    folder = Path(args.folder)
    files = sorted(p for p in folder.glob("*.gp5") if not p.name.endswith((".fixed.gp5", ".marked.gp5")))
    index = None
    if args.plagiarism:
        if args.reference == "dadagp":
            path = REPO_DIR / "data" / "processed" / "dadagp_index.sqlite"
        else:
            if acfg is None:
                raise SystemExit("--plagiarism needs --album (or --reference dadagp)")
            path = REPO_DIR / "runs" / acfg.album_id / "plagiarism_index.sqlite"
        if args.extra_reference:
            path = folder / "_extra_index.sqlite"
            log.info("building index with extra references: %s",
                     plag.build_album_index(acfg, path, args.extra_reference) if acfg else "")
        if not path.is_file():
            raise SystemExit(f"no plagiarism index at {path}; run --build-index first")
        index = plag.Index(path)
    prof = play.load_profile(args.profile)
    tuning = acfg.get("tuning", "drop_a7") if acfg else "drop_a7"
    rows, issues = [], {}
    for p in files:
        row = {"file": p.name}
        try:
            song = pl.process_gp_file(p, folder, pl.PipelineConfig(analyze=False, require_tuning=None), kind="check")
        except SkipSong as e:
            rows.append({**row, "notes": f"cannot read: {e}"})
            continue
        bars = checked_bars(folder, song)
        if index is not None:
            m = plag.check_song(index, song, bars)
            row.update(plagiarism=m.label(), level=m.level, source=m.source, source_bars=m.source_bars,
                       cand_bars=m.cand_bars, run_bars=m.run_bars, coverage=m.coverage, rhythm_like=m.rhythm_source)
            if args.mark and m.verdict != "GREEN":
                gp_song = guitarpro.parse(str(p))
                a = int(m.cand_bars.split("-")[0]) - 1 if m.cand_bars else 0
                orig = song.bars[(bars.start if bars else 0) + a].orig if song.bars else 0
                gp_song.measureHeaders[orig].marker = guitarpro.Marker(
                    title=f"COPY? {Path(m.source).name} bars {m.source_bars} ({m.level})")
                writer.write_song(gp_song, p.with_suffix(".marked.gp5"))
        if args.playability:
            rep = play.check_song(song, bars, prof, require_tuning=tuning)
            gp_song = guitarpro.parse(str(p))
            tracks = [t.orig_index for t in song.tracks.values() if t.orig_index >= 0]
            rep.issues += play.gp_bar_length_issues(gp_song, tracks)
            row.update(playability=round(rep.score, 1), errors=rep.errors, warnings=rep.warnings)
            issues[p.name] = rep.issues
            if args.fix and rep.errors:
                n = play.fix_song(song, bars, prof)
                if n:
                    idx = list(bars) if bars else list(range(len(song.bars)))
                    first = {}
                    for i in idx:
                        first.setdefault(song.bars[i].orig, i)
                    order = sorted(first)
                    fixed = writer.splice(gp_song, order, [song.bars[first[o]] for o in order],
                                          {s: t.orig_index for s, t in song.tracks.items()})
                    out = p.with_suffix(".fixed.gp5")
                    writer.write_song(fixed, out)
                    row["fixed_file"] = out.name
                    row["notes"] = f"{n} notes re-fingered"
        rows.append(row)
    write_reports(folder, rows, f"Check of {folder}", issues)
    for r in rows:
        log.info("%-24s %-22s playability %-6s errors %-3s warnings %s", r["file"], r.get("plagiarism", "-"),
                 r.get("playability", "-"), r.get("errors", "-"), r.get("warnings", "-"))
    log.info("reports: %s", folder / "check_report.md")


def build_dadagp_index() -> None:
    from riffgen.dataset.store import Store, song_from_view
    cfg = load_config(CONFIG_DIR / "plagiarism.yaml")
    store = Store(REPO_DIR / "data" / "processed" / "dadagp")
    vocab = store.load_vocab()
    out = REPO_DIR / "data" / "processed" / "dadagp_index.sqlite"
    if out.exists():
        out.unlink()
    idx = plag.Index(out)
    from tqdm import tqdm
    for i in tqdm(range(len(store)), desc="dadagp index"):
        idx.add_song(song_from_view(store.view(i), vocab), own=False, cfg=cfg)
    idx.finalize(cfg)
    idx.close()
    log.info("DadaGP index: %s (%.1f GB)", out, out.stat().st_size / 2**30)


if __name__ == "__main__":
    main()
