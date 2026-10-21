"""Build processed stores.

    python -m riffgen.dataset.preprocess dadagp --src data/dadaGP --out data/processed/dadagp --workers 12
    python -m riffgen.dataset.preprocess album configs/albums/compound_structure.yaml

DadaGP: every ``*.tokens.txt`` is used (with the GP file next to it for markers, time
signatures, track names and real tunings when present). If the download has only GP files,
they are encoded with the DadaGP encoder (D20).

Album: all GP3/4/5 files below the album's folders; every guitar must be in 7-string Drop A
(D10). Holdout songs and all versions of them (``song__v2.gp5``) are marked ``holdout``.

Outputs next to the store: ``skipped.csv``, ``filters.csv`` (D18), ``stats.json``.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import zlib
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

from riffgen import REPO_DIR
from riffgen.dataset import pipeline as pl
from riffgen.dataset.store import StoreWriter, song_to_arrays
from riffgen.gp.reader import SkipSong
from riffgen.tokens.vocab import Vocab, default_vocab
from riffgen.utils.config import load_config
from riffgen.utils.log import get_logger

log = get_logger("riffgen.preprocess")

_WORKER: dict = {}


def find_dadagp_root(src: str | Path) -> Path:
    """data/dadaGP vs data/DadaGP: accept either spelling."""
    p = Path(src)
    if p.is_dir():
        return p
    parent = p.parent if p.parent.is_dir() else REPO_DIR / p.parent
    for c in parent.iterdir() if parent.is_dir() else []:
        if c.is_dir() and c.name.lower() == p.name.lower():
            return c
    raise FileNotFoundError(f"DadaGP folder not found: {src}")


def _init_worker(cfg_dict: dict) -> None:
    _WORKER["cfg"] = cfg_dict
    _WORKER["vocab"] = default_vocab()


def _audit_rows(song, rel: str) -> list[dict]:
    rows = []
    for seg in song.analysis.get("segments", []):
        if seg["kind"] != "music":
            continue
        if seg["action"] == "keep":
            continue
        cx = seg["scores"].get("complexity", {})
        orig = sorted({song.bars[i].orig + 1 for i in range(seg["start"], seg["end"])})
        rows.append({
            "song": rel, "bars": f"{seg['start'] + 1}-{seg['end']}", "gp_bars": f"{orig[0]}-{orig[-1]}" if orig else "",
            "action": seg["action"], "reason": seg["reason"], "pitch_complexity": cx.get("pitch"),
            "rhythm_complexity": cx.get("rhythm"), "section": seg.get("section") or "",
        })
    return rows


def _work(job: tuple) -> tuple:
    kind, path, root, extra = job
    cfg: pl.PipelineConfig = _WORKER["cfg"]["pipeline"]
    vocab: Vocab = _WORKER["vocab"]
    try:
        if kind == "tokens":
            song = pl.process_dadagp_file(path, root, cfg)
        else:
            song = pl.process_gp_file(path, root, cfg, kind=extra.get("song_kind", "own"),
                                      family_group=extra.get("family_group", "own"))
            if extra.get("artist_from_folder"):
                rel = Path(path).relative_to(root)
                song.artist = rel.parts[-2] if len(rel.parts) >= 2 else song.artist
                song.family = pl.family_key(song.artist, Path(path).stem)
        if not song.bars or not any(b.has_guitar_notes() for b in song.bars):
            raise SkipSong("no guitar notes after processing")
        rel = song.song_id
        bars, events, tokens, meta = song_to_arrays(song, vocab)
        meta["set"] = extra.get("set", "")
        audit = _audit_rows(song, rel)
        export = None
        if extra.get("export_dir"):
            export = (song.bars, song.tracks, song.tempo)
        return ("ok", rel, (bars, events, tokens, meta), audit, export)
    except SkipSong as e:
        return ("skip", str(path), str(e), None, None)
    except Exception as e:  # keep going on broken files, but log them
        return ("error", str(path), f"{type(e).__name__}: {e}", None, None)


def run_jobs(jobs: list[tuple], cfg: pl.PipelineConfig, workers: int, desc: str):
    cfg_dict = {"pipeline": cfg}
    if workers <= 1:
        _init_worker(cfg_dict)
        for job in tqdm(jobs, desc=desc):
            yield job, _work(job)
        return
    import multiprocessing as mp
    # fork() after torch started its threads can deadlock; spawn is safe (and what Windows does anyway)
    ctx = mp.get_context("spawn") if "torch" in sys.modules else None
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(cfg_dict,),
                             mp_context=ctx) as ex:
        futs = {ex.submit(_work, job): job for job in jobs}
        for fut in tqdm(as_completed(futs), total=len(futs), desc=desc):
            yield futs[fut], fut.result()


def write_store(out: Path, results: list, vocab: Vocab, split_fn, info: dict) -> StoreWriter:
    writer = StoreWriter(out, vocab, info)
    results.sort(key=lambda r: r[1])
    for _, rel, (bars, events, tokens, meta), _, _ in results:
        meta["split"] = split_fn(meta)
        meta.update(n_bars=len(bars), bar_off=writer._n_bars, ev_off=writer._n_events, tok_off=writer._n_tokens,
                    n_events=len(events), n_tokens=len(tokens))
        writer._bars.append(bars)
        writer._events.append(events)
        writer._tokens.append(tokens)
        writer._meta.append(meta)
        writer._n_bars += len(bars)
        writer._n_events += len(events)
        writer._n_tokens += len(tokens)
    writer.close()
    return writer


def write_reports(out: Path, results: list, skipped: list, audit_csv: Path | None = None) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "skipped.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "status", "reason"])
        for status, path, reason in skipped:
            w.writerow([path, status, reason])
    audit = [row for r in results for row in (r[3] or [])]
    for path in [out / "filters.csv"] + ([audit_csv] if audit_csv else []):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            fields = ["song", "bars", "gp_bars", "action", "reason", "pitch_complexity", "rhythm_complexity",
                      "section"]
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(audit)
    # statistics (D27: tokens per bar, how many bars fit into the context)
    tpb, tunings, ts, n_bars, gaps, ctx, drops = [], Counter(), Counter(), [], 0, 0, 0
    for r in results:
        bars, events, tokens, meta = r[2]
        n_bars.append(len(bars))
        for t in meta["tracks"].values():
            from riffgen.tokens import tunings as T
            tunings[T.name_of(tuple(t["tuning"]))] += 1
        ev_counts = bars["ev_count"]
        starts = bars["ev_start"]
        for b in range(len(bars)):
            if bars[b]["gap"]:
                gaps += 1
                continue
            evs = events[starts[b]: starts[b] + ev_counts[b]]
            n_ticks = len(set(evs["tick"].tolist())) if len(evs) else 0
            tpb.append(1 + int(evs["tok_len"].sum()) + max(1, n_ticks))
            ts[f"{bars[b]['ts_num']}/{bars[b]['ts_den']}"] += 1
            ctx += int(bars[b]["ctx"])
            drops += int(bars[b]["drop"])
    tpb = np.array(tpb or [0])
    long_range = long_range_stats(results)
    stats = {
        "songs": len(results), "skipped": Counter(s[0] for s in skipped),
        "skip_reasons": Counter(s[2].split(":")[0][:60] for s in skipped).most_common(15),
        "bars": int(sum(n_bars)), "bars_per_song_median": float(np.median(n_bars)) if n_bars else 0,
        "tokens_per_bar": {"mean": float(tpb.mean()), "p50": float(np.percentile(tpb, 50)),
                           "p90": float(np.percentile(tpb, 90)), "p99": float(np.percentile(tpb, 99))},
        "bars_in_4096": float(4096 / max(tpb.mean(), 1)), "bars_in_8192": float(8192 / max(tpb.mean(), 1)),
        "gap_bars": gaps, "context_bars": ctx, "dropped_bars": drops,
        "tunings": tunings.most_common(30), "time_signatures": ts.most_common(20),
        "long_context": long_range,
    }
    from riffgen.utils.config import save_json
    save_json(stats, out / "stats.json")
    return stats


def long_range_stats(results: list, far: int = 32, edge: int = 16) -> dict:
    """How long are whole songs in tokens, and how often does music return from far away?

    * song length in tokens (estimate: event tokens + one wait per onset + new_measure);
    * share of songs that fit into 4k/8k/16k/32k tokens;
    * long-range returns: bars whose repeat partner (similarity >= 0.9) is >= ``far`` bars away;
    * bookends: a bar of the first ``edge`` bars returns in the last ``edge`` bars (intro riff at the end).
    """
    lengths, far_bars, all_bars, songs_far = [], 0, 0, 0
    for r in results:
        bars, events, tokens, meta = r[2]
        n = len(bars)
        total = 0
        for b in range(n):
            evs = events[bars[b]["ev_start"]: bars[b]["ev_start"] + bars[b]["ev_count"]]
            total += 1 + int(evs["tok_len"].sum()) + max(1, len(set(evs["tick"].tolist())) if len(evs) else 1)
        lengths.append(total)
        partners = bars["partners"]
        has_far = False
        for i in range(n):
            ps = [int(j) for j in partners[i] if j >= 0]
            if ps:
                all_bars += 1
            if any(abs(j - i) >= far for j in ps):
                far_bars += 1
                has_far = True
        songs_far += has_far
    L = np.array(lengths or [0])
    fits = {f"fits_{k // 1024}k": float(np.mean(L <= k)) for k in (4096, 8192, 16384, 32768)}
    n_songs = max(1, len(results))
    return {
        "song_tokens": {"p50": float(np.percentile(L, 50)), "p90": float(np.percentile(L, 90)),
                        "p99": float(np.percentile(L, 99)), "max": int(L.max())},
        **fits,
        f"songs_with_returns_beyond_{far}_bars": songs_far / n_songs,
        f"repeated_bars_with_partner_beyond_{far}_bars": far_bars / max(1, all_bars),
        "songs_with_bookends": _bookend_songs(results, edge) / n_songs,
    }


def _bookend_songs(results: list, edge: int) -> int:
    count = 0
    for r in results:
        bars = r[2][0]
        n = len(bars)
        if n < 3 * edge:
            continue
        p = bars["partners"][:edge]
        if np.any((p >= n - edge) & (p >= 0)):
            count += 1
    return count


def export_gp5(results: list, export_dir: Path) -> None:
    from riffgen.gp import writer
    for r in results:
        if r[4] is None:
            continue
        bars, tracks, tempo = r[4]
        dst = export_dir / (r[1] + ".normalized.gp5")
        try:
            writer.write_bars(dst, bars, tracks, tempo, title=Path(r[1]).stem)
        except Exception as e:
            log.warning("normalized export failed for %s: %s", r[1], e)


def export_audit_snippets(results: list, audit_dir: Path, max_files: int = 400) -> None:
    """D18: a short GP5 per filtered segment, to listen to what the filters did."""
    from riffgen.gp import writer
    n = 0
    for r in results:
        if r[4] is None or not r[3]:
            continue
        bars, tracks, tempo = r[4]
        for row in r[3]:
            if n >= max_files:
                return
            a, b = (int(x) for x in row["bars"].split("-"))
            seg = bars[a - 1: b]
            name = f"{Path(r[1]).stem}__bars{row['gp_bars']}__{row['action']}.gp5".replace("/", "_")
            try:
                writer.write_bars(audit_dir / name, seg, tracks, tempo, title=f"{r[1]} {row['gp_bars']} {row['reason']}")
                n += 1
            except Exception as e:
                log.warning("audit export failed for %s: %s", name, e)


# ---------------------------------------------------------------------------

def cmd_dadagp(args) -> None:
    src = find_dadagp_root(args.src)
    out = Path(args.out)
    vocab = default_vocab()
    token_files = sorted(src.rglob("*.tokens.txt"))
    jobs: list[tuple] = []
    if token_files:
        jobs = [("tokens", p, src, {}) for p in token_files]
        log.info("found %d DadaGP token files under %s", len(jobs), src)
    else:
        gp_files = sorted(p for ext in ("*.gp3", "*.gp4", "*.gp5") for p in src.rglob(ext) if ".pygp." not in p.name)
        jobs = [("gp", p, src, {"song_kind": "dadagp", "artist_from_folder": True}) for p in gp_files]
        log.info("no token files; encoding %d GP files with the DadaGP encoder", len(jobs))
    if args.limit:
        rng = np.random.default_rng(0)
        jobs = [jobs[i] for i in sorted(rng.choice(len(jobs), min(args.limit, len(jobs)), replace=False))]
    cfg = pl.PipelineConfig(use_gp_files=not args.no_gp, require_tuning=None)
    results, skipped = [], []
    for job, res in run_jobs(jobs, cfg, args.workers, "dadagp"):
        (results if res[0] == "ok" else skipped).append(res if res[0] == "ok" else (res[0], res[1], res[2]))
    val_fraction = args.val_fraction

    def split_fn(meta):
        h = zlib.crc32(meta["family"].encode("utf-8")) % 10000
        return "val" if h < val_fraction * 10000 else "train"

    write_store(out, results, vocab, split_fn, {"source": "dadagp", "src": str(src)})
    stats = write_reports(out, results, skipped)
    if args.export_gp5:
        rng = np.random.default_rng(0)
        pick = [results[i] for i in rng.choice(len(results), min(args.export_gp5, len(results)), replace=False)]
        export_gp5([(r[0], r[1], r[2], r[3], _reload_for_export(r[1], src, cfg)) for r in pick], out / "normalized")
    log.info("store written to %s: %d songs, %d skipped", out, len(results), len(skipped))
    log.info("tokens/bar mean %.1f p90 %.1f -> ~%.0f bars in 4096 tokens, ~%.0f in 8192",
             stats["tokens_per_bar"]["mean"], stats["tokens_per_bar"]["p90"], stats["bars_in_4096"], stats["bars_in_8192"])
    lc = stats["long_context"]
    log.info("whole songs: median %.0f tokens, p99 %.0f; fit in 4k/8k/16k/32k: %.0f%%/%.0f%%/%.0f%%/%.0f%%; "
             "songs with a riff returning >= 32 bars later: %.0f%%, with intro-outro bookends: %.0f%%",
             lc["song_tokens"]["p50"], lc["song_tokens"]["p99"], 100 * lc["fits_4k"], 100 * lc["fits_8k"],
             100 * lc["fits_16k"], 100 * lc["fits_32k"], 100 * lc["songs_with_returns_beyond_32_bars"],
             100 * lc["songs_with_bookends"])


def _reload_for_export(rel: str, src: Path, cfg: pl.PipelineConfig):
    path = src / (rel + ".tokens.txt")
    try:
        song = pl.process_dadagp_file(path, src, cfg) if path.is_file() else None
    except Exception:
        return None
    return (song.bars, song.tracks, song.tempo) if song else None


def album_paths(acfg) -> tuple[Path, list[tuple[str, Path]]]:
    root = Path(acfg.data.root)
    if not root.is_absolute():
        root = REPO_DIR / root
    sets = []
    own = acfg.data.get("own") or []
    for p in ([own] if isinstance(own, str) else own):
        sets.append(("own", p))
    for p in acfg.data.get("others") or []:
        sets.append((Path(p).name, p))
    out = []
    for name, p in sets:
        path = Path(p)
        if not path.is_absolute():
            path = root / path if (root / path).exists() else REPO_DIR / path
        out.append((name, path))
    return root, out


def holdout_families(acfg, root: Path) -> set[str]:
    fams = set()
    for h in acfg.data.get("holdout") or []:
        fams.add(pl.own_family(str(h), "x").split("|", 1)[1])
    return fams


def cmd_album(args) -> None:
    acfg = load_config(args.album)
    root, sets = album_paths(acfg)
    out = Path(args.out or REPO_DIR / "data" / "processed" / "albums" / acfg.album_id)
    jobs = []
    for set_name, folder in sets:
        if not folder.is_dir():
            log.warning("album folder missing: %s", folder)
            continue
        files = sorted(p for ext in ("*.gp3", "*.gp4", "*.gp5") for p in folder.rglob(ext))
        log.info("%s: %d files in %s", set_name, len(files), folder)
        jobs += [("gp", p, root, {"set": set_name, "family_group": acfg.album_id,
                                  "export_dir": True}) for p in files]
    if not jobs:
        log.error("no GP files found for album %s", acfg.album_id)
        sys.exit(1)
    cfg = pl.PipelineConfig.from_cfg(acfg, require_tuning=acfg.get("tuning", "drop_a7"))
    results, skipped = [], []
    for job, res in run_jobs(jobs, cfg, args.workers, acfg.album_id):
        (results if res[0] == "ok" else skipped).append(res if res[0] == "ok" else (res[0], res[1], res[2]))
    hold = holdout_families(acfg, root)

    def split_fn(meta):
        fam = meta["family"].split("|", 1)[1]
        return "holdout" if fam in hold else "train"

    write_store(out, results, default_vocab(), split_fn, {"source": "album", "album_id": acfg.album_id})
    reports = Path(acfg.get("reports_dir") or REPO_DIR / "reports")
    audit_dir = reports / f"audit_{acfg.album_id}"
    stats = write_reports(out, results, skipped, audit_csv=reports / f"filters_{acfg.album_id}.csv")
    export_gp5(results, out / "normalized")
    export_audit_snippets(results, audit_dir)
    n_hold = sum(1 for r in results if split_fn(r[2][3]) == "holdout")
    log.info("album %s: %d songs (%d holdout), %d skipped -> %s", acfg.album_id, len(results), n_hold, len(skipped), out)
    if hold and n_hold == 0:
        log.warning("none of the holdout songs %s was found; check data.holdout in the album config", sorted(hold))
    log.info("tokens/bar mean %.1f p90 %.1f; filters: %d context bars, %d dropped bars (see %s)",
             stats["tokens_per_bar"]["mean"], stats["tokens_per_bar"]["p90"], stats["context_bars"],
             stats["dropped_bars"], audit_dir)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("dadagp")
    a.add_argument("--src", default="data/dadaGP")
    a.add_argument("--out", default="data/processed/dadagp")
    a.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a.add_argument("--limit", type=int, default=0, help="process a random subset (for a quick look)")
    a.add_argument("--no-gp", action="store_true", help="ignore GP files next to the token files")
    a.add_argument("--val-fraction", type=float, default=0.02)
    a.add_argument("--export-gp5", type=int, default=0, help="export N random normalized songs as GP5")
    b = sub.add_parser("album")
    b.add_argument("album")
    b.add_argument("--out", default=None)
    b.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args(argv)
    {"dadagp": cmd_dadagp, "album": cmd_album}[args.cmd](args)


if __name__ == "__main__":
    main()
