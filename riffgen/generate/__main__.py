"""Fill the gaps of a song with candidate riffs.

    python -m riffgen.generate --album album_01 --song songs/song_03.gp5
    python -m riffgen.generate --album album_01 --song songs/song_03.gp5 --gap 33-40 --section bridge --candidates 32

One run per gap (D35); every gap gets its own folder in outputs/<album>/ with the whole song
(gap filled) per kept candidate, report.json and check_report.md/csv.
"""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import guitarpro
import torch

from riffgen import REPO_DIR
from riffgen.analysis import filters as filters_mod
from riffgen.analysis.notes import bar_onsets
from riffgen.check import plagiarism as plag
from riffgen.check import playability as play
from riffgen.check.report import write_reports
from riffgen.dataset.loader import collate
from riffgen.generate import pipeline as gp_pipe
from riffgen.generate.constraints import GapSpec
from riffgen.generate.decode import generate
from riffgen.gp import writer
from riffgen.train.checkpoint import load_model
from riffgen.train.common import repo_path, sampler_config
from riffgen.utils.config import load_config, save_json
from riffgen.utils.device import amp_dtype, describe, pick_device, to_device
from riffgen.utils.log import get_logger

log = get_logger("riffgen.generate")


def find_album_config(album: str) -> Path:
    p = Path(album)
    if p.suffix in (".yaml", ".yml"):
        return p
    return REPO_DIR / "configs" / "albums" / f"{album}.yaml"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--album", required=True, help="album id (configs/albums/<id>.yaml) or a config path")
    ap.add_argument("--song", required=True, help="GP5 file with empty bars (gaps)")
    ap.add_argument("--gap", default=None, help="bars to fill, Guitar Pro numbering, e.g. 33-40")
    ap.add_argument("--section", default=None, help="what the gap should be: " + ", ".join(
        ["intro", "verse", "chorus", "bridge", "solo", "breakdown", "interlude", "outro", "other"]))
    ap.add_argument("--model", default=None, help="baseline | big | huge (default: album config)")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--candidates", type=int, default=None)
    ap.add_argument("--keep", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--min-complexity", type=float, default=None)
    ap.add_argument("--use-gap-signatures", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--precision", default="auto", help="auto | fp32 | fp16 | bf16")
    ap.add_argument("--batch", type=int, default=None, help="candidates per batch (memory)")
    ap.add_argument("--fix", action="store_true", help="also save re-fingered versions of hard candidates")
    ap.add_argument("--no-plagiarism", action="store_true")
    ap.add_argument("--index", default=None, help="plagiarism index (default runs/<album>/plagiarism_index.sqlite)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    acfg = load_config(find_album_config(args.album))
    g = dict(acfg.get("generation") or {})
    n_cand = args.candidates or int(g.get("candidates", 32))
    keep = args.keep or int(g.get("keep", 10))
    temperature = args.temperature if args.temperature is not None else float(g.get("temperature", 1.0))
    top_p = args.top_p if args.top_p is not None else float(g.get("top_p", 0.95))
    min_cx = args.min_complexity if args.min_complexity is not None else float(g.get("min_complexity", 0.25))
    seed = args.seed if args.seed is not None else int(acfg.get("seed", 1234))
    model_name = args.model or acfg.get("model", "baseline")
    ckpt = repo_path(args.checkpoint) if args.checkpoint else REPO_DIR / "runs" / acfg.album_id / model_name / "best.pt"
    if not ckpt.is_file():
        raise SystemExit(f"no album network at {ckpt}\nFine-tune first: python -m riffgen.train.finetune {find_album_config(args.album)}")

    device = pick_device(args.device)
    model, vocab, ck = load_model(ckpt, "cpu")
    amp = amp_dtype(device, args.precision)
    model.to(device).eval()
    batch = args.batch or (32 if device.type == "cuda" and torch.cuda.get_device_properties(device).total_memory > 12e9 else 8)
    log.info("model %s (%s) on %s, amp=%s", model.cfg.name, ckpt, describe(device), amp)

    song_path = Path(args.song)
    for cand in (song_path, repo_path(args.song),
                 repo_path(acfg.get_path("data.root", ".")) / str(acfg.get_path("data.songs", "songs")) / args.song,
                 repo_path(acfg.get_path("data.root", ".")) / args.song):
        if cand.is_file():
            song_path = cand
            break
    else:
        raise SystemExit(f"song not found: {args.song}")
    from riffgen.gp.reader import SkipSong
    from riffgen.tokens import sections as sections_mod
    if args.section and args.section not in sections_mod.labels():
        raise SystemExit(f"--section must be one of: {', '.join(sections_mod.labels())}")
    try:
        song = gp_pipe.load_input_song(song_path, acfg.get("tuning", "drop_a7"), int(acfg.get("min_gap_bars", 4)))
    except SkipSong as e:
        raise SystemExit(f"{song_path.name}: {e}")
    gaps = gp_pipe.find_gaps(song, args.gap, int(acfg.get("min_gap_bars", 4)))
    if not gaps:
        raise SystemExit("no gap found (4 or more empty bars in all guitar tracks); use --gap A-B")
    log.info("%s: %d bars, guitars %s, gaps %s", song_path.name, len(song.bars),
             [t.name or t.slot for t in song.tracks.values()], [g_.label for g_ in gaps])
    data_cfg = load_config(REPO_DIR / "configs" / "train" / "finetune.yaml")
    sc = sampler_config(data_cfg, model.cfg)
    original = guitarpro.parse(str(song_path))
    slot_track = {slot: tr.orig_index for slot, tr in song.tracks.items()}
    index = None
    idx_path = repo_path(args.index) if args.index else REPO_DIR / "runs" / acfg.album_id / "plagiarism_index.sqlite"
    if not args.no_plagiarism:
        if idx_path.is_file():
            index = plag.Index(idx_path)
        else:
            log.warning("no plagiarism index at %s (built by finetune); plagiarism check skipped", idx_path)
    prof = play.load_profile()

    for gap in gaps:
        t0 = time.time()
        view, ticks, ts = gp_pipe.prepare_view(song, gap, vocab, args.use_gap_signatures)
        prompt, feat = gp_pipe.build_prompt(view, vocab, sc, gap, args.section)
        f = to_device(collate([prompt], vocab.pad_id, 1), device)
        spec = GapSpec(n_bars=gap.length, bar_ticks=ticks, bar_ts=ts,
                       slots={s: tr.n_strings for s, tr in song.tracks.items()},
                       force_ts_each_bar=args.use_gap_signatures, prev_notes=gp_pipe.last_notes_before(song, gap))
        gap_key = f"{song_path.stem}:{gap.label}"
        try:
            cands = generate(model, vocab, f, feat, spec, n_cand, seed, gap_key, temperature, top_p, batch, amp=amp,
                             device=device)
        except FloatingPointError:
            log.warning("fp16 overflow; retrying in fp32")
            cands = generate(model, vocab, f, feat, spec, n_cand, seed, gap_key, temperature, top_p, batch, amp=None,
                             device=device)
        log.info("gap %s: %d candidates in %.0fs (prompt %d tokens)", gap.label, len(cands), time.time() - t0,
                 f["ids"].shape[1])
        out_dir = Path(args.out) if args.out else gp_pipe.default_out_dir(acfg.album_id, song_path, gap)
        if len(gaps) > 1 and args.out:
            out_dir = out_dir / f"gap_{gap.label}"
        out_dir.mkdir(parents=True, exist_ok=True)
        results = []
        for c in cands:
            bars = gp_pipe.candidate_bars(c, vocab, song, ticks, ts)
            filled = gp_pipe.song_with_bars(song, gap, bars)
            rng = range(gap.start, gap.start + gap.length)
            r = {"candidate": c.index, "seed": c.seed, "mean_logprob": round(c.mean_logprob, 4),
                 "invalid_mass": round(c.invalid_mass, 4), "finished": c.finished, "tokens": len(c.tokens)}
            r["neighbour_copy"] = round(gp_pipe.neighbour_copy_score(filled, gap), 3)
            cx = filters_mod.complexity({s: gap_onsets(filled, s, rng) for s in filled.tracks}, 0, gap.length,
                                        [b.ticks for b in bars])
            r["complexity"] = cx["complexity"]
            r["pitch_complexity"], r["rhythm_complexity"] = cx["pitch"], cx["rhythm"]
            r["odd_grid"] = gp_pipe.grid_flag(bars)
            if index is not None:
                m = plag.check_song(index, filled, rng)
                r.update(plagiarism=m.label(), level=m.level, source=m.source, source_bars=m.source_bars,
                         cand_bars=m.cand_bars, run_bars=m.run_bars, coverage=m.coverage, rhythm_like=m.rhythm_source)
            else:
                r["plagiarism"] = "not checked"
            rep = play.check_song(filled, rng, prof)
            r.update(playability=round(rep.score, 1), errors=rep.errors, warnings=rep.warnings)
            reasons = []
            if not c.finished:
                reasons.append("unfinished")
            if r["neighbour_copy"] >= float((acfg.get("generation") or {}).get("max_neighbour_similarity", 0.95)):
                reasons.append("repeats the riff next to the gap")
            if r["complexity"] < min_cx:
                reasons.append("too simple")
            if r["plagiarism"].startswith("RED") and "SELF" not in r["plagiarism"]:
                reasons.append("plagiarism RED")
            r["removed"] = "; ".join(reasons)
            r["score"] = round(r["mean_logprob"] + 0.5 * r["playability"] / 100 + 0.3 * r["complexity"]
                               - (0.3 if r["plagiarism"].startswith(("YELLOW", "SELF")) else 0.0)
                               - (0.2 if r["odd_grid"] else 0.0), 4)
            results.append((r, bars, filled, rep))
        kept = sorted([x for x in results if not x[0]["removed"]], key=lambda x: -x[0]["score"])[:keep]
        rows, issues = [], {}
        for rank, (r, bars, filled, rep) in enumerate(kept, 1):
            name = f"cand_{rank:02d}.gp5"
            spliced = writer.splice(original, gap.orig_measures, bars, slot_track,
                                    marker=f"RiffGen #{rank}" + (f" ({args.section})" if args.section else ""))
            writer.write_song(spliced, out_dir / name)
            r["file"] = name
            fixed = ""
            if args.fix and rep.errors:
                fbars = copy.deepcopy(bars)          # re-fingering edits event tokens in place
                fsong = gp_pipe.song_with_bars(song, gap, fbars)
                if play.fix_song(fsong, range(gap.start, gap.start + gap.length), prof):
                    fixed = name.replace(".gp5", ".fixed.gp5")
                    writer.write_song(writer.splice(original, gap.orig_measures, fbars, slot_track), out_dir / fixed)
            rows.append({"file": name, **{k: r.get(k, "") for k in ("plagiarism", "level", "source", "source_bars",
                                                                      "cand_bars", "run_bars", "coverage", "rhythm_like",
                                                                      "playability", "errors", "warnings")},
                         "fixed_file": fixed, "notes": f"score {r['score']}, complexity {r['complexity']}"})
            issues[name] = rep.issues
        report = {
            "song": str(song_path), "gap": gap.label, "gap_bars_unrolled": [gap.start, gap.start + gap.length],
            "section": args.section, "model": model.cfg.name, "checkpoint": str(ckpt), "seed": seed,
            "temperature": temperature, "top_p": top_p, "candidates": n_cand, "kept": len(kept),
            "time_signature": ts[0] if ts else None, "tempo": song.bars[max(0, gap.start - 1)].tempo,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "results": [r for r, *_ in sorted(results, key=lambda x: -x[0]["score"])],
            "regenerate": f"python -m riffgen.generate --album {args.album} --song {args.song} --gap {gap.label} "
                          f"--seed {seed} --temperature {temperature} --top-p {top_p} --candidates {n_cand}",
        }
        save_json(report, out_dir / "report.json")
        write_reports(out_dir, rows, f"{song_path.name}, gap {gap.label}", issues)
        removed = [r for r, *_ in results if r["removed"]]
        log.info("gap %s: kept %d of %d -> %s (removed: %s)", gap.label, len(kept), len(results), out_dir,
                 _count_reasons(removed))
    if index is not None:
        index.close()


def gap_onsets(song, slot, rng):
    tr = song.tracks[slot]
    return [bar_onsets(song.bars[b], slot, tr.tuning) for b in rng]


def _count_reasons(removed: list[dict]) -> str:
    from collections import Counter
    c = Counter(x for r in removed for x in r["removed"].split("; ") if x)
    return ", ".join(f"{k}: {v}" for k, v in c.items()) or "none"


if __name__ == "__main__":
    main()
