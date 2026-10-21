"""Evaluate a model on fixed gaps (README §6.4).

    python -m riffgen.eval --checkpoint runs/album_01/big/best.pt --store data/processed/albums/album_01 --split holdout
    python -m riffgen.eval --checkpoint runs/pretrain_big/best.pt --store data/processed/dadagp --split val --gaps 40

Reports
* teacher-forced loss on the gaps, split by token type (note / wait / fx / bar ...);
* generation: format adherence (probability mass on forbidden tokens), finished rate,
  "copies the riff next to the gap" rate, too-simple rate, similarity of pitch classes and
  rhythm to the real (hidden) bars, playability, plagiarism verdicts (with --index);
* --listen DIR: a blind listening set -- for every gap the real bars and the candidates as
  GP5 files with neutral names, shuffled, plus key.json. Rate them before looking at the key.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from riffgen import REPO_DIR
from riffgen.analysis import filters as filters_mod
from riffgen.analysis.notes import bar_onsets, bar_vectors
from riffgen.check import plagiarism as plag
from riffgen.check import playability as play
from riffgen.dataset.loader import build_fixed_examples, collate
from riffgen.dataset.sampler import ExampleBuilder, holdout_specs, make_val_specs
from riffgen.dataset.store import Store, song_from_view
from riffgen.eval.metrics import evaluate_examples
from riffgen.generate import pipeline as gp_pipe
from riffgen.generate.constraints import GapSpec
from riffgen.generate.decode import generate
from riffgen.gp import writer
from riffgen.train.checkpoint import load_model
from riffgen.train.common import repo_path, sampler_config
from riffgen.utils.config import load_config, save_json
from riffgen.utils.device import amp_dtype, pick_device, to_device
from riffgen.utils.log import get_logger

log = get_logger("riffgen.eval")


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


def gap_profile(song, rng: range) -> tuple[np.ndarray, np.ndarray]:
    """Pitch-class histogram and onset grid of a range of bars (all guitars)."""
    pc, on = np.zeros(12), np.zeros(48)
    for slot, tr in song.tracks.items():
        for b in rng:
            r, _, h, _ = bar_vectors(bar_onsets(song.bars[b], slot, tr.tuning), song.bars[b].ticks)
            pc += h
            on += r
    return pc, on


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--store", required=True)
    ap.add_argument("--split", default="holdout")
    ap.add_argument("--gaps", type=int, default=30, help="gaps used for generation metrics")
    ap.add_argument("--candidates", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--index", default=None, help="plagiarism index (sqlite)")
    ap.add_argument("--listen", default=None, help="write a blind listening set here")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default=None, help="metrics json (default next to the checkpoint)")
    args = ap.parse_args(argv)

    device = pick_device(args.device)
    model, vocab, _ = load_model(repo_path(args.checkpoint), device)
    model.eval()
    amp = amp_dtype(device)
    store = Store(repo_path(args.store))
    sc = sampler_config(load_config(REPO_DIR / "configs" / "train" / "finetune.yaml"), model.cfg)
    builder = ExampleBuilder(vocab, sc)
    views = [(i, store.view(i)) for i in store.indices(args.split)]
    if not views:
        raise SystemExit(f"no songs with split '{args.split}' in {store.path}")
    specs = holdout_specs(views, builder) if args.split == "holdout" else make_val_specs(views, builder, args.seed, 200, 0)
    examples = build_fixed_examples(specs, store, vocab, sc)
    res = {"teacher_forced": evaluate_examples(model, examples, vocab, device, amp)}
    log.info("teacher-forced: %s", {k: round(v, 4) for k, v in res["teacher_forced"].items() if isinstance(v, float)})

    index = plag.Index(repo_path(args.index)) if args.index else None
    prof = play.load_profile()
    rng = np.random.default_rng(args.seed)
    pick = rng.choice(len(specs), size=min(args.gaps, len(specs)), replace=False) if specs else []
    stats = Counter()
    sims_pc, sims_on, cx_gen, cx_real, plays, inval, uniq = [], [], [], [], [], [], []
    listen = []
    for n, si in enumerate(pick):
        spec = specs[int(si)]
        view = store.view(store.by_id[spec["song"]])
        song = song_from_view(view, vocab)
        gap = gp_pipe.Gap(spec["start"], spec["len"], [song.bars[b].orig for b in range(spec["start"], spec["start"] + spec["len"])])
        real_rng = range(gap.start, gap.start + gap.length)
        pc_real, on_real = gap_profile(song, real_rng)
        cxr = filters_mod.complexity({s: [bar_onsets(song.bars[b], s, t.tuning) for b in real_rng] for s, t in song.tracks.items()},
                                     0, gap.length, [song.bars[b].ticks for b in real_rng])["complexity"]
        v2, ticks, ts = gp_pipe.prepare_view(song, gap, vocab, use_gap_signatures=True)
        prompt, feat = gp_pipe.build_prompt(v2, vocab, sc, gap, None)
        gspec = GapSpec(gap.length, ticks, ts, {s: t.n_strings for s, t in song.tracks.items()}, force_ts_each_bar=True,
                        max_fret=29, prev_notes=gp_pipe.last_notes_before(song, gap))
        cands = generate(model, vocab, to_device(collate([prompt], vocab.pad_id, 1), device), feat, gspec,
                         args.candidates, args.seed, f"eval:{spec['song']}:{gap.start}", args.temperature, args.top_p,
                         batch_size=args.candidates, amp=amp, device=device)
        seen = set()
        for c in cands:
            stats["candidates"] += 1
            stats["finished"] += c.finished
            inval.append(c.invalid_mass)
            seen.add(tuple(c.tokens))
            bars = gp_pipe.candidate_bars(c, vocab, song, ticks, ts)
            filled = gp_pipe.song_with_bars(song, gap, bars)
            pc, on = gap_profile(filled, real_rng)
            sims_pc.append(_cos(pc, pc_real))
            sims_on.append(_cos(on, on_real))
            cx = filters_mod.complexity({s: [bar_onsets(filled.bars[b], s, t.tuning) for b in real_rng]
                                         for s, t in filled.tracks.items()}, 0, gap.length, [b.ticks for b in bars])
            cx_gen.append(cx["complexity"])
            cx_real.append(cxr)
            stats["too_simple"] += cx["complexity"] < 0.25
            stats["neighbour_copy"] += gp_pipe.neighbour_copy_score(filled, gap) >= 0.95
            rep = play.check_song(filled, real_rng, prof, require_tuning=None)
            plays.append(rep.score)
            stats["unplayable"] += rep.errors > 0
            if index is not None:
                stats["plag_" + plag.check_song(index, filled, real_rng).verdict] += 1
            if args.listen:
                listen.append((f"gap{n:02d}", "generated", c.index, filled, gap))
        uniq.append(len(seen) / max(1, len(cands)))
        if args.listen:
            listen.append((f"gap{n:02d}", "real", -1, song, gap))
        log.info("gap %d/%d %s bars %s: pc-sim %.2f onset-sim %.2f", n + 1, len(pick), spec["song"], gap.label,
                 np.mean(sims_pc[-len(cands):]), np.mean(sims_on[-len(cands):]))
    N = max(1, stats["candidates"])
    res["generation"] = {
        "gaps": len(pick), "candidates": stats["candidates"], "finished_rate": stats["finished"] / N,
        "invalid_mass": float(np.mean(inval)) if inval else None,
        "pitch_class_similarity_to_real": float(np.mean(sims_pc)) if sims_pc else None,
        "onset_similarity_to_real": float(np.mean(sims_on)) if sims_on else None,
        "complexity_generated": float(np.mean(cx_gen)) if cx_gen else None,
        "complexity_real": float(np.mean(cx_real)) if cx_real else None,
        "too_simple_rate": stats["too_simple"] / N, "neighbour_copy_rate": stats["neighbour_copy"] / N,
        "playability_mean": float(np.mean(plays)) if plays else None, "unplayable_rate": stats["unplayable"] / N,
        "unique_rate": float(np.mean(uniq)) if uniq else None,
        **{k: v / N for k, v in stats.items() if k.startswith("plag_")},
    }
    log.info("generation: %s", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res["generation"].items()})
    out = Path(args.out) if args.out else repo_path(args.checkpoint).parent / f"eval_{args.split}.json"
    save_json(res, out)
    log.info("metrics -> %s", out)
    if args.listen:
        write_listening_set(Path(args.listen), listen, args.seed)


def write_listening_set(folder: Path, items: list, seed: int) -> None:
    """GP5 files with neutral names: 4 bars before, the gap, 4 bars after. key.json says which is real."""
    rng = np.random.default_rng(seed)
    folder.mkdir(parents=True, exist_ok=True)
    order = rng.permutation(len(items))
    key = {}
    for k, i in enumerate(order):
        gap_name, kind, cand, song, gap = items[i]
        a, b = max(0, gap.start - 4), min(len(song.bars), gap.start + gap.length + 4)
        name = f"{gap_name}_{k:03d}.gp5"
        bars = song.bars[a:b]
        for bar in bars:
            bar.marker = None
        writer.write_bars(folder / name, bars, song.tracks, bars[0].tempo or song.tempo, title=name)
        key[name] = {"gap": gap_name, "kind": kind, "candidate": int(cand), "gap_bars_in_file": f"{gap.start - a + 1}-{gap.start - a + gap.length}"}
    (folder / "key.json").write_text(json.dumps(key, indent=1))
    (folder / "ratings_template.csv").write_text("file,rating_1_to_5,notes\n" + "".join(f"{n},,\n" for n in sorted(key)))
    log.info("listening set: %d files in %s (key.json: do not open before rating)", len(key), folder)


if __name__ == "__main__":
    main()
