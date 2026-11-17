#!/usr/bin/env python
"""Use the trained model as a genre classifier, and MEASURE ITS ACCURACY before anything relies on it.

    .venv/bin/python runs/genre_classify.py --validate [--n 600]       # accuracy on known-genre val songs
    .venv/bin/python runs/genre_classify.py --folder "tabs - generic" --csv out.csv

The model conditions on a genre label token (riff.py:876, flat_example takes an explicit `slots`), so for
one gap we can score the same bars under all 28 genre labels and take the lowest loss as the prediction.
That is a generative classifier with 60M trained parameters behind it, rather than the unigram
nearest-centroid screen in mislabel_scan.py, which flagged 34,103 of 44,506 songs and is useless.

--validate is the point of this script and must come first. It runs the classifier on validation gaps
whose genre is already known and reports top-1 / top-3 accuracy plus a confusion summary. Only if that
number is good is there any reason to run the classifier over the 9,525 unlabelled files in
"tabs - generic". A classifier of unknown accuracy applied to 13.55% of the corpus would silently bake its
own errors into every later genre result.

Fairness detail: the "rel" slot (how a gap relates to the preceding bars) is left empty for every
candidate rather than computed, so all 28 scores differ ONLY in the genre token. The album and src slots
come from the song itself and are identical across candidates too.
"""
from __future__ import annotations

import argparse
import collections
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import riff  # noqa: E402
import torch  # noqa: E402


def score_all(model, c, cfg, V, gap, dev, cands):
    """loss under each candidate genre for one gap; lower is better."""
    view = c.view(gap["song"])
    rng = riff.rng_for(0, gap["song"])
    base = {k: v for k, v in view.labels.items() if k != "genre"}
    out = {}
    for g in cands:
        slots = V.slots(base | {"genre": g})
        ex = riff.flat_example(V, view, riff.rng_for(0, gap["song"]), cfg["ctx"],
                               (gap["start"], gap["len"]), train=False, slots=slots)
        if ex is None:
            return None
        with torch.no_grad(), riff.autocast(dev):
            _, st = model.loss(riff.to_dev(riff.collate([ex], V, cfg), dev))
        w = max(float(st.get("w", 0.0)), 1e-9)
        out[g] = float(st.get("loss", 0.0)) / w
    del rng
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/final10/best.pt")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--folder", help="classify the songs in this top-level folder instead")
    ap.add_argument("--n", type=int, default=600, help="how many songs to score")
    ap.add_argument("--csv")
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    c = riff.Corpus(riff.ROOT / "data_encoded" / "encoded")
    model, V, cfg, _ = riff.load_ckpt(a.ckpt, dev)
    model.eval()
    cands = list(riff.GENRES)

    if a.validate:
        # only gaps whose song carries a real genre, and which sit in a plain genre folder
        sel = [g for g in c.gaps if g["split"] == "val"
               and (lab := c.songs[g["song"]]["labels"].get("genre")) and lab in cands
               and not c.songs[g["song"]]["file"].startswith("tabs - own - ")]
        # The gaps come out grouped by song, so a plain head() would sample a single genre - the first
        # smoke test scored 40 gaps and every one of them was acoustic. Take a round-robin across genres
        # instead, so top-1 is not dominated by whichever label happens to sort first.
        per = collections.defaultdict(list)
        for g in sel:
            per[c.songs[g["song"]]["labels"]["genre"]].append(g)
        rr, keys = [], sorted(per)
        for k in range(max(len(v) for v in per.values())):
            for key in keys:
                if k < len(per[key]):
                    rr.append(per[key][k])
        sel = rr[:a.n]
        print(f"validating on {len(sel)} val gaps with known genres, {len(cands)} candidates each, "
              f"{len({c.songs[g['song']]['labels']['genre'] for g in sel})} genres represented")
        top1 = top3 = n = 0
        conf = collections.Counter()
        rows = []
        for i, g in enumerate(sel):
            truth = c.songs[g["song"]]["labels"]["genre"]
            sc = score_all(model, c, cfg, V, g, dev, cands)
            if sc is None:
                continue
            order = sorted(sc, key=sc.get)
            n += 1
            top1 += order[0] == truth
            top3 += truth in order[:3]
            conf[(truth, order[0])] += 1
            rows.append({"file": c.songs[g["song"]]["file"], "truth": truth,
                         "pred": order[0], "pred2": order[1], "pred3": order[2],
                         "loss_pred": round(sc[order[0]], 4), "loss_truth": round(sc[truth], 4)})
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(sel)}  top1 {top1 / n:.3f}  top3 {top3 / n:.3f}", flush=True)
        print(f"\nscored {n} gaps")
        print(f"  top-1 accuracy {top1 / max(n, 1):.3f}   (chance is {1 / len(cands):.3f})")
        print(f"  top-3 accuracy {top3 / max(n, 1):.3f}")
        print("\nmost common confusions (truth -> predicted):")
        for (t, p), k in conf.most_common(14):
            if t != p:
                print(f"    {k:4d}  {t:22s} -> {p}")
        print("\ncorrect by genre:")
        per = collections.Counter()
        tot = collections.Counter()
        for (t, p), k in conf.items():
            tot[t] += k
            if t == p:
                per[t] += k
        for t in sorted(tot, key=lambda t: -(per[t] / tot[t])):
            print(f"    {per[t]:3d}/{tot[t]:3d}  {per[t] / tot[t]:.2f}  {t}")
        if a.csv:
            with open(a.csv, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
            print(f"\n-> {a.csv}")
        return 0

    if not a.folder:
        sys.exit("pass --validate or --folder")
    # classify unlabelled songs: one gap per song, taken from the middle, 16 bars
    want = [i for i, s in enumerate(c.songs) if s["file"].startswith(a.folder + "/") and s["nb"] >= 24]
    want = want[:a.n]
    print(f"classifying {len(want)} songs from {a.folder}")
    rows = []
    for k, i in enumerate(want):
        s = c.songs[i]
        g = {"song": i, "start": max(0, s["nb"] // 2 - 8), "len": 16}
        sc = score_all(model, c, cfg, V, g, dev, cands)
        if sc is None:
            continue
        order = sorted(sc, key=sc.get)
        rows.append({"file": s["file"], "pred": order[0], "pred2": order[1], "pred3": order[2],
                     "loss1": round(sc[order[0]], 4), "margin": round(sc[order[1]] - sc[order[0]], 4)})
        if (k + 1) % 100 == 0:
            print(f"  {k + 1}/{len(want)}", flush=True)
    print("\npredicted distribution:", collections.Counter(r["pred"] for r in rows).most_common())
    if a.csv and rows:
        with open(a.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"-> {a.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
