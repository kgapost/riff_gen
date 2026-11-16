#!/usr/bin/env python
"""Keep the checkpoint that is best on the TUNED LABEL's own validation songs, not on the whole split.

    python runs/label_best_watcher.py        # follows runs/status.txt; stops when the B/C driver exits

Why this exists: train_run picks best.pt by the loss on the entire validation split, which for a genre run
is ~96 % other genres. A run that pushes one genre to 40 % of the draws can lower that global average while
raising the loss on the genre it was tuned for - which is what genre_acoustic did (global 0.2012 -> 0.1976,
acoustic's own songs 0.2501 -> 0.2598). riff.py cannot select on a label subset and is not to be edited, so
this watches from outside.

How: train_run rewrites last.pt at every evaluation (riff.py:1826). This polls its mtime, and on each change
scores that checkpoint on the val gaps of the label the run is tuning, appending a row to
<run>/label_evals.jsonl. The best one so far is copied to <run>/label_best.pt with the optimiser state
dropped, so the copy costs ~230 MB rather than ~690 MB.

It never writes to best.pt or last.pt: riff.py owns those, and the point is to add a third choice, not to
overwrite the run's own record. Evaluation shares the GPU with the training step; the per-label val sets are
small (2-63 songs), so a pass costs seconds.
"""
from __future__ import annotations

import json
import re
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import riff  # noqa: E402

ROOT = riff.ROOT
STATUS = ROOT / "runs" / "status.txt"
POLL = 20.0


def current() -> tuple[Path, str] | None:
    """The run directory and label the driver says it is on, or None."""
    try:
        txt = STATUS.read_text(encoding="utf-8")
    except OSError:
        return None
    # Preferred form: the driver states the directory and label outright. The stage/run parsing below
    # infers the directory as runs/<kind>_<name>, which only holds while every driver names its runs that
    # way; run_stage_b2.sh uses a prefix (runs/b2_rock), so it writes these two lines instead.
    rd = re.search(r"^rundir\s+(\S+)", txt, re.M)
    lb = re.search(r"^tuning\s+(\S+)", txt, re.M)
    if rd and lb:
        return ROOT / rd.group(1), lb.group(1)
    stage = re.search(r"^stage\s+(\S+)\s*(\S*)", txt, re.M)
    run = re.search(r"^run\s+\d+/\d+\s+(\S+)", txt, re.M)
    if not stage or not run:
        return None
    kind = stage.group(2) or stage.group(1)
    name = run.group(1)
    if kind.startswith("genre"):
        return ROOT / "runs" / f"genre_{name}", f"genre:{name}"
    if kind.startswith("album"):
        return ROOT / "runs" / f"album_{name}", f"album:{name}"
    return None


def driver_alive() -> bool:
    import subprocess
    out = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
    return any(d in out for d in ("run_stages_bc.sh", "run_stage_b2.sh"))


def main() -> int:
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    c = riff.Corpus(ROOT / "data_encoded" / "encoded")
    gaps_cache: dict[str, list] = {}
    seen: dict[str, float] = {}          # run dir -> mtime of last.pt already scored
    best: dict[str, float] = {}          # run dir -> best label loss so far

    print(f"[{time.strftime('%F %T')}] watching {STATUS}", flush=True)
    while True:
        if not driver_alive():
            print(f"[{time.strftime('%F %T')}] driver gone, stopping", flush=True)
            return 0
        cur = current()
        if cur is None:
            time.sleep(POLL)
            continue
        run, label = cur
        ck = run / "last.pt"
        try:
            mt = ck.stat().st_mtime
        except OSError:
            time.sleep(POLL)
            continue
        key = str(run)
        if seen.get(key) == mt:
            time.sleep(POLL)
            continue
        # a fresh evaluation checkpoint: score it on the label's own val gaps
        time.sleep(3)                                     # let the rename settle
        try:
            model, V, cfg, _ = riff.load_ckpt(ck, dev)
        except Exception as e:                            # a half-written file, or vocabulary mismatch
            print(f"[{time.strftime('%F %T')}] {label}: cannot load last.pt ({e}); retrying", flush=True)
            time.sleep(POLL)
            continue
        if label not in gaps_cache:
            kind, _, value = label.partition(":")
            gaps_cache[label] = [(j, g) for j, g in enumerate(g for g in c.gaps if g["split"] == "val")
                                 if c.songs[g["song"]]["labels"].get(kind) == value]
        sel = gaps_cache[label]
        seen[key] = mt
        if not sel:
            print(f"[{time.strftime('%F %T')}] {label}: no val gaps, nothing to select on", flush=True)
            del model
            torch.cuda.empty_cache()
            time.sleep(POLL)
            continue
        model.eval()
        with torch.no_grad():
            res = riff.summarise(riff.gap_loss(model, c, cfg, sel, dev))
        loss = res["loss"]
        row = {"time": time.strftime("%F %T"), "label": label, "n_gaps": len(sel), "loss": loss,
               "acc": res["acc"], "mtime": mt}
        with open(run / "label_evals.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
        mark = ""
        if loss < best.get(key, float("inf")):
            best[key] = loss
            ckobj = {"cfg": cfg, "tokens": V.tokens, "model": model.state_dict(),
                     "lora": getattr(model, "_lora_rank", 0) or 0, "state": {"label_loss": loss}}
            tmp = run / "label_best.tmp"
            torch.save(ckobj, tmp)
            shutil.move(str(tmp), str(run / "label_best.pt"))
            mark = "  <- best on label"
        print(f"[{row['time']}] {label}: {len(sel)} gaps, loss {loss:.4f}{mark}", flush=True)
        del model
        torch.cuda.empty_cache()
        time.sleep(POLL)


if __name__ == "__main__":
    raise SystemExit(main())
