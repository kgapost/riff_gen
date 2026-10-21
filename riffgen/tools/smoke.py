"""End-to-end smoke test on synthetic data (CPU, a few minutes). Run it before the real data
is there, and after installing on a new machine:

    python -m riffgen.tools.smoke                 # tiny flat model (same code path as "big")
    python -m riffgen.tools.smoke --huge          # tiny hierarchical model (same code path as "huge")
    python -m riffgen.tools.smoke --keep DIR      # keep the files to look at them

Everything happens in a temporary folder; runs/, outputs/ and reports/ are not touched.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
import time
import traceback
from pathlib import Path

from riffgen.tools.synth import write_album, write_dadagp_corpus
from riffgen.utils.log import get_logger

log = get_logger("riffgen.smoke")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--huge", action="store_true")
    ap.add_argument("--keep", default=None)
    ap.add_argument("--songs", type=int, default=40)
    args = ap.parse_args(argv)
    model = "test_tiny_huge" if args.huge else "test_tiny"
    root = Path(args.keep) if args.keep else Path(tempfile.mkdtemp(prefix="riffgen_smoke_"))
    root.mkdir(parents=True, exist_ok=True)
    steps = []

    def step(name, fn):
        t0 = time.time()
        try:
            fn()
            steps.append((name, "ok", time.time() - t0))
            log.info("[ok] %s (%.0fs)", name, time.time() - t0)
        except BaseException as e:  # noqa: BLE001 - report every failure
            steps.append((name, f"FAILED: {type(e).__name__}: {e}", time.time() - t0))
            log.error("[FAILED] %s\n%s", name, traceback.format_exc())
            raise

    album_cfg = root / "album.yaml"
    try:
        step("synthetic data", lambda: (write_dadagp_corpus(root / "dadagp", args.songs), write_album(root / "album")))
        album_cfg.write_text(f"""inherit: train/finetune.yaml
album_id: smoke_album
model: {model}
reports_dir: {root / 'reports'}
data:
  root: {root / 'album'}
  store: {root / 'album_store'}
  own: own
  others: [favourites]
  holdout: [own/track03.gp5]
finetune:
  max_epochs: 2
  evals_per_epoch: 1
dadagp_mix:
  enabled: true
  store: {root / 'dadagp_store'}
  share: 0.15
  max_songs: 10
generation:
  candidates: 4
  keep: 3
  min_complexity: 0.0
""", encoding="utf-8")
        from riffgen.dataset import preprocess
        step("preprocess DadaGP", lambda: preprocess.main(["dadagp", "--src", str(root / "dadagp"), "--out",
                                                            str(root / "dadagp_store"), "--workers", "2",
                                                            "--val-fraction", "0.15"]))
        step("preprocess album", lambda: preprocess.main(["album", str(album_cfg), "--out", str(root / "album_store"),
                                                           "--workers", "2"]))
        from riffgen.train import finetune, pretrain
        step("pretrain", lambda: pretrain.main(["--model", model, "--run", str(root / "runs" / "pretrain"),
                                                f"data.stores=[{{path: {root / 'dadagp_store'}, split: train}}]",
                                                f"data.val_store={root / 'dadagp_store'}", "data.val_infill=8",
                                                "data.val_song=4"]))
        step("pretrain --resume", lambda: pretrain.main(["--model", model, "--run", str(root / "runs" / "pretrain"),
                                                         "--resume", "train.max_steps=8",
                                                         f"data.stores=[{{path: {root / 'dadagp_store'}, split: train}}]",
                                                         f"data.val_store={root / 'dadagp_store'}", "data.val_infill=8",
                                                         "data.val_song=4"]))
        step("finetune", lambda: finetune.main([str(album_cfg), "--base", str(root / "runs" / "pretrain" / "best.pt"),
                                                "--run", str(root / "runs" / "smoke_album" / model),
                                                f"dadagp_mix.store={root / 'dadagp_store'}"]))
        from riffgen.generate.__main__ import main as gen_main
        out = root / "outputs"
        song = sorted((root / "album" / "songs").glob("*.gp5"))[0]
        step("generate", lambda: gen_main(["--album", str(album_cfg), "--song", str(song),
                                           "--checkpoint", str(root / "runs" / "smoke_album" / model / "best.pt"),
                                           "--index", str(root / "runs" / "smoke_album" / "plagiarism_index.sqlite"),
                                           "--out", str(out), "--fix"]))
        from riffgen.check.__main__ import main as check_main
        step("check folder", lambda: check_main(["--album", str(album_cfg), "--folder", str(out), "--playability"]))
        from riffgen.eval.__main__ import main as eval_main
        step("eval", lambda: eval_main(["--checkpoint", str(root / "runs" / "smoke_album" / model / "best.pt"),
                                        "--store", str(root / "album_store"), "--split", "holdout", "--gaps", "2",
                                        "--candidates", "2", "--listen", str(root / "listen")]))
    except BaseException:
        pass
    finally:
        print("\nSMOKE TEST SUMMARY")
        for name, status, dt in steps:
            print(f"  {name:22s} {status} ({dt:.0f}s)")
        ok = all(s == "ok" for _, s, _ in steps) and len(steps) >= 9
        print("ALL OK" if ok else "SOMETHING FAILED", "-", root)
        if not args.keep and ok:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
