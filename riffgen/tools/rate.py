"""Rate candidates 1-5 after listening (README §6.4). Ratings go to outputs/<album>/ratings.jsonl
with the candidate's seed and scores, so they can later train a ranking model.

    python -m riffgen.tools.rate outputs/album_01/2026-10-18_song_03_gap_33-40
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from riffgen.utils.log import JsonlWriter


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    args = ap.parse_args(argv)
    folder = Path(args.folder)
    report = json.loads((folder / "report.json").read_text())
    by_file = {r.get("file"): r for r in report["results"] if r.get("file")}
    log = JsonlWriter(folder.parent / "ratings.jsonl")
    print(f"{report['song']} gap {report['gap']}: rate each candidate 1-5 (Enter = skip, q = quit)")
    for name in sorted(by_file):
        r = by_file[name]
        ans = input(f"  {name} (score {r['score']}, {r.get('plagiarism', '')}): ").strip().lower()
        if ans == "q":
            break
        if ans not in {"1", "2", "3", "4", "5"}:
            continue
        note = input("    note (optional): ").strip()
        log.write({"song": report["song"], "gap": report["gap"], "file": name, "rating": int(ans), "note": note,
                   "candidate": r["candidate"], "seed": r["seed"], "checkpoint": report["checkpoint"],
                   "temperature": report["temperature"], "top_p": report["top_p"],
                   **{k: r.get(k) for k in ("mean_logprob", "complexity", "playability", "plagiarism", "neighbour_copy")}})
    print(f"saved to {folder.parent / 'ratings.jsonl'}")


if __name__ == "__main__":
    main()
