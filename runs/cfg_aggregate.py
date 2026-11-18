#!/usr/bin/env python
"""Average the per-song cfg_control.py reports into one table, so a default can be chosen from it.

    .venv/bin/python runs/cfg_aggregate.py runs/cfg_cal

Reads the "mean pairwise divergence" block and the per-genre log-probabilities out of each report and
reports, per guidance weight: mean divergence across songs, its spread, and the worst log-probability
seen. The weight to pick is the largest divergence that has not yet cost log-probability - below that the
label does nothing, above it the output leaves the distribution.
"""
from __future__ import annotations

import re
import statistics as st
import sys
from pathlib import Path

DIV = re.compile(r"^\s*cfg\s+([\d.]+):\s+mean\s+([\d.]+)\s+max\s+([\d.]+)", re.M)
ROW = re.compile(r"^\s*cfg\s+([\d.]+)\s+(\S+)\s+density\s+([\d.]+)\s+string\s+([\d.]+)\s+"
                 r"pm\s+([\d.]+)\s+notes\s+([\d.]+)\s+logp\s+([-+\d.]+)", re.M)


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/cfg_cal")
    files = sorted(root.glob("*.txt"))
    if not files:
        print(f"no reports in {root}")
        return 1
    div: dict[float, list[float]] = {}
    logp: dict[float, list[float]] = {}
    dens: dict[float, list[float]] = {}
    for f in files:
        txt = f.read_text(encoding="utf-8", errors="replace")
        for w, m, _mx in DIV.findall(txt):
            div.setdefault(float(w), []).append(float(m))
        for w, _g, d, _s, _pm, _n, lp in ROW.findall(txt):
            logp.setdefault(float(w), []).append(float(lp))
            dens.setdefault(float(w), []).append(float(d))

    print(f"{len(files)} songs\n")
    print(f"{'cfg':>5s} {'divergence':>18s} {'worst logp':>11s} {'mean logp':>10s} {'density':>9s}")
    for w in sorted(div):
        ds = div[w]
        lp = logp.get(w, [0.0])
        sd = st.pstdev(ds) if len(ds) > 1 else 0.0
        print(f"{w:5.1f} {st.mean(ds):8.3f} +-{sd:<7.3f} {min(lp):11.3f} {st.mean(lp):10.3f} "
              f"{st.mean(dens.get(w, [0])):9.1f}")
    print("\nPick the largest weight whose worst logp is still near 0: below it the genre label changes")
    print("nothing, above it the model is being pushed off its own distribution rather than restyled.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
