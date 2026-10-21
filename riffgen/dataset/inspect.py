"""Look at the raw DadaGP download before preprocessing (checks our assumptions).

    python -m riffgen.dataset.inspect --src data/dadaGP --sample 1000

Prints: how many token files / GP files, header tokens, which instruments occur, how
many guitars per song, downtunes, 7-string and drop-tuning usage, frets outside 0-29,
repeat signs, bar lengths, and an estimate of the guitar-only token count.
"""

from __future__ import annotations

import argparse
from collections import Counter

import numpy as np

from riffgen.dataset.preprocess import find_dadagp_root
from riffgen.dataset.song import parse_dadagp, read_token_file
from riffgen.tokens.vocab import GUITARS


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="data/dadaGP")
    ap.add_argument("--sample", type=int, default=1000)
    args = ap.parse_args(argv)
    src = find_dadagp_root(args.src)
    toks = sorted(src.rglob("*.tokens.txt"))
    gps = [p for ext in ("*.gp3", "*.gp4", "*.gp5") for p in src.rglob(ext) if ".pygp." not in p.name]
    jsons = sorted(src.rglob("*.json"))
    print(f"folder: {src}")
    print(f"token files: {len(toks)}   GP files: {len(gps)}   json files: {[p.name for p in jsons][:10]}")
    if not toks:
        print("no *.tokens.txt files: preprocessing will encode the GP files with the DadaGP encoder (D20)")
        return
    with_gp = sum(1 for p in toks[:2000] if p.with_name(p.name[:-len('.tokens.txt')]).is_file())
    print(f"token files with their GP file next to them: {with_gp}/{min(2000, len(toks))}")
    rng = np.random.default_rng(0)
    pick = [toks[i] for i in rng.choice(len(toks), min(args.sample, len(toks)), replace=False)]
    heads, insts, n_guitars, downtunes, kinds = Counter(), Counter(), Counter(), Counter(), Counter()
    s7 = drop = badfret = repeats = 0
    ticks = Counter()
    total = guitar = 0
    for p in pick:
        t = read_token_file(p)
        total += len(t)
        heads[t[0].split(":")[0]] += 1
        raw = parse_dadagp(t)
        downtunes[raw.downtune] += 1
        present = set()
        for bar in raw.bars:
            ticks[bar.ticks] += 1
            if any(m.startswith("measure:repeat") for m in bar.measure_tokens):
                repeats += 1
            for ev in bar.events:
                insts[ev.inst] += 1
                if ev.inst in GUITARS:
                    present.add(ev.inst)
                    guitar += len(ev.tokens) + 1
                    for tok in ev.tokens:
                        parts = tok.split(":")
                        if len(parts) == 4 and parts[1] == "note":
                            s, f = int(parts[2][1:]), int(parts[3][1:])
                            s7 += s == 7
                            drop += f < 0
                            badfret += f > 29
        n_guitars[len(present)] += 1
        for tok in t:
            kinds[tok.split(":")[0] if ":" in tok else tok] += 1
    n = len(pick)
    print(f"\nsample of {n} songs, {total} tokens ({total / n:.0f} per song)")
    print("first header token:", dict(heads))
    print("downtune:", sorted(downtunes.items()))
    print("events per instrument:", insts.most_common())
    print("guitars per song:", sorted(n_guitars.items()))
    print(f"notes on string 7: {s7}   negative frets (DadaGP drop normalisation): {drop}   frets > 29: {badfret}")
    print(f"bars with repeat signs: {repeats}")
    print("most common bar lengths (ticks):", ticks.most_common(8))
    print("token kinds:", kinds.most_common(15))
    print(f"guitar-only share (rough): {guitar / max(total, 1):.0%} -> about "
          f"{guitar / max(total, 1) * 116.7:.0f}M tokens for all 26k songs (before repeat unrolling)")


if __name__ == "__main__":
    main()
