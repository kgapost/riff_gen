"""See and hear what the sampler builds (§3.8.3): decode augmented examples to GP5.

    python -m riffgen.dataset.preview --album album_01 --n 20
    python -m riffgen.dataset.preview --store data/processed/dadagp --n 10 --task song

The GP5 is rebuilt from the example's *tokens*, so it shows exactly what the model sees:
swapped/dropped guitars, ``gap:N`` placeholders (empty bars with a marker), the hidden
part (marker "HIDDEN n bars (to write)") placed back between the context bars. A .txt next
to it has the exact token sequence.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from riffgen import REPO_DIR
from riffgen.dataset.sampler import ExampleBuilder
from riffgen.dataset.song import Bar, Event, TrackInfo
from riffgen.dataset.store import Store
from riffgen.gp import writer
from riffgen.tokens import tunings as tunings_mod
from riffgen.tokens.vocab import GUITAR_SLOT
from riffgen.train.common import load_model_yaml, model_config, repo_path, sampler_config
from riffgen.utils.config import load_config
from riffgen.utils.seed import derive


def tokens_to_bars(toks: list[str]) -> list[Bar]:
    bars: list[Bar] = []
    cur: Bar | None = None
    tick = 0
    ev: Event | None = None

    def close():
        if cur is not None:
            cur.ticks = max(tick, 1)

    for t in toks:
        head, _, rest = t.partition(":")
        if t == "new_measure":
            close()
            cur = Bar(ticks=0)
            bars.append(cur)
            tick, ev = 0, None
        elif head == "gap":
            close()
            cur = None
            n = int(rest)
            last = bars[-1].ticks if bars else 3840
            for k in range(n):
                bars.append(Bar(ticks=last, marker=f"gap:{n} placeholder" if k == 0 else None, gap=True))
            tick = 0
        elif cur is None:
            continue
        elif head == "ts":
            a, b = rest.split("/")
            cur.ts = (int(a), int(b))
        elif head == "section":
            cur.marker = rest
        elif head == "wait":
            tick += int(rest)
            ev = None
        elif head in GUITAR_SLOT:
            slot = GUITAR_SLOT[head]
            ev = next((e for e in cur.events if e.tick == tick and e.track == slot), None)
            if ev is None:
                ev = Event(tick, slot, [])
                cur.events.append(ev)
            ev.tokens.append(t)
        elif head in ("nfx", "param", "bfx") and ev is not None:
            ev.tokens.append(t)
    close()
    prev_ts = (4, 4)
    for b in bars:
        b.ts = b.ts or prev_ts
        prev_ts = b.ts
    return bars


def decode_example(ex: dict, vocab) -> tuple[list[Bar], dict[int, TrackInfo]]:
    toks = vocab.decode(ex["ids"])
    tracks: dict[int, TrackInfo] = {}
    for a, b in zip(toks, toks[1:]):
        if a.startswith("track:") and b.startswith("tuning:"):
            slot = GUITAR_SLOT[a.split(":")[1]]
            name = b.split(":")[1]
            tuning = tunings_mod.by_name(name) if name != "other" else tunings_mod.STD6
            tracks[slot] = TrackInfo(slot=slot, tuning=tuning, name=f"{a.split(':')[1]} ({name})")
    if ex["meta"]["task"] == "infill":
        i_p, i_s, i_m = (toks.index(x) for x in ("<fim_prefix>", "<fim_suffix>", "<fim_middle>"))
        pre, suf, mid = tokens_to_bars(toks[i_p:i_s]), tokens_to_bars(toks[i_s:i_m]), tokens_to_bars(toks[i_m:])
        if pre:
            pre[0].marker = pre[0].marker or "CONTEXT"
        if mid:
            mid[0].marker = f"HIDDEN {len(mid)} bars (to write)"
        if suf:
            suf[0].marker = suf[0].marker or "CONTEXT (after)"
        bars = pre + mid + suf
    else:
        bars = tokens_to_bars(toks)
    return bars, tracks


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--album", default=None)
    ap.add_argument("--store", default=None)
    ap.add_argument("--model", default="baseline", help="for the context length")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--task", default="infill", help="infill | song")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if args.album:
        p = Path(args.album)
        acfg = load_config(p if p.suffix else REPO_DIR / "configs" / "albums" / f"{args.album}.yaml")
        store = Store(repo_path(acfg.get_path("data.store", f"data/processed/albums/{acfg.album_id}")))
        cfg, name = acfg, acfg.album_id
    else:
        store = Store(repo_path(args.store))
        cfg, name = load_config(REPO_DIR / "configs" / "train" / "pretrain.yaml"), store.path.name
    vocab = store.load_vocab()
    sc = sampler_config(cfg, model_config(load_model_yaml(args.model), vocab))
    builder = ExampleBuilder(vocab, sc)
    out = Path(args.out) if args.out else REPO_DIR / "reports" / f"preview_{name}"
    out.mkdir(parents=True, exist_ok=True)
    idx = store.indices("train")
    rng = derive(args.seed, "preview")
    written = 0
    for k in range(args.n * 3):
        if written >= args.n:
            break
        i = int(idx[rng.integers(len(idx))])
        ex = builder.build(store.view(i), args.task, derive(args.seed, "preview", k))
        if ex is None:
            continue
        bars, tracks = decode_example(ex, vocab)
        if not bars or not tracks:
            continue
        m = ex["meta"]
        tempo = next((float(t.split(":")[1]) for t in vocab.decode(ex["ids"][:24]) if t.startswith("tempo:")), 120.0)
        stem = f"{written:02d}_{Path(m['song']).stem}_{m['task']}_bar{m['start'] + 1}"
        writer.write_bars(out / f"{stem}.gp5", bars, tracks, tempo, title=stem)
        (out / f"{stem}.txt").write_text(f"{m}\n\n" + " ".join(vocab.decode(ex["ids"])) + "\n", encoding="utf-8")
        written += 1
    print(f"{written} previews in {out}")


if __name__ == "__main__":
    main()
