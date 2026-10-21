"""Similar songs from DadaGP for album fine-tuning (§3.9, D47, D48).

    python -m riffgen.dataset.similar configs/albums/album_01.yaml [--method features|embeddings]

1. pool: preprocessed DadaGP songs, minus songs that are also in the album (title/artist or
   shared 2-bar tab pieces) and songs that are mostly filtered out;
2. a style vector per riff (option A: musical features; option B: base-model embeddings);
3. every DadaGP riff: mean similarity to its 5 nearest album riffs; song score = mean of
   its 3 best riffs (riffs, not songs: one prog song can hold very different parts);
4. select: best songs up to max_songs, max_per_artist per artist, always/never artists,
   7-string tunings preferred;
5. writes runs/<album>/dadagp_mix.json, reports/dadagp_mix_<album>.csv and GP5 snippets of
   20 random selected riffs (listen to them, like the filter audit).
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from tqdm import tqdm

from riffgen import REPO_DIR
from riffgen.analysis.filters import complexity
from riffgen.analysis.notes import song_onsets
from riffgen.dataset.pipeline import normalize_title
from riffgen.dataset.store import Store, song_from_view
from riffgen.utils.config import load_config, save_json
from riffgen.utils.log import get_logger

log = get_logger("riffgen.similar")
TECH = ("hammer", "bend", "slide", "dead", "harmonic", "vibrato", "tremoloPicking")
FEATURE_DIM = 59


def riff_vectors(song, segments: list[dict]) -> list[tuple[int, np.ndarray]]:
    """Option A style vector of every kept riff segment (>= 2 bars)."""
    onsets = song_onsets(song)
    ticks = [b.ticks for b in song.bars]
    out = []
    for k, seg in enumerate(segments):
        if seg["kind"] != "music" or seg["action"] != "keep" or seg["end"] - seg["start"] < 2:
            continue
        a, b = seg["start"], seg["end"]
        n_on = n_notes = pm = chords = 0
        tech = np.zeros(len(TECH))
        sfb = np.zeros(7)
        ivl = np.zeros(25)
        pos = np.zeros(12)
        low_rel = []
        for slot, bars in onsets.items():
            tuning = song.tracks[slot].tuning
            n_str = len(tuning)
            prev = None
            for i in range(a, b):
                for on in bars[i]:
                    if not on.notes or not on.is_attack:
                        continue
                    n_on += 1
                    n_notes += len(on.notes)
                    pm += on.palm_mute
                    chords += len(on.notes) >= 2
                    pos[(on.tick % 960) * 12 // 960] += 1
                    for n in on.notes:
                        for t_i, t in enumerate(TECH):
                            if t in n.fx:
                                tech[t_i] += 1
                        sfb[min(6, max(0, n_str - n.string))] += 1
                    lp = on.low_pitch
                    if lp is not None:
                        low_rel.append(lp - tuning[-1])
                        if prev is not None:
                            ivl[int(np.clip(lp - prev, -12, 12)) + 12] += 1
                        prev = lp
        if n_on == 0:
            continue
        beats = sum(ticks[a:b]) / 960.0
        odd = np.mean([1.0 if song.bars[i].ts and (song.bars[i].ts[0] in (5, 7, 9, 11, 13, 15)
                                                   or (song.bars[i].ts[1] == 8 and song.bars[i].ts[0] % 3)) else 0.0
                       for i in range(a, b)])
        cx = complexity(onsets, a, b, ticks)
        v = np.concatenate([
            [min(song.bars[a].tempo / 250.0, 1.5), odd, min(n_on / max(beats, 1) / 8.0, 1.5), pm / n_on, chords / n_on],
            tech / max(n_notes, 1), sfb / max(sfb.sum(), 1), ivl / max(ivl.sum(), 1), pos / max(pos.sum(), 1),
            [cx["pitch"], cx["rhythm"], float(np.mean(low_rel)) / 24.0 if low_rel else 0.0],
        ]).astype(np.float32)
        out.append((k, v))
    return out


_STORE: dict = {}   # per worker process: path -> (store, vocab)


def _song_features(args) -> tuple[int, list[tuple[int, np.ndarray]]]:
    path, idx = args
    if path not in _STORE:
        st = Store(path)
        _STORE[path] = (st, st.load_vocab())
    store, vocab = _STORE[path]
    view = store.view(idx)
    return idx, riff_vectors(song_from_view(view, vocab), view.meta.get("segments", []))


def store_features(store: Store, indices: list[int], workers: int = 8, cache: Path | None = None) -> dict:
    if cache and cache.is_file():
        d = np.load(cache, allow_pickle=True)
        return {"song": d["song"], "seg": d["seg"], "vec": d["vec"]}
    songs, segs, vecs = [], [], []
    jobs = [(str(store.path), i) for i in indices]
    if workers > 1:
        import multiprocessing as mp
        import sys
        ctx = mp.get_context("spawn") if "torch" in sys.modules else None   # no fork() after torch threads
        with ProcessPoolExecutor(workers, mp_context=ctx) as ex:
            results = list(tqdm(ex.map(_song_features, jobs, chunksize=16), total=len(jobs), desc="riff features"))
    else:
        results = [_song_features(j) for j in tqdm(jobs, desc="riff features")]
    for idx, rv in results:
        for k, v in rv:
            songs.append(idx)
            segs.append(k)
            vecs.append(v)
    out = {"song": np.array(songs, np.int32), "seg": np.array(segs, np.int32),
           "vec": np.stack(vecs) if vecs else np.zeros((0, FEATURE_DIM), np.float32)}
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, **out)
    return out


def embedding_features(store: Store, indices: list[int], checkpoint: str, device: str = "auto") -> dict:
    """Option B: mean final hidden state of the base model over each riff (song-mode sequence)."""
    import torch

    from riffgen.dataset.loader import collate
    from riffgen.dataset.sampler import ExampleBuilder, SamplerConfig
    from riffgen.train.checkpoint import load_model
    from riffgen.utils.device import pick_device, to_device
    dev = pick_device(device)
    model, vocab, _ = load_model(checkpoint, dev)
    model.eval()
    b = ExampleBuilder(vocab, SamplerConfig(max_len=1024, hierarchical=model.cfg.hierarchical))
    songs, segs, vecs = [], [], []
    for i in tqdm(indices, desc="riff embeddings"):
        view = store.view(i)
        for k, seg in enumerate(view.meta.get("segments", [])):
            if seg["kind"] != "music" or seg["action"] != "keep" or seg["end"] - seg["start"] < 2:
                continue
            ex = b.build_song(view, np.random.default_rng(0), {"task": "song", "start": seg["start"]}, augment=False)
            if ex is None:
                continue
            bars = ex["bar"]
            ex["loss"] = ((bars >= seg["start"]) & (bars < seg["end"])).astype(np.float32)
            with torch.no_grad():
                e = model.bar_embeddings(to_device(collate([ex], vocab.pad_id), dev))[0].cpu().numpy()
            songs.append(i)
            segs.append(k)
            vecs.append(e)
    return {"song": np.array(songs, np.int32), "seg": np.array(segs, np.int32),
            "vec": np.stack(vecs) if vecs else np.zeros((0, 1), np.float32)}


def _normalize(train: np.ndarray, *others: np.ndarray):
    mu, sd = train.mean(0), train.std(0) + 1e-6
    outs = []
    for x in (train,) + others:
        z = (x - mu) / sd
        outs.append(z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-6))
    return outs


def select_for_album(acfg, out: Path | None = None, method: str | None = None, workers: int = 8) -> dict:
    mix = acfg.get("dadagp_mix") or {}
    method = method or mix.get("method", "features")
    from riffgen.train.common import repo_path
    album = Store(repo_path(acfg.get_path("data.store", f"data/processed/albums/{acfg.album_id}")))
    dpath = repo_path(mix.get("store") or "data/processed/dadagp")
    dstore = Store(dpath)
    album_idx = album.indices("train")
    # 1. pool and exclusions
    album_titles = {normalize_title(s["title"]) for s in album.songs} | {normalize_title(Path(s["source"]).stem)
                                                                        for s in album.songs}
    album_hold_titles = {normalize_title(s["title"]) for s in album.songs if s.get("split") == "holdout"}
    exclude = {a.lower() for a in (mix.get("exclude_artists") or [])}
    always = {a.lower() for a in (mix.get("always_include_artists") or [])}
    pool = []
    dropped = {"same_title": 0, "excluded_artist": 0, "mostly_filtered": 0}
    for i, s in enumerate(dstore.songs):
        if s["artist"].lower() in exclude:
            dropped["excluded_artist"] += 1
            continue
        if normalize_title(s["title"]) in album_titles | album_hold_titles:
            dropped["same_title"] += 1
            continue
        keep_bars = sum(seg["end"] - seg["start"] for seg in s.get("segments", [])
                        if seg["kind"] == "music" and seg["action"] == "keep")
        if keep_bars < 0.5 * max(1, s["n_bars"]):
            dropped["mostly_filtered"] += 1
            continue
        pool.append(i)
    log.info("DadaGP pool: %d songs (removed %s)", len(pool), dropped)
    # 2. style vectors
    if method == "embeddings":
        ckpt = acfg.get("base_model") or f"runs/pretrain_{acfg.get('model', 'baseline')}/best.pt"
        A = embedding_features(album, album_idx, str(REPO_DIR / ckpt))
        D = embedding_features(dstore, pool, str(REPO_DIR / ckpt))
    else:
        A = store_features(album, album_idx, workers=1)
        D = store_features(dstore, list(range(len(dstore))), workers, cache=dpath / "riff_features.npz")
        keep = np.isin(D["song"], np.array(pool, np.int32))
        D = {k: v[keep] for k, v in D.items()}
    if len(A["vec"]) == 0 or len(D["vec"]) == 0:
        raise RuntimeError("no riffs to compare (empty album or pool)")
    Dn, An = _normalize(D["vec"], A["vec"])
    # 3. riff scores: mean similarity to the k nearest album riffs, in chunks
    k = int(mix.get("k_nearest", 5))
    riff_score = np.zeros(len(Dn), np.float32)
    for s in range(0, len(Dn), 4096):
        sim = Dn[s:s + 4096] @ An.T
        kk = min(k, sim.shape[1])
        riff_score[s:s + 4096] = np.sort(sim, axis=1)[:, -kk:].mean(1)
    top = int(mix.get("top_riffs_per_song", 3))
    by_song: dict[int, list[tuple[float, int]]] = {}
    for sc, si, sg in zip(riff_score, D["song"], D["seg"]):
        by_song.setdefault(int(si), []).append((float(sc), int(sg)))
    prefer = set(mix.get("prefer_tunings") or [])
    song_score = {}
    for si, lst in by_song.items():
        lst.sort(reverse=True)
        sc = float(np.mean([x for x, _ in lst[:top]]))
        tracks = dstore.songs[si]["tracks"].values()
        from riffgen.tokens import tunings as T
        names = {T.name_of(tuple(t["tuning"])) for t in tracks}
        if names & prefer or any(len(t["tuning"]) == 7 for t in tracks):
            sc += float(mix.get("tuning_bonus", 0.05))
        song_score[si] = (sc, [g for _, g in lst[:top]])
    # 4. select
    max_songs = int(mix.get("max_songs", 1000))
    max_artist = int(mix.get("max_per_artist", 20))
    per_artist: dict[str, int] = {}
    chosen = []
    ranked = sorted(song_score, key=lambda i: -song_score[i][0])
    first = [i for i in ranked if dstore.songs[i]["artist"].lower() in always]
    for i in first + [i for i in ranked if i not in set(first)]:
        art = dstore.songs[i]["artist"].lower()
        if per_artist.get(art, 0) >= max_artist:
            continue
        chosen.append(i)
        per_artist[art] = per_artist.get(art, 0) + 1
        if len(chosen) >= max_songs:
            break
    chosen = _drop_tab_duplicates(chosen, album, dstore)
    ids = [dstore.songs[i]["id"] for i in chosen]
    result = {"album": acfg.album_id, "method": method, "ids": ids,
              "scores": {dstore.songs[i]["id"]: round(song_score[i][0], 4) for i in chosen}, "dropped": dropped}
    out = out or REPO_DIR / "runs" / acfg.album_id / "dadagp_mix.json"
    save_json(result, out)
    _report(acfg, dstore, chosen, song_score, Path(acfg.get("reports_dir") or REPO_DIR / "reports"))
    log.info("selected %d DadaGP songs for %s -> %s", len(ids), acfg.album_id, out)
    return result


def _drop_tab_duplicates(chosen: list[int], album: Store, dstore: Store, min_shared: int = 4) -> list[int]:
    """D48: remove DadaGP songs that share several exact 2-bar tab pieces with an album song."""
    from riffgen.check.plagiarism import KIND_BAR2, fingerprints, track_events
    av = album.load_vocab()
    album_fp = set()
    for i in range(len(album)):
        s = song_from_view(album.view(i), av)
        for slot in s.tracks:
            album_fp |= {h for lv, kind, h, _, _ in fingerprints(track_events(s, slot)) if lv == 1 and kind == KIND_BAR2}
    dv = dstore.load_vocab()
    keep = []
    for i in chosen:
        s = song_from_view(dstore.view(i), dv)
        shared = 0
        for slot in s.tracks:
            shared += sum(1 for lv, kind, h, _, _ in fingerprints(track_events(s, slot))
                          if lv == 1 and kind == KIND_BAR2 and h in album_fp)
        if shared < min_shared:
            keep.append(i)
    if len(keep) < len(chosen):
        log.info("removed %d DadaGP songs that share tab with album songs", len(chosen) - len(keep))
    return keep


def _report(acfg, dstore: Store, chosen: list[int], song_score: dict, reports: Path) -> None:
    rep = reports / f"dadagp_mix_{acfg.album_id}.csv"
    rep.parent.mkdir(parents=True, exist_ok=True)
    with open(rep, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "artist", "title", "score", "best_riffs_bars", "tunings"])
        for i in chosen:
            s = dstore.songs[i]
            segs = [s["segments"][g] for g in song_score[i][1]]
            from riffgen.tokens import tunings as T
            w.writerow([s["id"], s["artist"], s["title"], round(song_score[i][0], 4),
                        " ".join(f"{g['start'] + 1}-{g['end']}" for g in segs),
                        " ".join(sorted({T.name_of(tuple(t["tuning"])) for t in s["tracks"].values()}))])
    # 20 random riffs as GP5 snippets
    from riffgen.gp import writer
    rng = np.random.default_rng(0)
    dv = dstore.load_vocab()
    snip_dir = reports / f"dadagp_mix_{acfg.album_id}"
    for i in rng.choice(chosen, size=min(20, len(chosen)), replace=False) if chosen else []:
        s = dstore.songs[int(i)]
        song = song_from_view(dstore.view(int(i)), dv)
        g = s["segments"][song_score[int(i)][1][0]]
        try:
            writer.write_bars(snip_dir / f"{s['artist']}__{Path(s['title']).stem}__{g['start'] + 1}-{g['end']}.gp5",
                              song.bars[g["start"]:g["end"]], song.tracks, song.bars[g["start"]].tempo, s["title"])
        except Exception as e:
            log.warning("snippet failed: %s", e)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("album")
    ap.add_argument("--method", default=None, help="features | embeddings")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)
    select_for_album(load_config(args.album), method=args.method, workers=args.workers)


if __name__ == "__main__":
    main()
