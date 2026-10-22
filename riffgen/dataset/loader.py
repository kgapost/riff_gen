"""Streams of training examples and token-budget batching.

Determinism (D46): the epoch plan (which song, which task) is made from the seed and the
epoch number; example ``i`` of an epoch is built with ``derive(seed, "example", epoch, i)``.
Worker ``w`` of ``W`` builds the examples with ``i % W == w``, so the set of examples does
not depend on the number of workers (a classic bug is every worker producing the same
stream).

Batching: examples have very different lengths (short songs, infill examples), so we sort
a buffer of examples by length and fill micro-batches up to a token budget instead of a
fixed batch size. The trainer accumulates micro-batches until ``tokens_per_step`` target
tokens are reached and normalises the loss by that count (not by micro-batch means).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from riffgen.dataset.sampler import ExampleBuilder, SamplerConfig
from riffgen.dataset.serialize import FEATURE_KEYS
from riffgen.dataset.store import Store
from riffgen.tokens.vocab import Vocab
from riffgen.utils.log import JsonlWriter
from riffgen.utils.seed import derive


@dataclass
class SourceSpec:
    path: str
    split: str | None = "train"
    share: float | None = None          # fixed share of examples (e.g. 0.15 DadaGP mix, D47)
    weight: float = 1.0                 # relative weight among sources without a fixed share
    ids: list[str] | None = None        # restrict to these song ids
    bars_per_example: float | str = "auto"
    task_mix: dict | None = None


@dataclass
class StreamConfig:
    sources: list[SourceSpec]
    seed: int = 1234
    sampler_log: str | None = None
    sampler_log_fraction: float = 0.01
    max_epochs: int | None = None


class _Sources:
    """Opened stores (one per process; memory-mapped)."""

    def __init__(self, specs: list[SourceSpec]):
        self.stores = [Store(s.path) for s in specs]
        self.songs = []
        for spec, st in zip(specs, self.stores):
            wanted = set(spec.ids) if spec.ids else None
            self.songs.append(st.indices(spec.split, wanted))


def bars_per_example(store: Store, max_len: int) -> float:
    """Bars one example covers: ~one example per context window of every song, per epoch."""
    event_tokens_per_bar = max(1, store.info.get("tokens", 0)) / max(1, store.info.get("bars", 1))
    tokens_per_bar = 1.3 * event_tokens_per_bar + 2    # + wait tokens and new_measure
    return max(4.0, 0.8 * max_len / tokens_per_bar)


def epoch_plan(specs: list[SourceSpec], srcs: _Sources, seed: int, epoch: int, max_len: int,
               default_mix: dict) -> list[tuple[int, int, str]]:
    plan: list[tuple[int, int, str]] = []
    rng = derive(seed, "plan", epoch)
    free = [k for k, s in enumerate(specs) if s.share is None]
    fixed = [k for k, s in enumerate(specs) if s.share is not None]
    for k in free:
        st, spec = srcs.stores[k], specs[k]
        bpe = bars_per_example(st, max_len) if spec.bars_per_example == "auto" else float(spec.bars_per_example)
        for idx in srcs.songs[k]:
            n = max(1, int(round(spec.weight * st.songs[idx]["n_bars"] / bpe)))
            plan += [(k, idx, "")] * n
    base = len(plan)
    total_share = sum(specs[k].share for k in fixed)
    if total_share >= 1.0:
        raise ValueError("source shares must add up to less than 1")
    for k in fixed:
        if not srcs.songs[k]:
            continue
        n = int(round(base * specs[k].share / max(1e-9, 1.0 - total_share))) if base else 1000
        pick = rng.choice(srcs.songs[k], size=n, replace=True)
        plan += [(k, int(i), "") for i in pick]
    order = rng.permutation(len(plan))
    plan = [plan[i] for i in order]
    out = []
    for i, (k, idx, _) in enumerate(plan):
        mix = specs[k].task_mix or default_mix
        tasks = list(mix)
        p = np.array([mix[t] for t in tasks], float)
        t = tasks[int(derive(seed, "task", epoch, i).choice(len(tasks), p=p / p.sum()))]
        out.append((k, idx, t))
    return out


class TrainStream(IterableDataset):
    def __init__(self, cfg: StreamConfig, vocab: Vocab, sampler_cfg: SamplerConfig, start_epoch: int = 0,
                 skip_examples: int = 0):
        super().__init__()
        self.cfg = cfg
        self.vocab = vocab
        self.sampler_cfg = sampler_cfg
        self.start_epoch = start_epoch
        self.skip = skip_examples

    def __iter__(self) -> Iterator[dict]:
        info = get_worker_info()
        w, W = (info.id, info.num_workers) if info else (0, 1)
        srcs = _Sources(self.cfg.sources)
        builder = ExampleBuilder(self.vocab, self.sampler_cfg)
        log = JsonlWriter(self.cfg.sampler_log) if (self.cfg.sampler_log and w == 0) else None
        epoch = self.start_epoch
        while self.cfg.max_epochs is None or epoch < self.cfg.max_epochs:
            plan = epoch_plan(self.cfg.sources, srcs, self.cfg.seed, epoch, self.sampler_cfg.max_len,
                              self.sampler_cfg.task_mix)
            for i, (k, idx, task) in enumerate(plan):
                if i % W != w or (epoch == self.start_epoch and i < self.skip):
                    continue
                rng = derive(self.cfg.seed, "example", epoch, i)
                ex = builder.build(srcs.stores[k].view(idx), task, rng)
                if ex is None:
                    continue
                ex["meta"].update(epoch=epoch, index=i, source=k, n_plan=len(plan))
                if log and rng.random() < self.cfg.sampler_log_fraction:
                    log.write(ex["meta"])
                yield ex
            epoch += 1


def build_fixed_examples(specs: list[dict], store: Store, vocab: Vocab, sampler_cfg: SamplerConfig) -> list[dict]:
    builder = ExampleBuilder(vocab, sampler_cfg)
    out = []
    for spec in specs:
        idx = store.by_id.get(spec["song"], spec.get("store_song"))
        if idx is None:
            continue
        ex = builder.build(store.view(idx), spec["task"], np.random.default_rng(0), spec=spec, augment=False)
        if ex is not None:
            ex["meta"]["spec"] = spec
            out.append(ex)
    return out


# ---------------------------------------------------------------------------
# batching

def _pad_len(n: int, multiple: int) -> int:
    return int(math.ceil(n / multiple) * multiple)


def collate(examples: list[dict], pad_id: int, pad_multiple: int = 64) -> dict:
    T = _pad_len(max(len(e["ids"]) for e in examples), pad_multiple)
    B = len(examples)
    batch = {
        "ids": torch.full((B, T), pad_id, dtype=torch.long),
        "loss": torch.zeros((B, T), dtype=torch.float32),
        "lengths": torch.tensor([len(e["ids"]) for e in examples], dtype=torch.long),
    }
    for k in FEATURE_KEYS:
        dt = torch.float32 if k in ("time", "phase") else torch.long
        batch[k] = torch.zeros((B, T), dtype=dt)
    for i, e in enumerate(examples):
        n = len(e["ids"])
        batch["ids"][i, :n] = torch.from_numpy(e["ids"].astype(np.int64))
        batch["loss"][i, :n] = torch.from_numpy(e["loss"])
        for k in FEATURE_KEYS:
            batch[k][i, :n] = torch.from_numpy(np.asarray(e[k]).astype(np.float32 if k in ("time", "phase") else np.int64))
        # padding continues the time axis so rotary positions stay monotone
        if n < T and n > 0:
            batch["time"][i, n:] = batch["time"][i, n - 1]
    if "song" in examples[0]:
        batch["song"] = collate_song([e["song"] for e in examples], T)
    batch["meta"] = [e["meta"] for e in examples]
    return batch


SONG_INT_KEYS = ("bar_ids", "bar_inst", "bar_pitch", "bar_fret", "bar_sfb", "bar_beat", "bar_pos")


def collate_song(songs: list[dict], T: int) -> dict:
    B = len(songs)
    NB = max(1, max(len(s["mask_kind"]) for s in songs))
    Lb = songs[0]["bar_ids"].shape[1]
    out = {k: torch.zeros((B, NB, Lb), dtype=torch.long) for k in SONG_INT_KEYS}
    out["bar_len"] = torch.zeros((B, NB), dtype=torch.long)
    out["mask_kind"] = torch.full((B, NB), 3, dtype=torch.long)      # 3 = padding
    for k in ("rel", "tempo", "ts_num", "ts_den", "phrase"):
        out[k] = torch.zeros((B, NB), dtype=torch.long)
    out["bar_lo"] = torch.tensor([int(s.get("bar_lo", 0)) for s in songs], dtype=torch.long)
    out["section"] = torch.full((B, NB), -1, dtype=torch.long)
    out["aux_pc"] = torch.zeros((B, NB, 12))
    out["aux_onset"] = torch.zeros((B, NB, songs[0]["aux_onset"].shape[1]))
    out["aux_mask"] = torch.zeros((B, NB), dtype=torch.bool)
    out["tok_bar"] = torch.full((B, T), -1, dtype=torch.long)
    pairs = []
    for i, s in enumerate(songs):
        nb = len(s["mask_kind"])
        for k in SONG_INT_KEYS:
            out[k][i, :nb] = torch.from_numpy(s[k].astype(np.int64))
        out["bar_len"][i, :nb] = torch.from_numpy(s["bar_len"].astype(np.int64))
        out["mask_kind"][i, :nb] = torch.from_numpy(s["mask_kind"].astype(np.int64))
        for k in ("rel", "tempo", "ts_num", "ts_den", "section", "phrase"):
            out[k][i, :nb] = torch.from_numpy(s[k].astype(np.int64))
        out["aux_pc"][i, :nb] = torch.from_numpy(s["aux_pc"])
        out["aux_onset"][i, :nb] = torch.from_numpy(s["aux_onset"])
        out["aux_mask"][i, :nb] = torch.from_numpy(s["aux_mask"])
        n = len(s["tok_bar"])
        out["tok_bar"][i, :n] = torch.from_numpy(s["tok_bar"].astype(np.int64))
        if len(s["pairs"]):
            p = torch.from_numpy(s["pairs"].astype(np.int64))
            pairs.append(torch.cat([torch.full((len(p), 1), i, dtype=torch.long), p], 1))
    out["pairs"] = torch.cat(pairs) if pairs else torch.zeros((0, 3), dtype=torch.long)
    return out


class TokenBudgetBatcher:
    """Sort a buffer of examples by length and cut it into micro-batches under a token budget."""

    def __init__(self, examples, micro_batch_tokens: int, pad_id: int, buffer_size: int = 256, seed: int = 0,
                 pad_multiple: int = 64, shuffle: bool = True, max_batch: int = 64):
        self.examples = examples
        self.budget = micro_batch_tokens
        self.pad_id = pad_id
        self.buffer_size = buffer_size
        self.seed = seed
        self.pad_multiple = pad_multiple
        self.shuffle = shuffle
        self.max_batch = max_batch
        self.n_flush = 0

    def _batches(self, buf: list[dict]) -> list[list[dict]]:
        buf.sort(key=lambda e: len(e["ids"]))
        batches, cur = [], []
        for e in buf:
            L = _pad_len(len(e["ids"]), self.pad_multiple)
            if cur and (L * (len(cur) + 1) > self.budget or len(cur) >= self.max_batch):
                batches.append(cur)
                cur = []
            cur.append(e)
        if cur:
            batches.append(cur)
        if self.shuffle:
            order = derive(self.seed, "batches", self.n_flush).permutation(len(batches))
            batches = [batches[i] for i in order]
        self.n_flush += 1
        return batches

    def __iter__(self):
        buf: list[dict] = []
        for ex in self.examples:
            buf.append(ex)
            if len(buf) >= self.buffer_size:
                for b in self._batches(buf):
                    yield collate(b, self.pad_id, self.pad_multiple)
                buf = []
        if buf:
            for b in self._batches(buf):
                yield collate(b, self.pad_id, self.pad_multiple)


def _identity(x):
    return x


def make_loader(stream: TrainStream, workers: int, prefetch: int = 4):
    """Workers return raw example dicts (no tensor conversion); batching happens in the main process."""
    from torch.utils.data import DataLoader
    return DataLoader(stream, batch_size=None, num_workers=workers, persistent_workers=workers > 0,
                      prefetch_factor=prefetch if workers > 0 else None, pin_memory=False, collate_fn=_identity)
