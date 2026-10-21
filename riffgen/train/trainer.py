"""The training loop shared by pretraining and album fine-tuning.

Details that matter in practice:

* gradient accumulation by *target tokens*: micro-batches are collected until
  ``tokens_per_step`` loss-weighted tokens are reached; the loss is summed and divided by
  that total, so a micro-batch with few targets (infill) does not get the same weight as a
  full one (a silent bias of "mean of micro-batch means");
* bf16 autocast on Ampere/Ada, fp16 + GradScaler on older GPUs, fp32 on CPU;
* cosine schedule with warmup, AdamW without weight decay on norms/embeddings/biases;
* NaN/inf losses skip the step; 10 in a row stop the run;
* CUDA out-of-memory skips the step and lowers the micro-batch token budget by 25 %;
* ``last.pt`` (with optimizer, for ``--resume``) and ``best.pt`` (by validation loss),
  written atomically; early stopping with patience (fine-tuning, D32);
* metrics.jsonl + optional TensorBoard.
"""

from __future__ import annotations

import math
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path

import torch

from riffgen.dataset.loader import TokenBudgetBatcher, make_loader
from riffgen.eval.metrics import evaluate_examples
from riffgen.train import checkpoint as ckpt_mod
from riffgen.utils.device import describe, to_device
from riffgen.utils.log import JsonlWriter, get_logger


@dataclass
class TrainConfig:
    lr: float = 3e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 1000
    max_steps: int = 20000
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    grad_clip: float = 1.0
    tokens_per_step: int = 131072
    micro_batch_tokens: int = 32768
    eval_every: int = 1000
    save_every: int = 1000
    log_every: int = 20
    workers: int = 4
    buffer_size: int = 256
    early_stopping_patience: int = 0          # evals without improvement; 0 = off
    select_metric: str = "val/loss"           # metric for best.pt
    compile: bool = False
    tensorboard: bool = False
    aux_weights: dict = field(default_factory=dict)

    @classmethod
    def from_cfg(cls, d: dict) -> "TrainConfig":
        d = dict(d or {})
        if "betas" in d:
            d["betas"] = tuple(d["betas"])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def lr_at(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / max(1, cfg.warmup_steps)
    prog = min(1.0, (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps))
    return cfg.lr * (cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * prog)))


def make_optimizer(model, cfg: TrainConfig, device) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or "emb" in n or "norm" in n or n.endswith(".gate") or "null" in n or "pool_q" in n:
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [{"params": decay, "weight_decay": cfg.weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    kw = {"lr": cfg.lr, "betas": cfg.betas, "eps": 1e-8}
    if device.type == "cuda":
        kw["fused"] = True
    return torch.optim.AdamW(groups, **kw)


class Trainer:
    def __init__(self, run_dir: str | Path, model, vocab, cfg: TrainConfig, stream, val_sets: dict[str, list[dict]],
                 device, amp, payload_fn, run_info: dict | None = None, resume: dict | None = None):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log = get_logger("riffgen.train", self.run_dir / "train.log")
        self.model = model
        self.vocab = vocab
        self.cfg = cfg
        self.stream = stream
        self.val_sets = val_sets
        self.device = device
        self.amp = amp
        self.payload_fn = payload_fn        # (model, **extra) -> checkpoint payload
        self.metrics = JsonlWriter(self.run_dir / "metrics.jsonl")
        self.opt = make_optimizer(model, cfg, device)
        self.scaler = torch.amp.GradScaler("cuda") if (amp == torch.float16 and device.type == "cuda") else None
        self.step = 0
        self.best = float("inf")
        self.bad_evals = 0
        self.examples_seen = 0
        self.epoch = 0
        self.tb = None
        if cfg.tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb = SummaryWriter(str(self.run_dir / "tb"))
            except Exception:
                self.log.warning("tensorboard not installed; logging to metrics.jsonl only")
        if resume:
            self.opt.load_state_dict(resume["optimizer"])
            self.step = resume["step"]
            self.best = resume.get("best", self.best)
            self.bad_evals = resume.get("bad_evals", 0)
            self.epoch = resume.get("epoch", 0)
            if self.scaler and resume.get("scaler"):
                self.scaler.load_state_dict(resume["scaler"])
        self.train_model = torch.compile(model) if cfg.compile else model
        if run_info:
            from riffgen.utils.config import save_json
            save_json(run_info, self.run_dir / "run_info.json")

    def _ctx(self):
        return torch.autocast(self.device.type, dtype=self.amp) if self.amp is not None else nullcontext()

    def evaluate(self) -> dict:
        res = {}
        for name, exs in self.val_sets.items():
            r = evaluate_examples(self.model, exs, self.vocab, self.device, self.amp, self.cfg.micro_batch_tokens)
            for k, v in r.items():
                res[f"{name}/{k}"] = v
        return res

    def save(self, name: str, with_optimizer: bool) -> None:
        extra = {"step": self.step, "best": self.best, "bad_evals": self.bad_evals, "epoch": self.epoch,
                 "examples_seen": self.examples_seen}
        if with_optimizer:
            extra["optimizer"] = self.opt.state_dict()
            if self.scaler:
                extra["scaler"] = self.scaler.state_dict()
        ckpt_mod.save(self.run_dir / name, self.payload_fn(self.model, **extra))

    def fit(self) -> dict:
        cfg = self.cfg
        self.log.info("device %s, amp %s, %.1fM params (%.1fM trainable)", describe(self.device), self.amp,
                      self.model.n_params() / 1e6, self.model.n_params(True) / 1e6)
        loader = make_loader(self.stream, cfg.workers)
        batcher = TokenBudgetBatcher(loader, cfg.micro_batch_tokens, self.vocab.pad_id, cfg.buffer_size,
                                     seed=self.step)
        it = iter(batcher)
        self.model.train()
        t0 = time.time()
        tok_count = 0
        nan_streak = 0
        last_eval = {}
        while self.step < cfg.max_steps:
            micro, wsum = [], 0.0
            while wsum < cfg.tokens_per_step:
                b = next(it)
                micro.append(b)
                wsum += float(b["loss"][:, 1:].sum())
                for m in b["meta"]:
                    self.epoch = max(self.epoch, m.get("epoch", 0))
            lr = lr_at(self.step, cfg)
            for g in self.opt.param_groups:
                g["lr"] = lr
            self.opt.zero_grad(set_to_none=True)
            loss_log, aux_log = 0.0, {}
            try:
                for b in micro:
                    bw = float(b["loss"][:, 1:].sum())
                    b = to_device(b, self.device)
                    with self._ctx():
                        out = self.train_model(b)
                        loss = out["loss_sum"] / wsum
                        for k, v in (out.get("aux") or {}).items():
                            w = cfg.aux_weights.get(k, 0.0)
                            if w:
                                loss = loss + w * v * (bw / wsum)
                            aux_log[k] = aux_log.get(k, 0.0) + float(v.detach()) * (bw / wsum)
                    if self.scaler:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    loss_log += float(out["loss_sum"].detach()) / wsum
                    tok_count += int(b["lengths"].sum())
                    self.examples_seen += len(b["meta"])
            except torch.cuda.OutOfMemoryError:
                self.opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                batcher.budget = int(batcher.budget * 0.75)
                self.log.warning("CUDA OOM at step %d; micro-batch budget lowered to %d tokens", self.step, batcher.budget)
                continue
            if not math.isfinite(loss_log):
                nan_streak += 1
                self.log.warning("non-finite loss at step %d (%d in a row), step skipped", self.step, nan_streak)
                self.opt.zero_grad(set_to_none=True)
                if nan_streak >= 10:
                    raise RuntimeError("loss is NaN/inf 10 steps in a row; lower the learning rate")
                continue
            nan_streak = 0
            if self.scaler:
                self.scaler.unscale_(self.opt)
            gnorm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip))
            if self.scaler:
                self.scaler.step(self.opt)
                self.scaler.update()
            else:
                self.opt.step()
            self.step += 1
            if self.step % cfg.log_every == 0:
                dt = time.time() - t0
                rec = {"step": self.step, "loss": round(loss_log, 4), "lr": lr, "grad_norm": round(gnorm, 3),
                       "tok_per_s": round(tok_count / max(dt, 1e-6)), "epoch": self.epoch,
                       **{f"aux_{k}": round(v, 4) for k, v in aux_log.items()}}
                if self.device.type == "cuda":
                    rec["mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
                self.log.info(" ".join(f"{k}={v}" for k, v in rec.items()))
                self.metrics.write({"split": "train", **rec})
                if self.tb:
                    for k, v in rec.items():
                        if isinstance(v, (int, float)) and k != "step":
                            self.tb.add_scalar(f"train/{k}", v, self.step)
                t0, tok_count = time.time(), 0
            if self.step % cfg.eval_every == 0 or self.step == cfg.max_steps:
                last_eval = self.evaluate()
                if last_eval:
                    self.log.info("eval step %d: %s", self.step,
                                  " ".join(f"{k}={v:.4f}" for k, v in last_eval.items() if isinstance(v, float)))
                    self.metrics.write({"split": "eval", "step": self.step, **last_eval})
                    if self.tb:
                        for k, v in last_eval.items():
                            self.tb.add_scalar(k, v, self.step)
                    score = last_eval.get(cfg.select_metric)
                    if score is not None and score < self.best:
                        self.best = score
                        self.bad_evals = 0
                        self.save("best.pt", with_optimizer=False)
                        self.log.info("new best %s = %.4f", cfg.select_metric, score)
                    elif score is not None:
                        self.bad_evals += 1
                        if cfg.early_stopping_patience and self.bad_evals >= cfg.early_stopping_patience:
                            self.log.info("early stopping: %s did not improve for %d evals", cfg.select_metric,
                                          self.bad_evals)
                            self.save("last.pt", with_optimizer=True)
                            break
            if self.step % cfg.save_every == 0:
                self.save("last.pt", with_optimizer=True)
        self.save("last.pt", with_optimizer=True)
        if not (self.run_dir / "best.pt").exists():
            self.save("best.pt", with_optimizer=False)
        return {"step": self.step, "best": self.best, **last_eval}
