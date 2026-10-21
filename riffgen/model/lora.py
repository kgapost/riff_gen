"""LoRA adapters for album fine-tuning (option ``finetune.method: lora``).

With ~200 tabs per album, full fine-tuning of a 260-380M model overfits quickly and needs
one full copy of the weights per album. LoRA trains small low-rank updates instead: it is a
strong regulariser, the base model is not forgotten, and an album network is a few MB.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

DEFAULT_TARGETS = ("attn.qkv", "attn.proj", "mlp.w12", "mlp.w3", "cross.q", "cross.kv", "cross.proj")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        self.base = base
        self.rank = rank
        self.scale = alpha / rank
        self.A = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=base.weight.dtype))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=base.weight.dtype))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout)
        self.merged = False

    @property
    def weight(self):
        return self.base.weight

    @property
    def in_features(self):
        return self.base.in_features

    @property
    def out_features(self):
        return self.base.out_features

    def forward(self, x):
        y = self.base(x)
        if self.merged:
            return y
        return y + (self.drop(x) @ self.A.t().to(x.dtype) @ self.B.t().to(x.dtype)) * self.scale

    @torch.no_grad()
    def merge(self):
        if not self.merged:
            self.base.weight += (self.B @ self.A).to(self.base.weight.dtype) * self.scale
            self.merged = True


def apply_lora(model: nn.Module, rank: int = 16, alpha: float = 32.0, dropout: float = 0.05,
               targets=DEFAULT_TARGETS, train_norms: bool = False) -> int:
    """Wrap target Linear layers, freeze everything else. Returns the number of trainable params."""
    for p in model.parameters():
        p.requires_grad = False
    replaced = 0
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            full = f"{name}.{child_name}" if name else child_name
            if isinstance(child, nn.Linear) and any(full.endswith(t) for t in targets):
                setattr(module, child_name, LoRALinear(child, rank, alpha, dropout))
                replaced += 1
    if train_norms:
        for n, p in model.named_parameters():
            if "norm" in n or n.endswith(".n1.weight") or n.endswith(".n2.weight"):
                p.requires_grad = True
    if replaced == 0:
        raise ValueError("LoRA: no target layers found")
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def lora_state_dict(model: nn.Module) -> dict:
    return {k: v for k, v in model.state_dict().items() if k.endswith(".A") or k.endswith(".B")}


def merge_lora(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.merge()


def strip_lora(model: nn.Module) -> None:
    """Replace merged LoRALinear wrappers by their base Linear (plain state dict again)."""
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, LoRALinear):
                child.merge()
                setattr(module, child_name, child.base)
