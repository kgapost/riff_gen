"""Device and precision choices.

* RTX 4090 (Ada, sm89): bf16 autocast, flash attention, TF32 matmuls.
* GTX 1650 (Turing, sm75): no bf16. fp16 inference works; a model trained in bf16 can
  overflow in fp16, so the generator checks the logits and falls back to fp32 if needed.
* GTX 1050 Ti and other Pascal cards (sm61): fp16 is very slow there, use fp32. Recent
  PyTorch CUDA 12.8+/13 wheels may not include Pascal kernels; use a cu126 build.
"""

from __future__ import annotations

import torch


def pick_device(pref: str = "auto") -> torch.device:
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def amp_dtype(device: torch.device, pref: str = "auto") -> torch.dtype | None:
    """Autocast dtype, or None for plain fp32."""
    if pref == "fp32":
        return None
    if pref == "bf16":
        return torch.bfloat16
    if pref == "fp16":
        return torch.float16
    if device.type != "cuda":
        return None
    major, _ = torch.cuda.get_device_capability(device)
    if major >= 8 and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if major >= 7:
        return torch.float16
    return None


def setup_backends(device: torch.device) -> None:
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def describe(device: torch.device) -> str:
    if device.type == "cuda":
        p = torch.cuda.get_device_properties(device)
        return f"{p.name} ({p.total_memory / 2**30:.1f} GB, sm{p.major}{p.minor})"
    return str(device)


def to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        elif isinstance(v, dict):
            out[k] = to_device(v, device)
        else:
            out[k] = v
    return out
