"""Device selection and out-of-memory handling.

Priority for ``runtime.device: auto``: CUDA GPU, then Apple GPU (MPS), then CPU. A requested
device that is not available falls back along the same order with a warning. Running out of
GPU memory never stops a run: callers catch :func:`is_oom` errors and degrade (smaller SR
tiles, gradient accumulation, smaller evaluation batches) and finally continue on the CPU.
"""

from __future__ import annotations

import gc

import torch

ORDER = ("cuda", "mps", "cpu")


def available(kind: str) -> bool:
    if kind == "cuda":
        return torch.cuda.is_available()
    if kind == "mps":
        mps = getattr(torch.backends, "mps", None)
        return bool(mps and mps.is_available())
    return kind == "cpu"


def resolve(preference: str) -> tuple[torch.device, list[str]]:
    """Return the device to use and warnings about any fallback."""
    if preference == "auto":
        kind = next(k for k in ORDER if available(k))
        return torch.device(kind), []
    if available(preference):
        return torch.device(preference), []
    start = ORDER.index(preference) + 1
    kind = next(k for k in ORDER[start:] if available(k))
    names = {"cuda": "a CUDA GPU", "mps": "an Apple GPU (MPS)", "cpu": "the CPU"}
    return torch.device(kind), [f"runtime.device is '{preference}' but {names[preference]} is not available; "
                                f"SR4Rec used {names[kind]} instead."]


def is_oom(err: BaseException) -> bool:
    """True for out-of-memory errors of CUDA, MPS and the CPU allocator."""
    if isinstance(err, getattr(torch.cuda, "OutOfMemoryError", ())):
        return True
    if not isinstance(err, RuntimeError):
        return False
    msg = str(err).lower()
    return "out of memory" in msg or "can't allocate memory" in msg or "failed to allocate" in msg


def free_memory(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    if device.type == "mps":
        return "Apple GPU (MPS)"
    from .fingerprint import cpu_name

    return cpu_name()


CPU = torch.device("cpu")
