"""CPU latency: batch 1, fixed LR input size, 10 warm-up and 100 timed runs; median and p95."""

from __future__ import annotations

import time

import numpy as np
import torch

from .lr.letterbox import recognizer_input
from .rec.models import create_backbone
from .sr.base import SRSource

WARMUP = 10
RUNS = 100


def _time(fn, warmup: int = WARMUP, runs: int = RUNS) -> tuple[float, float]:
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            fn()
            times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(times)), float(np.percentile(times, 95))


def measure(sources: list[SRSource], backbones: list[str], num_classes: int, lr_size: int, threads: int,
            input_size: int, warmup: int = WARMUP, runs: int = RUNS) -> list[dict]:
    """Rows: component ('sr' | 'recognizer' | 'pipeline'), sr, backbone, median_ms, p95_ms."""
    old = torch.get_num_threads()
    torch.set_num_threads(threads)
    cpu = torch.device("cpu")
    x = torch.rand(1, 3, lr_size, lr_size, generator=torch.Generator().manual_seed(0))
    rows = []
    try:
        models = {}
        for b in backbones:
            m = create_backbone(b, num_classes, pretrained=False).eval()
            models[b] = m
            inp = torch.rand(1, 3, input_size, input_size)
            med, p95 = _time(lambda m=m, inp=inp: m(inp), warmup, runs)
            rows.append({"component": "recognizer", "sr": "", "backbone": b, "median_ms": med, "p95_ms": p95})
        for s in sources:
            if not s.computes:
                continue
            s.to(cpu)
            med, p95 = _time(lambda s=s: s.upscale(x), warmup, runs)
            rows.append({"component": "sr", "sr": s.name, "backbone": "", "median_ms": med, "p95_ms": p95})
            for b, m in models.items():
                def pipe(s=s, m=m):
                    y = s.upscale(x)[0]
                    return m(recognizer_input(y, input_size)[None])
                med, p95 = _time(pipe, warmup, runs)
                rows.append({"component": "pipeline", "sr": s.name, "backbone": b, "median_ms": med, "p95_ms": p95})
    finally:
        torch.set_num_threads(old)
    return rows
