"""Compute (cached) SR images for one SR source, surviving out-of-memory errors.

On an out-of-memory error the image is retried with tiled inference (tiles of half the
current size, down to 64 px), and if even that fails the model continues on the CPU for the
rest of the images. Every fallback is counted and reported.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import torch

from ..device import CPU, free_memory, is_oom
from ..utils import load_rgb, pil_to_tensor, png_name, save_png
from .base import SRSource

MIN_AUTO_TILE = 64


@dataclass
class SRRunStats:
    """Fallbacks used while computing one SR model."""

    images: int = 0
    tiled: int = 0
    on_cpu: int = 0
    tile: int | None = None
    notes: list[str] = field(default_factory=list)

    def warning(self, name: str, device: torch.device) -> str | None:
        if not (self.tiled or self.on_cpu):
            return None
        parts = []
        if self.tiled:
            parts.append(f"{self.tiled:,} image(s) with automatic tiling (tile {self.tile} px, 16 px overlap)")
        if self.on_cpu:
            parts.append(f"{self.on_cpu:,} image(s) on the CPU")
        return (f"SR model {name}: {device.type.upper()} memory ran out, so SR4Rec processed "
                + " and ".join(parts) + ". Results are valid; tiling can change pixels slightly at tile borders.")


class RobustRunner:
    """Runs ``source`` on one image at a time with the out-of-memory fallbacks."""

    def __init__(self, source: SRSource, device: torch.device):
        self.source = source
        self.device = device
        self.stats = SRRunStats(tile=getattr(source, "tile", None))
        self.cpu_only = device.type == "cpu"
        self.tileable = hasattr(source, "tile")
        source.to(device)

    def _call(self, lr: torch.Tensor, device: torch.device) -> torch.Tensor:
        if self.tileable:
            old = self.source.tile
            self.source.tile = self.stats.tile
            try:
                return self.source.upscale(lr.to(device)).cpu()
            finally:
                self.source.tile = old
        return self.source.upscale(lr.to(device)).cpu()

    @torch.no_grad()
    def upscale(self, lr: torch.Tensor) -> torch.Tensor:
        """``lr``: (1, 3, h, w) on the CPU. Returns the SR image on the CPU."""
        self.stats.images += 1
        device = CPU if self.cpu_only else self.device
        while True:
            try:
                out = self._call(lr, device)
                if self.stats.tile is not None and self.stats.tile != getattr(self.source, "tile", None):
                    self.stats.tiled += 1
                if device.type == "cpu" and self.device.type != "cpu":
                    self.stats.on_cpu += 1
                return out
            except RuntimeError as err:
                if not is_oom(err):
                    raise
                free_memory(device)
                current = self.stats.tile or max(lr.shape[-2:])
                if self.tileable and current // 2 >= MIN_AUTO_TILE:
                    self.stats.tile = current // 2
                    continue
                if device.type != "cpu":  # last resort: the CPU, for this and every later image
                    self.cpu_only = True
                    self.source.to(CPU)
                    device = CPU
                    continue
                raise

    def close(self) -> None:
        self.source.to(CPU)
        free_memory(self.device)


def compute_sr(source: SRSource, items: list[tuple[str, Path]], out_dir: Path, device: torch.device,
               log: Callable[[str], None] | None = None) -> tuple[float, SRRunStats]:
    """Run ``source`` on every (dataset path, LR file) pair and store PNGs under ``out_dir``.
    Existing outputs are reused. Returns the seconds spent on new images and the fallback stats."""
    todo = [(p, f) for p, f in items if not (out_dir / png_name(p)).is_file()]
    if not todo:
        return 0.0, SRRunStats()
    runner = RobustRunner(source, device)
    start = time.perf_counter()
    last = start
    try:
        for i, (path, lr_file) in enumerate(todo, 1):
            lr = pil_to_tensor(load_rgb(lr_file))[None]
            sr = runner.upscale(lr)[0]
            save_png(sr, out_dir / png_name(path))
            now = time.perf_counter()
            if log and (now - last > 30 or i == len(todo)):
                log(f"    {source.name}: {i:,}/{len(todo):,} images")
                last = now
    finally:
        runner.close()
    return time.perf_counter() - start, runner.stats
