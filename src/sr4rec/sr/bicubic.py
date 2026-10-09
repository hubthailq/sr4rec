"""The bicubic baseline: MATLAB-style bicubic upscaling (same routine as the downsampling)."""

from __future__ import annotations

import torch

from ..lr.resize import RESIZE_VERSION, upsample
from .base import SRSource


class BicubicSR(SRSource):
    def __init__(self, scale: int):
        super().__init__(name="bicubic", kind="bicubic", scale=scale, key=f"bicubic:{RESIZE_VERSION}:x{scale}",
                         description="built-in baseline", trained_for="n/a")

    def to(self, device: torch.device) -> BicubicSR:
        return self

    @torch.no_grad()
    def upscale(self, lr: torch.Tensor) -> torch.Tensor:
        return torch.stack([upsample(img.cpu(), self.scale) for img in lr]).to(lr.device)
