"""Method 1: a weights file whose architecture spandrel detects."""

from __future__ import annotations

from pathlib import Path

import torch

from ..errors import SRModelError
from ..utils import sha256_file
from . import validated
from .base import SizeRule, SRSource, run_padded, run_tiled


class SpandrelSR(SRSource):
    def __init__(self, name: str, weights: Path, scale: int, tile: int | None = None):
        if not weights.is_file():
            raise SRModelError(f"SR model {name!r}: weights file not found: {weights}")
        try:
            import spandrel
        except ImportError as err:  # pragma: no cover - spandrel is a dependency
            raise SRModelError(f"spandrel is required for method 1: {err}") from None
        try:
            desc = spandrel.ModelLoader(device="cpu").load_from_file(str(weights))
        except spandrel.UnsupportedModelError:
            raise SRModelError(
                f"SR model {name!r}: spandrel could not detect the architecture of {weights.name}. "
                "Use method 2 (`module`: your PyTorch class) or method 3 (`images`: pre-computed SR images); "
                "see docs/adding_sr_models.md."
            ) from None
        except Exception as err:  # noqa: BLE001
            raise SRModelError(f"SR model {name!r}: {weights.name} could not be loaded ({type(err).__name__}: {err})") from None
        if not isinstance(desc, spandrel.ImageModelDescriptor):
            raise SRModelError(f"SR model {name!r}: {weights.name} is not an image-to-image model.")
        if desc.input_channels != 3 or desc.output_channels != 3:
            raise SRModelError(f"SR model {name!r}: {weights.name} must map RGB to RGB "
                               f"(found {desc.input_channels} -> {desc.output_channels} channels).")
        if desc.scale != scale:
            raise SRModelError(
                f"SR model {name!r}: {weights.name} upscales x{desc.scale}, but the configuration uses scale {scale}. "
                "Every SR model in a run must have exactly the configured scale."
            )
        sha = sha256_file(weights)
        info = validated.lookup(sha)
        super().__init__(
            name=name, kind="weights", scale=scale, key=f"weights:{sha}:tile{tile}", files={str(weights): sha},
            description=f"weights  {desc.architecture.name} (spandrel)  sha256 {sha[:7]}...",
            trained_for=info["trained_for"] if info else "unknown", display_file=weights.name,
        )
        self.architecture = desc.architecture.name
        self.descriptor = desc
        req = desc.size_requirements
        self.rule = SizeRule(minimum=max(1, req.minimum), multiple_of=max(1, req.multiple_of), square=req.square)
        self.tile = tile
        self.validated = info

    def to(self, device: torch.device) -> SpandrelSR:
        self.descriptor.to(device)
        self.descriptor.eval()
        return self

    @torch.no_grad()
    def upscale(self, lr: torch.Tensor) -> torch.Tensor:
        fn = self.descriptor
        if self.tile:
            y = run_tiled(fn, lr, self.scale, self.tile, self.rule)
        else:
            y = run_padded(fn, lr, self.scale, self.rule)
        return y.float().clamp(0.0, 1.0)
