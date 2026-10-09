"""Common interface of every SR source and the shared inference helpers."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

TILE_OVERLAP = 16


@dataclass
class SizeRule:
    """Input-size requirements of an SR network."""

    minimum: int = 1
    multiple_of: int = 1
    square: bool = False


@dataclass
class SRSource:
    """Base class. ``kind`` is one of bicubic, weights, module, images."""

    name: str
    kind: str
    scale: int
    key: str = ""  # content key used for caching and fingerprinting
    files: dict[str, str] = field(default_factory=dict)  # file path -> SHA-256
    description: str = ""  # shown by --dry-run
    trained_for: str = "unknown"
    display_file: str = "n/a"

    @property
    def computes(self) -> bool:
        """True if SR4Rec runs this model itself (False for pre-computed images)."""
        return self.kind != "images"

    def to(self, device: torch.device) -> SRSource:  # pragma: no cover - overridden
        return self

    def upscale(self, lr: torch.Tensor) -> torch.Tensor:
        """(1, 3, h, w) in [0, 1] -> (1, 3, h*scale, w*scale) in [0, 1]."""
        raise NotImplementedError


def _pad_amount(size: int, rule: SizeRule, other: int) -> int:
    target = max(size, rule.minimum)
    if rule.square:
        target = max(target, other, rule.minimum)
    if rule.multiple_of > 1:
        target = -(-target // rule.multiple_of) * rule.multiple_of
    return target - size


def pad_for(x: torch.Tensor, rule: SizeRule) -> tuple[torch.Tensor, int, int]:
    """Pad (N, 3, h, w) at the bottom/right so that it meets ``rule``. Reflection is used
    when possible, replication otherwise (reflection needs pad < size)."""
    h, w = x.shape[-2:]
    ph = _pad_amount(h, rule, w)
    pw = _pad_amount(w, rule, h)
    if rule.square:
        side = max(h + ph, w + pw)
        ph, pw = side - h, side - w
    if ph == 0 and pw == 0:
        return x, 0, 0
    while ph > 0 or pw > 0:  # reflect in steps so that large pads on tiny images still work
        sh = min(ph, x.shape[-2] - 1)
        sw = min(pw, x.shape[-1] - 1)
        if sh <= 0 and sw <= 0:
            x = F.pad(x, (0, pw, 0, ph), mode="replicate")
            break
        x = F.pad(x, (0, max(sw, 0), 0, max(sh, 0)), mode="reflect")
        ph -= max(sh, 0)
        pw -= max(sw, 0)
    return x, h, w


def run_padded(fn, x: torch.Tensor, scale: int, rule: SizeRule) -> torch.Tensor:
    padded, h, w = pad_for(x, rule)
    y = fn(padded)
    if h:
        y = y[..., : h * scale, : w * scale]
    return y


def run_tiled(fn, x: torch.Tensor, scale: int, tile: int, rule: SizeRule) -> torch.Tensor:
    """Tiled inference with ``TILE_OVERLAP`` px overlap; overlapping outputs are averaged."""
    _, c, h, w = x.shape
    if h <= tile and w <= tile:
        return run_padded(fn, x, scale, rule)
    step = tile - TILE_OVERLAP
    out = torch.zeros(x.shape[0], c, h * scale, w * scale, dtype=torch.float32, device=x.device)
    weight = torch.zeros_like(out[:, :1])
    ys = list(range(0, max(h - tile, 0) + 1, step))
    xs = list(range(0, max(w - tile, 0) + 1, step))
    if ys[-1] + tile < h:
        ys.append(h - tile)
    if xs[-1] + tile < w:
        xs.append(w - tile)
    for y0 in ys:
        for x0 in xs:
            y1, x1 = min(y0 + tile, h), min(x0 + tile, w)
            patch = run_padded(fn, x[..., y0:y1, x0:x1], scale, rule).float()
            out[..., y0 * scale : y1 * scale, x0 * scale : x1 * scale] += patch
            weight[..., y0 * scale : y1 * scale, x0 * scale : x1 * scale] += 1.0
    return out / weight
