"""MATLAB-compatible bicubic ``imresize`` (with antialiasing when downsampling).

This is an independent implementation of the algorithm of MATLAB's
``imresize(img, scale, 'bicubic')``: cubic kernel with a = -0.5, kernel width 4
(widened by 1/scale when downsampling, which is MATLAB's antialiasing), and
symmetric boundary handling computed with MATLAB's index-folding rule
(``aux = [1:n, n:-1:1]``), so it is also correct for images smaller than the
kernel. The same routine is used for the synthetic LR images and for the
bicubic baseline. The result is validated against the BasicSR port of the same
MATLAB routine (tests/test_resize_letterbox.py).
"""

from __future__ import annotations

import math

import numpy as np
import torch

RESIZE_VERSION = "matlab-bicubic-v1"


def _cubic(x: torch.Tensor) -> torch.Tensor:
    a = x.abs()
    a2, a3 = a * a, a * a * a
    return (1.5 * a3 - 2.5 * a2 + 1.0) * (a <= 1).to(x.dtype) + (-0.5 * a3 + 2.5 * a2 - 4.0 * a + 2.0) * (
        ((a > 1) & (a <= 2)).to(x.dtype)
    )


def _weights_indices(in_len: int, out_len: int, scale: float, antialias: bool) -> tuple[torch.Tensor, torch.Tensor]:
    kernel_width = 4.0
    if scale < 1 and antialias:
        kernel_width = kernel_width / scale
    x = torch.arange(1, out_len + 1, dtype=torch.float64)
    u = x / scale + 0.5 * (1.0 - 1.0 / scale)
    left = torch.floor(u - kernel_width / 2.0)
    p = int(math.ceil(kernel_width)) + 2
    indices = left[:, None] + torch.arange(p, dtype=torch.float64)[None, :]
    dist = u[:, None] - indices
    if scale < 1 and antialias:
        weights = scale * _cubic(dist * scale)
    else:
        weights = _cubic(dist)
    weights = weights / weights.sum(dim=1, keepdim=True)
    aux = torch.cat([torch.arange(in_len), torch.arange(in_len - 1, -1, -1)])  # 0-based [1:n, n:-1:1]
    idx = aux[torch.remainder(indices.long() - 1, 2 * in_len)]
    keep = (weights != 0).any(dim=0)
    return weights[:, keep], idx[:, keep]


def _resize_dim(img: torch.Tensor, dim: int, out_len: int, scale: float, antialias: bool) -> torch.Tensor:
    """Resize ``img`` (C, H, W), float64, along ``dim`` (1 = H, 2 = W)."""
    w, idx = _weights_indices(img.shape[dim], out_len, scale, antialias)
    if dim == 1:
        gathered = img[:, idx, :]  # (C, out, p, W)
        return (gathered * w[None, :, :, None]).sum(dim=2)
    gathered = img[:, :, idx]  # (C, H, out, p)
    return (gathered * w[None, None, :, :]).sum(dim=3)


@torch.no_grad()
def imresize(img: torch.Tensor | np.ndarray, scale: float, antialias: bool = True) -> torch.Tensor:
    """Resize like MATLAB ``imresize(img, scale, 'bicubic')``.

    ``img``: float tensor (3, H, W) in [0, 1] or uint8/float array (H, W, 3).
    Returns a float32 tensor (3, ceil(H*scale), ceil(W*scale)), not clamped or rounded.
    """
    if isinstance(img, np.ndarray):
        arr = img.astype(np.float64)
        if img.dtype == np.uint8:
            arr = arr / 255.0
        t = torch.from_numpy(arr.transpose(2, 0, 1).copy())
    else:
        t = img.to(torch.float64)
    _, h, w = t.shape
    out_h, out_w = int(math.ceil(h * scale - 1e-9)), int(math.ceil(w * scale - 1e-9))
    t = _resize_dim(t, 1, out_h, scale, antialias)
    t = _resize_dim(t, 2, out_w, scale, antialias)
    return t.to(torch.float32)


def downsample(img: torch.Tensor, scale: int) -> torch.Tensor:
    """HR (3, H, W) with H and W multiples of ``scale`` -> LR (3, H/scale, W/scale), clamped to [0, 1]."""
    return imresize(img, 1.0 / scale).clamp_(0.0, 1.0)


def upsample(img: torch.Tensor, scale: int) -> torch.Tensor:
    """LR (3, h, w) -> bicubic (3, h*scale, w*scale), clamped to [0, 1]."""
    return imresize(img, float(scale)).clamp_(0.0, 1.0)
