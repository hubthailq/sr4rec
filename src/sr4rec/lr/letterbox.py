"""Recognizer input: letterbox to a square (aspect ratio kept) and ImageNet normalization.

Every image (bicubic, every SR output and HR) goes through exactly the same two steps:

1. Letterbox: the longer side is resized to ``size`` with antialiased bicubic interpolation, the
   shorter side keeps the aspect ratio, and the image is centred on a square canvas filled with
   the ImageNet mean colour. The image is never stretched.
2. ImageNet normalization: ``(x - IMAGENET_MEAN) / IMAGENET_STD`` per channel. The padding
   therefore becomes exactly 0 after normalization.

The same ImageNet statistics are used for every backbone.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PREPROCESS_VERSION = "letterbox-imagenet-mean-pad-v2"


def letterbox_geometry(height: int, width: int, size: int) -> tuple[int, int, int, int]:
    """Return (new_h, new_w, top, left) for placing an image into a ``size`` x ``size`` canvas."""
    ratio = size / max(height, width)
    new_h = min(size, max(1, round(height * ratio)))
    new_w = min(size, max(1, round(width * ratio)))
    return new_h, new_w, (size - new_h) // 2, (size - new_w) // 2


def letterbox(img: torch.Tensor, size: int, fill: tuple[float, float, float] = IMAGENET_MEAN) -> torch.Tensor:
    """(3, H, W) float in [0, 1] -> (3, size, size), aspect ratio kept, padded with ``fill``."""
    _, h, w = img.shape
    new_h, new_w, top, left = letterbox_geometry(h, w, size)
    if (new_h, new_w) != (h, w):
        img = F.interpolate(img[None], size=(new_h, new_w), mode="bicubic", align_corners=False, antialias=True)[0]
        img = img.clamp(0.0, 1.0)
    canvas = torch.tensor(fill, dtype=img.dtype).view(3, 1, 1).expand(3, size, size).clone()
    canvas[:, top : top + new_h, left : left + new_w] = img
    return canvas


def normalize(img: torch.Tensor) -> torch.Tensor:
    """ImageNet normalization of a (3, H, W) or (N, 3, H, W) tensor in [0, 1]."""
    mean = torch.tensor(IMAGENET_MEAN, dtype=img.dtype, device=img.device).view(-1, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=img.dtype, device=img.device).view(-1, 1, 1)
    return (img - mean) / std


def recognizer_input(img: torch.Tensor, size: int) -> torch.Tensor:
    """Letterbox followed by ImageNet normalization (what every recognizer receives)."""
    return normalize(letterbox(img, size))
