"""Fixed image-size bins, based on the short side of the LR image given to SR."""

from __future__ import annotations

import numpy as np

SIZE_BINS = ("<32", "32-63", "64-127", ">=128")
SIZE_BIN_LABELS = {"<32": "<32 px", "32-63": "32–63 px", "64-127": "64–127 px", ">=128": "≥128 px"}
MIN_IMAGES_PER_BIN = 50


def size_bin(short_side: int) -> str:
    if short_side < 32:
        return "<32"
    if short_side < 64:
        return "32-63"
    if short_side < 128:
        return "64-127"
    return ">=128"


def lr_size(width: int, height: int, mode: str, scale: int) -> tuple[int, int]:
    """Size (width, height) of the LR image that SR receives."""
    if mode == "synthetic":
        return width // scale, height // scale
    return width, height


def assign_bins(short_sides) -> np.ndarray:
    return np.array([size_bin(int(s)) for s in short_sides], dtype=object)
