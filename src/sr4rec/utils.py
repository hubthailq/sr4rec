"""Small helpers shared by several modules."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_key(*parts: Any) -> str:
    """SHA-256 of a JSON encoding of ``parts`` (used for cache keys)."""
    return sha256_text(json.dumps(parts, sort_keys=True, default=str))


def round_half_up(x: float) -> int:
    """Round to the nearest integer, halves away from zero (floor(x + 0.5) for x >= 0)."""
    return int(math.floor(x + 0.5))


def png_name(path: str) -> str:
    """Relative path of the PNG file that stores a processed version of ``path``."""
    return str(Path(path).with_suffix(".png").as_posix())


HIGH_BIT_DEPTH_MODES = ("I;16", "I;16B", "I;16L", "I;16N", "I", "F")


def to_rgb(im: Image.Image) -> Image.Image:
    """Convert any PIL mode to 8-bit RGB. 16-bit and 32-bit single-channel images are scaled to
    0-255 (PIL's own conversion would clip them)."""
    if im.mode in HIGH_BIT_DEPTH_MODES:
        arr = np.asarray(im, dtype=np.float64)
        peak = float(arr.max(initial=0))
        if im.mode == "F":
            top = 1.0 if peak <= 1.0 else peak
        else:  # smallest common bit depth that holds the data (8, 10, 12, 14 or 16 bit)
            top = next((t for t in (255.0, 1023.0, 4095.0, 16383.0, 65535.0) if peak <= t), peak)
        im = Image.fromarray(np.clip(np.round(arr / top * 255.0), 0, 255).astype(np.uint8), mode="L")
    return im.convert("RGB")


def load_rgb(path: str | Path) -> Image.Image:
    with Image.open(path) as im:
        return to_rgb(im)


def pil_to_tensor(im: Image.Image) -> torch.Tensor:
    """PIL RGB image -> float32 tensor (3, H, W) in [0, 1]."""
    arr = np.asarray(im, dtype=np.uint8)
    return torch.from_numpy(arr.copy()).permute(2, 0, 1).float().div_(255.0)


def tensor_to_uint8(t: torch.Tensor) -> np.ndarray:
    """Float tensor (3, H, W) in [0, 1] -> uint8 array (H, W, 3), rounded."""
    t = t.detach().clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8)
    return t.permute(1, 2, 0).cpu().numpy()


def save_png(arr_or_tensor: np.ndarray | torch.Tensor, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = tensor_to_uint8(arr_or_tensor) if isinstance(arr_or_tensor, torch.Tensor) else arr_or_tensor
    tmp = path.with_name(path.name + ".tmp")
    Image.fromarray(arr).save(tmp, format="PNG")
    tmp.replace(path)


def mod_crop(im: Image.Image, scale: int) -> Image.Image:
    """Crop the right and bottom borders so that both sides are multiples of ``scale``."""
    w, h = im.size
    return im.crop((0, 0, w - w % scale, h - h % scale))


def chunks(seq: list, size: int) -> Iterable[list]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


BACKBONE_DISPLAY = {
    "resnet18": "ResNet-18",
    "mobilenetv3_small_100": "MobileNetV3-Small",
    "convnext_tiny": "ConvNeXt-Tiny",
}


def backbone_display(name: str) -> str:
    return BACKBONE_DISPLAY.get(name, name)
