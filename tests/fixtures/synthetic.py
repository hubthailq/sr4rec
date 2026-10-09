"""Tiny generated datasets for the tests only (never used in examples or the README).

Each class has its own colour and stripe orientation, so that even a small
recognizer trained for a few epochs on the CPU separates the classes.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image


def make_image(cls: int, idx: int, size: tuple[int, int], rng: np.random.Generator) -> Image.Image:
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w]
    base = np.array([(cls * 67) % 256, (cls * 131 + 60) % 256, (cls * 29 + 120) % 256], dtype=np.float64)
    angle = cls * np.pi / 5
    stripes = 0.5 + 0.5 * np.sin((xx * np.cos(angle) + yy * np.sin(angle)) / (2 + cls % 3))
    img = base[None, None, :] * (0.55 + 0.45 * stripes[..., None])
    img += rng.normal(0, 12, img.shape)
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


def make_dataset(root: Path, n_classes: int = 4, per_class: int = 12, size=(64, 48), with_split: bool = False,
                 sizes: list[tuple[int, int]] | None = None, seed: int = 0, ext: str = ".png") -> Path:
    """Write ``root/images/c<k>/img_<i><ext>`` and ``root/labels.csv``."""
    rng = np.random.default_rng(seed)
    (root / "images").mkdir(parents=True, exist_ok=True)
    rows = []
    for c in range(n_classes):
        for i in range(per_class):
            sz = sizes[(c * per_class + i) % len(sizes)] if sizes else size
            rel = f"c{c}/img_{i:02d}{ext}"
            f = root / "images" / rel
            f.parent.mkdir(parents=True, exist_ok=True)
            make_image(c, i, sz, rng).save(f)
            row = {"path": rel, "label": f"class_{c}"}
            if with_split:
                row["split"] = "test" if i % 4 == 0 else ("val" if i % 4 == 1 else "train")
            rows.append(row)
    with open(root / "labels.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    return root
