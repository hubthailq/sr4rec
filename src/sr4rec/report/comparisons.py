"""Before-SR / after-SR comparison panels (one PNG per selected test image).

Layout (one row): LR input -> Bicubic -> every SR model (configuration order)
-> HR (reference, synthetic mode only). Small images are enlarged with
nearest-neighbour interpolation for display only. Under each cell: method
name, predicted label with a correct/wrong mark (solid green / dashed orange
border) and, in synthetic mode, PSNR/SSIM. All text is English.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

from ..utils import to_rgb
from . import style
from .style import plt

SELECTION_SEED = 0
CELL_INCHES = 1.9


@dataclass
class Cell:
    title: str
    file: Path
    pred: str | None = None  # predicted label, None for the LR cell
    correct: bool | None = None
    psnr: float | None = None
    ssim: float | None = None
    crop_scale: int = 0


def select(paths: list[str], bicubic_correct: np.ndarray, sr_correct: list[np.ndarray], count: int,
           selection: str) -> list[tuple[str, str]]:
    """Deterministic selection of test images. Returns (path, reason) with reason in
    corrected | degraded | random, in that order."""
    n = len(paths)
    rng = np.random.default_rng(SELECTION_SEED)
    if selection == "all":
        return [(p, "random") for p in paths]
    count = min(count, n)
    if selection == "random":
        idx = np.sort(rng.choice(n, size=count, replace=False))
        return [(paths[i], "random") for i in idx]
    any_right = np.zeros(n, bool)
    any_wrong = np.zeros(n, bool)
    for c in sr_correct:
        any_right |= c.astype(bool)
        any_wrong |= ~c.astype(bool)
    bic = bicubic_correct.astype(bool)
    corrected = np.flatnonzero(~bic & any_right)
    degraded = np.flatnonzero(bic & any_wrong)
    third = count // 3
    chosen: list[tuple[int, str]] = []
    used: set[int] = set()
    for pool, k, reason in ((corrected, third, "corrected"), (degraded, third, "degraded")):
        pool = np.array([i for i in pool if i not in used], dtype=int)
        take = min(k, len(pool))
        for i in np.sort(rng.choice(pool, size=take, replace=False)) if take else []:
            chosen.append((int(i), reason))
            used.add(int(i))
    rest = np.array([i for i in range(n) if i not in used], dtype=int)
    k = min(count - len(chosen), len(rest))
    for i in np.sort(rng.choice(rest, size=k, replace=False)) if k else []:
        chosen.append((int(i), "random"))
    return [(paths[i], r) for i, r in chosen]


def _load(cell: Cell) -> np.ndarray:
    with Image.open(cell.file) as im:
        im = to_rgb(im)
        if cell.crop_scale:
            w, h = im.size
            im = im.crop((0, 0, w - w % cell.crop_scale, h - h % cell.crop_scale))
        return np.asarray(im)


def caption_lines(cell: Cell, synthetic: bool) -> list[str]:
    lines = [cell.title]
    if cell.pred is not None:
        mark = "✓ correct" if cell.correct else "✗ wrong"
        lines.append(f"pred: {cell.pred}")
        lines.append(mark)
    if synthetic and cell.psnr is not None:
        lines.append(f"PSNR {cell.psnr:.1f} dB | SSIM {cell.ssim:.3f}")
    return lines


def render(title: str, cells: list[Cell], synthetic: bool, out: Path) -> list[str]:
    """Draw one panel; returns every text string written on it (used by tests)."""
    style.apply()
    n = len(cells)
    fig, axes = plt.subplots(1, n, figsize=(CELL_INCHES * n, CELL_INCHES + 1.2), squeeze=False)
    texts = [title]
    images = [_load(c) for c in cells]
    side = max(max(im.shape[:2]) for im in images)
    for ax, cell, im in zip(axes[0], cells, images, strict=False):
        h, w = im.shape[:2]
        f = side / max(h, w)
        disp = np.asarray(Image.fromarray(im).resize((max(1, round(w * f)), max(1, round(h * f))), Image.NEAREST))
        canvas = np.full((side, side, 3), 255, np.uint8)
        top, left = (side - disp.shape[0]) // 2, (side - disp.shape[1]) // 2
        canvas[top : top + disp.shape[0], left : left + disp.shape[1]] = disp
        ax.imshow(canvas, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        if cell.correct is not None:
            color = style.CORRECT_COLOR if cell.correct else style.WRONG_COLOR
            ls = "-" if cell.correct else "--"
            ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False, edgecolor=color, linewidth=3,
                                   linestyle=ls, clip_on=False))
        lines = caption_lines(cell, synthetic)
        texts += lines
        ax.set_xlabel("\n".join(lines), fontsize=9.5)
    fig.suptitle(title, fontsize=10)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return texts


def file_name(index: int, path: str) -> str:
    stem = str(Path(path).with_suffix("")).replace("\\", "/").replace("/", "_")
    return f"{index:02d}_{stem}.png"
