"""The three report figures (English text only)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..data.sizes import SIZE_BIN_LABELS, SIZE_BINS
from ..utils import backbone_display
from . import style
from .style import plt


def label(m: str) -> str:
    return {"hr": "HR (upper bound)", "bicubic": "bicubic (baseline)"}.get(m, m)


def _grid(n: int):
    cols = min(n, 3)
    rows = -(-n // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.3 * rows), squeeze=False)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    return fig, list(axes.flat[:n])


def accuracy_by_size(metrics: pd.DataFrame, methods: list[str], backbones: list[str], out: Path) -> None:
    style.apply()
    fig, axes = _grid(len(backbones))
    width = 0.8 / max(1, len(methods))
    for ax, b in zip(axes, backbones, strict=False):
        sub = metrics[(metrics["backbone"] == b) & (metrics["size_bin"] != "all")]
        for i, m in enumerate(methods):
            vals = []
            for sb in SIZE_BINS:
                cell = sub[(sub["method"] == m) & (sub["size_bin"] == sb)]
                vals.append(100 * cell["rank1"].mean() if len(cell) else np.nan)
            st = style.style_for(i)
            ax.bar(np.arange(len(SIZE_BINS)) + (i - (len(methods) - 1) / 2) * width, vals, width,
                   color=st["color"], hatch=st["hatch"], edgecolor="white", linewidth=0.5, label=label(m))
        ax.set_xticks(np.arange(len(SIZE_BINS)), [SIZE_BIN_LABELS[s] for s in SIZE_BINS])
        ax.set_title(backbone_display(b))
        ax.set_xlabel("Short side of the LR image")
        ax.set_ylabel("Rank-1 accuracy (%)")
        ax.set_ylim(0, 100)
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle(y=1.02, t="Rank-1 accuracy by LR image size (mean over seeds)")
    fig.savefig(out)
    plt.close(fig)


def cmc_curves(cmc: pd.DataFrame, methods: list[str], backbones: list[str], out: Path) -> None:
    style.apply()
    fig, axes = _grid(len(backbones))
    ks = [c for c in cmc.columns if c.startswith("rank")]
    x = [int(c[4:]) for c in ks]
    for ax, b in zip(axes, backbones, strict=False):
        for i, m in enumerate(methods):
            sub = cmc[(cmc["backbone"] == b) & (cmc["method"] == m)]
            if not len(sub):
                continue
            y = 100 * sub[ks].mean(axis=0).to_numpy(dtype=float)
            st = style.style_for(i)
            ax.plot(x, y, color=st["color"], marker=st["marker"], linestyle=st["linestyle"], markersize=4, label=label(m))
        ax.set_title(backbone_display(b))
        ax.set_xlabel("Rank k")
        ax.set_ylabel("Rank-k accuracy (%)")
        ax.set_xticks(x)
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle(y=1.02, t="CMC curves (mean over seeds)")
    fig.savefig(out)
    plt.close(fig)


def quality_vs_accuracy(quality: pd.DataFrame, overall: pd.DataFrame, methods: list[str], backbones: list[str],
                        out: Path) -> None:
    """``quality``: method, psnr, ssim. ``overall``: backbone, method, rank1 (mean over seeds)."""
    style.apply()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, col, label in zip(axes, ("psnr", "ssim"), ("PSNR (dB, Y channel)", "SSIM (Y channel)"), strict=False):
        for j, b in enumerate(backbones):
            xs, ys, names = [], [], []
            for m in methods:
                q = quality.loc[quality["method"] == m, col]
                a = overall[(overall["backbone"] == b) & (overall["method"] == m)]["rank1"]
                if len(q) and len(a) and np.isfinite(q.iloc[0]):
                    xs.append(q.iloc[0])
                    ys.append(100 * a.iloc[0])
                    names.append(m)
            st = style.style_for(j + 1)
            ax.plot(xs, ys, linestyle="none", marker=st["marker"], color=st["color"], label=backbone_display(b))
            if j == 0:
                for xv, yv, m in zip(xs, ys, names, strict=False):
                    ax.annotate(m, (xv, yv), textcoords="offset points", xytext=(4, 3), fontsize=7)
        ax.set_xlabel(label)
        ax.set_ylabel("Rank-1 accuracy (%)")
        ax.margins(x=0.2, y=0.1)
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle(y=1.02, t="Image quality vs recognition accuracy (test set)")
    fig.savefig(out)
    plt.close(fig)
