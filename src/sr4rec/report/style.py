"""Shared figure style: DejaVu Sans, colours that stay distinguishable in grayscale, plus markers/hatches."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLORS = ["#4d4d4d", "#0072b2", "#e69f00", "#009e73", "#cc79a7", "#56b4e9", "#d55e00", "#f0e442", "#000000", "#999999"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]
LINESTYLES = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1)), (0, (1, 1)), "-", "--", "-."]
HATCHES = ["", "//", "\\\\", "xx", "..", "++", "--", "oo", "**", "||"]
CORRECT_COLOR = "#009e73"
WRONG_COLOR = "#d55e00"


def apply() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    })


def style_for(i: int) -> dict:
    return {"color": COLORS[i % len(COLORS)], "marker": MARKERS[i % len(MARKERS)],
            "linestyle": LINESTYLES[i % len(LINESTYLES)], "hatch": HATCHES[i % len(HATCHES)]}
