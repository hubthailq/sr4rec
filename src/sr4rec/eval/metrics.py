"""Recognition metrics (Rank-1, Rank-5, CMC, macro precision/recall/F1) and
image-quality metrics (PSNR, SSIM on the Y channel, MATLAB conventions)."""

from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import correlate1d
from sklearn.metrics import precision_recall_fscore_support

CMC_MAX_RANK = 10


def rank_k(true_rank: np.ndarray, k: int, num_classes: int) -> float:
    """Fraction of images whose true class is among the top ``k``; NaN when k > number of classes."""
    if k > num_classes:
        return float("nan")
    return float((np.asarray(true_rank) <= k).mean()) if len(true_rank) else float("nan")


def cmc(true_rank: np.ndarray, num_classes: int, max_rank: int = CMC_MAX_RANK) -> list[float]:
    return [rank_k(true_rank, k, num_classes) for k in range(1, max_rank + 1)]


def macro_prf(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> tuple[float, float, float]:
    """Macro precision, recall and F1 over all dataset classes (zero_division=0)."""
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=np.arange(num_classes), average="macro",
                                                 zero_division=0)
    return float(p), float(r), float(f)


def recognition_metrics(true_rank: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, num_classes: int,
                        labels: np.ndarray | None = None) -> dict:
    """``labels``: classes to average over for macro metrics (default: all dataset classes; the
    size-bin rows use the classes present in the bin)."""
    if labels is None:
        p, r, f = macro_prf(y_true, y_pred, num_classes)
    else:
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        p, r, f = float(p), float(r), float(f)
    return {
        "rank1": rank_k(true_rank, 1, num_classes),
        "rank5": rank_k(true_rank, 5, num_classes),
        "precision": p,
        "recall": r,
        "f1": f,
    }


# ----------------------------------------------------------------------------- image quality


def rgb_to_y(img: np.ndarray) -> np.ndarray:
    """uint8 RGB (H, W, 3) -> Y channel in [16, 235] as float64 (MATLAB ``rgb2ycbcr``)."""
    x = img.astype(np.float64) / 255.0
    return 16.0 + 65.481 * x[..., 0] + 128.553 * x[..., 1] + 24.966 * x[..., 2]


def _shave(y: np.ndarray, border: int) -> np.ndarray:
    if border <= 0:
        return y
    return y[border:-border, border:-border]


def psnr_y(sr: np.ndarray, hr: np.ndarray, border: int) -> float:
    a, b = _shave(rgb_to_y(sr), border), _shave(rgb_to_y(hr), border)
    if a.size == 0:
        return float("nan")
    mse = float(np.mean((a - b) ** 2))
    return float("inf") if mse == 0 else 10.0 * math.log10(255.0**2 / mse)


def _gaussian(size: int = 11, sigma: float = 1.5) -> np.ndarray:
    ax = np.arange(size) - (size - 1) / 2.0
    g = np.exp(-(ax**2) / (2 * sigma**2))
    return g / g.sum()


def _filter_valid(x: np.ndarray, g: np.ndarray) -> np.ndarray:
    y = correlate1d(correlate1d(x, g, axis=0, mode="constant"), g, axis=1, mode="constant")
    r = len(g) // 2
    return y[r:-r, r:-r]


def ssim_y(sr: np.ndarray, hr: np.ndarray, border: int) -> float:
    """SSIM of Wang et al. (2004) on the Y channel: Gaussian window 11, sigma 1.5, 'valid' filtering,
    K1 = 0.01, K2 = 0.03, L = 255 (as ``ssim_index.m``)."""
    a, b = _shave(rgb_to_y(sr), border), _shave(rgb_to_y(hr), border)
    if min(a.shape) < 11:
        return float("nan")
    g = _gaussian()
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu1, mu2 = _filter_valid(a, g), _filter_valid(b, g)
    s11 = _filter_valid(a * a, g) - mu1**2
    s22 = _filter_valid(b * b, g) - mu2**2
    s12 = _filter_valid(a * b, g) - mu1 * mu2
    num = (2 * mu1 * mu2 + c1) * (2 * s12 + c2)
    den = (mu1**2 + mu2**2 + c1) * (s11 + s22 + c2)
    return float(np.mean(num / den))
