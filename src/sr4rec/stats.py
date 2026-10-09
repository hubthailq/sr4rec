"""Paired comparison of every SR model with bicubic (see docs/methods.md).

* Unit of analysis: the test image. Per image, correctness (Rank-1) is averaged over seeds.
* Delta: mean per-image score of the SR model minus that of bicubic, in percentage points.
* 95% CI: percentile bootstrap over test images, 10,000 resamples, seed 0.
* p: two-sided paired sign-flip permutation test on per-image differences, 10,000
  permutations, seed 0, p = (1 + #{|T*| >= |T|}) / (1 + 10,000).
* Holm correction over the SR-vs-bicubic comparisons of one backbone (and one size bin).
"""

from __future__ import annotations

import numpy as np
from statsmodels.stats.multitest import multipletests

N_RESAMPLES = 10_000
ALPHA = 0.05
STATS_SEED = 0
CHUNK = 500

GAIN = "▲ significant gain"
HARM = "▼ significant harm"
NO_DIFF = "● no difference"
INSUFFICIENT = "insufficient data"


def bootstrap_ci(diff: np.ndarray, n: int = N_RESAMPLES, seed: int = STATS_SEED, level: float = 0.95) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    m = len(diff)
    means = np.empty(n)
    for start in range(0, n, CHUNK):
        k = min(CHUNK, n - start)
        idx = rng.integers(0, m, size=(k, m))
        means[start : start + k] = diff[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * (1 - level) / 2, 100 * (1 + level) / 2])
    return float(lo), float(hi)


def sign_flip_p(diff: np.ndarray, n: int = N_RESAMPLES, seed: int = STATS_SEED) -> float:
    obs = abs(diff.mean())
    nz = diff[diff != 0]
    if len(nz) == 0:
        return 1.0
    rng = np.random.default_rng(seed)
    m = len(diff)
    count = 0
    for start in range(0, n, CHUNK):
        k = min(CHUNK, n - start)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(k, len(nz)))
        stat = np.abs((signs * nz).sum(axis=1) / m)
        count += int((stat >= obs - 1e-12).sum())
    return (1 + count) / (1 + n)


def same_sign(sr: np.ndarray, bic: np.ndarray, mean_delta: float) -> tuple[int, int]:
    """Number of seeds whose delta has the same sign as ``mean_delta`` (zero counts as the same sign
    only when ``mean_delta`` is zero)."""
    per_seed = sr.mean(axis=1) - bic.mean(axis=1)
    target = np.sign(round(mean_delta, 12))
    return int((np.sign(np.round(per_seed, 12)) == target).sum()), len(per_seed)


def compare(sr_correct: np.ndarray, bic_correct: np.ndarray) -> dict:
    """``*_correct``: arrays (seeds, images) of 0/1 Rank-1 correctness on the same test images."""
    sr_score, bic_score = sr_correct.mean(axis=0), bic_correct.mean(axis=0)
    diff = sr_score - bic_score
    delta = float(diff.mean())
    lo, hi = bootstrap_ci(diff)
    p = sign_flip_p(diff)
    k, n = same_sign(sr_correct, bic_correct, delta)
    return {"delta": 100 * delta, "ci_low": 100 * lo, "ci_high": 100 * hi, "p_raw": p, "same_sign": k, "n_seeds": n,
            "n_images": len(diff)}


def holm(pvalues: list[float]) -> list[float]:
    if not pvalues:
        return []
    return [float(x) for x in multipletests(pvalues, alpha=ALPHA, method="holm")[1]]


def verdict(delta: float, ci_low: float, ci_high: float, p_holm: float) -> str:
    excludes_zero = ci_low > 0 or ci_high < 0
    if p_holm < ALPHA and excludes_zero:
        if delta > 0:
            return GAIN
        if delta < 0:
            return HARM
    return NO_DIFF
