"""Number formatting rules of report.md (docs/report_guide.md)."""

from __future__ import annotations

import math

MINUS = "−"


def _finite(x) -> bool:
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def fmt_pct(x: float | None, digits: int = 1) -> str:
    """Fraction in [0, 1] -> percentage text."""
    if not _finite(x):
        return "n/a"
    return f"{100 * x:.{digits}f}"


def fmt_mean_sd(values, digits: int = 1) -> str:
    vals = [v for v in values if _finite(v)]
    if not vals:
        return "n/a"
    mean = sum(vals) / len(vals)
    if len(vals) == 1:
        return f"{100 * mean:.{digits}f}"
    sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / (len(vals) - 1))
    return f"{100 * mean:.{digits}f} ± {100 * sd:.{digits}f}"


def fmt_delta(x: float | None, digits: int = 1) -> str:
    """Signed value in percentage points; the minus sign is U+2212."""
    if not _finite(x):
        return "n/a"
    r = round(x, digits)
    if r == 0:
        return f"{0:.{digits}f}"
    return f"+{r:.{digits}f}" if r > 0 else f"{MINUS}{abs(r):.{digits}f}"


def fmt_ci(lo: float, hi: float, digits: int = 1) -> str:
    return f"[{fmt_delta(lo, digits)}, {fmt_delta(hi, digits)}]"


def fmt_p(p: float | None) -> str:
    """Two significant digits; values below 0.001 are written <0.001."""
    if not _finite(p):
        return "n/a"
    if p < 0.001:
        return "<0.001"
    if p >= 0.1:
        return f"{p:.2f}"
    digits = 1 - int(math.floor(math.log10(p)))  # two significant digits
    if round(p, digits) >= 0.1:
        return f"{p:.2f}"
    return f"{p:.{digits}f}"


def fmt_int(n: int) -> str:
    return f"{n:,}"


def fmt_num(x: float | None, digits: int) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    if isinstance(x, float) and math.isinf(x):
        return "inf"
    return f"{x:.{digits}f}"


def fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f} s"
    minutes = seconds / 60
    if minutes < 90:
        return f"{minutes:.0f} min"
    return f"{minutes / 60:.1f} h"
