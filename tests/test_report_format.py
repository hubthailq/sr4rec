"""Report formatting rules, answer sentences and comparison-panel selection."""

import re
import unicodedata

import numpy as np
import pandas as pd
from PIL import Image

from sr4rec import stats
from sr4rec.report import comparisons as comp
from sr4rec.report.formatting import fmt_ci, fmt_delta, fmt_duration, fmt_int, fmt_mean_sd, fmt_p
from sr4rec.report.render import answer_text

VIETNAMESE = re.compile("[\u0300\u0301\u0302\u0303\u0306\u0309\u031b\u0323\u0110\u0111]")


def test_number_formats():
    assert fmt_delta(0.4) == "+0.4" and fmt_delta(-3.21) == "−3.2" and fmt_delta(0.04) == "0.0"
    assert fmt_p(0.0004) == "<0.001" and fmt_p(0.62) == "0.62" and fmt_p(0.0021) == "0.0021"
    assert fmt_ci(-4.6, -1.8) == "[−4.6, −1.8]"
    assert fmt_mean_sd([0.78, 0.79, 0.77]) == "78.0 ± 1.0"
    assert fmt_int(12345) == "12,345" and fmt_duration(3600 * 2) == "2.0 h"


def _stat(rows):
    return pd.DataFrame([{"size_bin": "all", **r} for r in rows])


def test_answer_rules():
    assert "No SR model was configured" in answer_text(_stat([]), [], 3)
    none = _stat([{"backbone": "b", "method": "a", "verdict": stats.NO_DIFF}])
    assert answer_text(none, ["a"], 1) == "No SR model changed Rank-1 accuracy significantly on any backbone."
    mixed = _stat([{"backbone": "b1", "method": "a", "verdict": stats.GAIN},
                   {"backbone": "b2", "method": "a", "verdict": stats.NO_DIFF},
                   {"backbone": "b1", "method": "x", "verdict": stats.HARM},
                   {"backbone": "b2", "method": "x", "verdict": stats.HARM}])
    text = answer_text(mixed, ["a", "x"], 2)
    assert text == ("a (1 of 2 backbones) improved Rank-1 accuracy significantly. "
                    "x (2 of 2 backbones) reduced Rank-1 accuracy significantly.")
    assert text.count(".") <= 3


def test_mixed_selection_is_deterministic_and_balanced():
    n = 60
    rng = np.random.default_rng(3)
    bic = rng.random(n) < 0.5
    srs = [rng.random(n) < 0.5, rng.random(n) < 0.5]
    paths = [f"p{i}.png" for i in range(n)]
    a = comp.select(paths, bic, srs, 12, "mixed")
    b = comp.select(paths, bic, srs, 12, "mixed")
    assert a == b and len(a) == 12
    reasons = [r for _, r in a]
    assert reasons.count("corrected") == 4 and reasons.count("degraded") == 4 and reasons.count("random") == 4
    idx = {p: i for i, p in enumerate(paths)}
    for p, r in a:
        i = idx[p]
        if r == "corrected":
            assert not bic[i] and any(s[i] for s in srs)
        if r == "degraded":
            assert bic[i] and any(not s[i] for s in srs)
    assert len(comp.select(paths, bic, srs, 5, "all")) == n
    assert len({p for p, _ in comp.select(paths, bic, srs, 7, "random")}) == 7


def test_panel_cells_and_english_text(tmp_path):
    files = []
    for i, size in enumerate([(12, 10), (48, 40), (48, 40), (48, 40)]):
        f = tmp_path / f"{i}.png"
        Image.new("RGB", size, (i * 40, 100, 100)).save(f)
        files.append(f)
    cells = [comp.Cell("LR input (12 × 10 px)", files[0]),
             comp.Cell("Bicubic", files[1], pred="cat", correct=False, psnr=25.1, ssim=0.7),
             comp.Cell("my_sr", files[2], pred="dog", correct=True, psnr=27.9, ssim=0.83),
             comp.Cell("HR (reference)", files[3], pred="dog", correct=True)]
    texts = comp.render("a/b.jpg | true: dog | ResNet-18, seed 0 | protocol: matched", cells, True, tmp_path / "o.png")
    assert (tmp_path / "o.png").is_file()
    joined = "\n".join(texts)
    assert "✗ wrong" in joined and "✓ correct" in joined and "PSNR 27.9 dB | SSIM 0.830" in joined
    assert not VIETNAMESE.search(unicodedata.normalize("NFD", joined))
    native = comp.render("t", cells[:3], False, tmp_path / "n.png")
    assert not any("PSNR" in t for t in native)
    assert comp.file_name(1, "Abyssinian/Abyssinian_10.jpg") == "01_Abyssinian_Abyssinian_10.png"
