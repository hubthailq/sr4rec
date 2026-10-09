"""Recognition metrics, image quality and statistics."""

import numpy as np
import pytest
from scipy import stats as sps
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from sr4rec import stats
from sr4rec.eval.metrics import cmc, macro_prf, psnr_y, rank_k, recognition_metrics, ssim_y


def test_rank_metrics_by_hand():
    ranks = np.array([1, 2, 1, 6, 3, 1])
    assert rank_k(ranks, 1, 10) == pytest.approx(3 / 6)
    assert rank_k(ranks, 5, 10) == pytest.approx(5 / 6)
    curve = cmc(ranks, 7)
    assert curve[:3] == pytest.approx([3 / 6, 4 / 6, 5 / 6]) and curve[5] == 1.0
    assert all(np.isnan(v) for v in curve[7:])


def test_rank1_equals_accuracy_and_sklearn_macro():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 6, 200)
    p = np.where(rng.random(200) < 0.6, y, rng.integers(0, 6, 200))
    p[p == 5] = 4  # class 5 is never predicted: precision 0 by convention
    ranks = np.where(p == y, 1, 2)
    m = recognition_metrics(ranks, y, p, 6)
    assert m["rank1"] == accuracy_score(y, p)
    kw = dict(labels=np.arange(6), average="macro", zero_division=0)
    assert (m["precision"], m["recall"], m["f1"]) == (precision_score(y, p, **kw), recall_score(y, p, **kw),
                                                       f1_score(y, p, **kw))
    assert macro_prf(y, p, 6)[0] == precision_score(y, p, **kw)


def test_psnr_ssim_basic():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 256, (40, 36, 3), dtype=np.uint8)
    assert psnr_y(a, a, 4) == float("inf")
    assert ssim_y(a, a, 4) == pytest.approx(1.0)
    b = np.clip(a.astype(int) + 5, 0, 255).astype(np.uint8)
    assert 30 < psnr_y(b, a, 4) < 50
    assert ssim_y(b, a, 4) < 1.0


def test_bootstrap_and_permutation_against_scipy():
    rng = np.random.default_rng(1)
    sr = (rng.random((3, 300)) < 0.62).astype(float)
    bic = (rng.random((3, 300)) < 0.55).astype(float)
    diff = sr.mean(0) - bic.mean(0)
    lo, hi = stats.bootstrap_ci(diff)
    ref = sps.bootstrap((diff,), np.mean, n_resamples=10_000, method="percentile", random_state=0).confidence_interval
    assert abs(lo - ref.low) < 0.01 and abs(hi - ref.high) < 0.01
    p = stats.sign_flip_p(diff)
    ref_p = sps.permutation_test((diff,), np.mean, permutation_type="samples", n_resamples=10_000,
                                 random_state=0).pvalue
    assert abs(p - ref_p) < 0.01


def test_holm_matches_statsmodels():
    from statsmodels.stats.multitest import multipletests

    ps = [0.01, 0.04, 0.03, 0.2]
    assert stats.holm(ps) == pytest.approx(list(multipletests(ps, method="holm")[1]), abs=1e-10)


def test_same_sign_and_verdicts():
    sr = np.array([[1, 1, 0, 1], [1, 0, 0, 1], [0, 0, 0, 1]], float)
    bic = np.array([[0, 1, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]], float)
    c = stats.compare(sr, bic)
    assert c["delta"] == pytest.approx(100 * (2 / 12))
    assert (c["same_sign"], c["n_seeds"]) == (2, 3)
    assert stats.verdict(3.0, 1.0, 5.0, 0.01) == stats.GAIN
    assert stats.verdict(-3.0, -5.0, -1.0, 0.01) == stats.HARM
    assert stats.verdict(3.0, -1.0, 5.0, 0.01) == stats.NO_DIFF
    assert stats.verdict(3.0, 1.0, 5.0, 0.2) == stats.NO_DIFF


def test_identical_methods_give_p_one():
    x = (np.random.default_rng(0).random((3, 100)) < 0.5).astype(float)
    c = stats.compare(x, x)
    assert c["p_raw"] == 1.0 and c["delta"] == 0 and c["same_sign"] == 3


def test_rank_is_consistent_with_argmax_on_ties():
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from sr4rec.rec.engine import rank_of_true

    class Const(torch.nn.Module):
        def forward(self, x):
            return torch.zeros(x.shape[0], 4)

    y = torch.tensor([0, 1, 2, 3])
    ranks, pred = rank_of_true(Const(), DataLoader(TensorDataset(torch.zeros(4, 1), y), batch_size=4),
                               torch.device("cpu"), return_pred=True)
    assert list(ranks) == [1, 2, 3, 4] and list(pred) == [0, 0, 0, 0]
    assert ((ranks == 1) == (pred == y.numpy())).all()
