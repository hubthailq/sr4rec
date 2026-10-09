"""MATLAB-style bicubic resize and letterbox."""

from pathlib import Path

import numpy as np
import pytest
import torch

from sr4rec.lr.letterbox import letterbox, letterbox_geometry
from sr4rec.lr.resize import imresize

GOLD = np.load(Path(__file__).parent / "data" / "resize_golden.npz")


@pytest.mark.parametrize("key,scale,src", [("down4", 0.25, "hr"), ("down2", 0.5, "hr"), ("down3", 1 / 3, "hr"),
                                           ("up4", 4, "lr")])
def test_matches_reference(key, scale, src):
    out = imresize(torch.from_numpy(GOLD[src]), scale).numpy()
    ref = GOLD[key]
    assert out.shape == ref.shape
    assert np.abs(np.round(out.clip(0, 1) * 255) - np.round(ref.clip(0, 1) * 255)).max() <= 1
    assert np.abs(out - ref).max() < 1e-5


def test_constant_image_stays_constant():
    x = torch.full((3, 13, 7), 0.4)
    assert torch.allclose(imresize(x, 0.25), torch.full((3, 4, 2), 0.4), atol=1e-6)
    assert torch.allclose(imresize(x, 4), torch.full((3, 52, 28), 0.4), atol=1e-6)


@pytest.mark.parametrize("h,w", [(1, 1), (1, 300), (224, 224), (500, 100), (37, 81)])
def test_letterbox_sizes_and_aspect(h, w):
    size = 224
    out = letterbox(torch.rand(3, h, w), size)
    assert out.shape == (3, size, size)
    nh, nw, top, left = letterbox_geometry(h, w, size)
    assert max(nh, nw) == size
    assert abs(nh / nw - h / w) <= max(1 / nw, 1 / nh) * (h / w + 1) + 1e-9
    if top:  # padding is the ImageNet mean colour, i.e. exactly 0 after ImageNet normalization
        from sr4rec.lr.letterbox import normalize

        assert normalize(out)[:, :top].abs().max() < 1e-6


def test_letterbox_identity_for_exact_size():
    x = torch.rand(3, 64, 64)
    assert torch.equal(letterbox(x, 64), x)


def test_training_augmentation_never_changes_the_aspect_ratio():
    from torchvision.transforms import v2

    from sr4rec.rec.data import RecognitionDataset

    ds = RecognitionDataset([], 224, train=True)
    crops = [t for t in ds.augment.transforms if isinstance(t, v2.RandomResizedCrop)]
    assert crops and all(c.ratio == (1.0, 1.0) for c in crops)


def test_recognizer_input_is_imagenet_normalized_and_not_stretched():
    from sr4rec.lr.letterbox import IMAGENET_MEAN, IMAGENET_STD, recognizer_input

    img = torch.ones(3, 50, 100)  # white, 2:1
    x = recognizer_input(img, 64)
    assert x.shape == (3, 64, 64)
    white = [(1 - m) / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD, strict=True)]
    assert torch.allclose(x[:, 32, 10], torch.tensor(white), atol=1e-5)  # inside the image
    assert x[:, :15].abs().max() < 1e-6 and x[:, -15:].abs().max() < 1e-6  # 32 rows of image, padding above/below
