"""Image datasets for recognizer training and evaluation.

Each image is letterboxed (aspect ratio kept, ImageNet-mean padding), optionally augmented
(training only; square crops, so no stretching) and ImageNet-normalized.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset as TorchDataset
from torchvision.transforms import v2

from ..lr.letterbox import letterbox, normalize
from ..utils import load_rgb, mod_crop, pil_to_tensor


@dataclass(frozen=True)
class Item:
    """One image as the recognizer sees it: the file to read, its class index and
    whether the HR image must be mod-cropped first."""

    path: str  # dataset path (identifier)
    file: Path
    target: int
    crop_scale: int = 0  # > 0: mod-crop by this scale (HR images in synthetic mode)


def load_item(item: Item) -> torch.Tensor:
    im = load_rgb(item.file)
    if item.crop_scale:
        im = mod_crop(im, item.crop_scale)
    return pil_to_tensor(im)


class RecognitionDataset(TorchDataset):
    def __init__(self, items: list[Item], size: int, train: bool, horizontal_flip: bool = False):
        self.items = items
        self.size = size
        aug = [v2.RandomResizedCrop(size, scale=(0.8, 1.0), ratio=(1.0, 1.0), antialias=True),
               v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1)]
        if horizontal_flip:
            aug.append(v2.RandomHorizontalFlip())
        self.augment = v2.Compose(aug) if train else None

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        item = self.items[i]
        x = letterbox(load_item(item), self.size)
        if self.augment is not None:
            x = self.augment(x).clamp(0.0, 1.0)
        return normalize(x), item.target
