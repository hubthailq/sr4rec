"""Device priority (CUDA > MPS > CPU) and out-of-memory fallbacks, simulated on the CPU."""

import numpy as np
import pytest
import torch
from torch import nn

from sr4rec import device as dev
from sr4rec.config import RecognizerConfig, TrainingConfig
from sr4rec.rec import engine
from sr4rec.sr.base import SRSource, run_padded, run_tiled
from sr4rec.sr.runner import RobustRunner

OOM = RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")


@pytest.mark.parametrize("have,pref,expected,warn", [
    ({"cuda", "mps", "cpu"}, "auto", "cuda", False),
    ({"mps", "cpu"}, "auto", "mps", False),
    ({"cpu"}, "auto", "cpu", False),
    ({"mps", "cpu"}, "cuda", "mps", True),
    ({"cpu"}, "cuda", "cpu", True),
    ({"cpu"}, "mps", "cpu", True),
    ({"cuda", "cpu"}, "cpu", "cpu", False),
])
def test_device_priority_and_fallback(monkeypatch, have, pref, expected, warn):
    monkeypatch.setattr(dev, "available", lambda k: k in have)
    d, warnings = dev.resolve(pref)
    assert d.type == expected and bool(warnings) == warn


def test_oom_detection():
    assert dev.is_oom(OOM)
    assert dev.is_oom(RuntimeError("MPS backend out of memory (MPS allocated: 8 GB)"))
    assert dev.is_oom(RuntimeError("[enforce fail at alloc_cpu.cpp] DefaultCPUAllocator: can't allocate memory"))
    assert not dev.is_oom(RuntimeError("shape mismatch"))


class BigInputOOM(SRSource):
    """Nearest-neighbour x4 'SR' that runs out of memory for inputs larger than 80 px."""

    def __init__(self):
        super().__init__(name="big", kind="module", scale=4)
        self.tile = None
        self.calls = 0
        from sr4rec.sr.base import SizeRule

        self.rule = SizeRule()

    def to(self, device):
        return self

    def _fn(self, x):
        self.calls += 1
        if max(x.shape[-2:]) > 80:
            raise OOM
        return nn.functional.interpolate(x, scale_factor=4, mode="nearest")

    def upscale(self, lr):
        if self.tile:
            return run_tiled(self._fn, lr, 4, self.tile, self.rule)
        return run_padded(self._fn, lr, 4, self.rule)


def test_sr_switches_to_automatic_tiling():
    src = BigInputOOM()
    runner = RobustRunner(src, torch.device("cpu"))
    lr = torch.rand(1, 3, 150, 90)
    out = runner.upscale(lr)
    assert out.shape == (1, 3, 600, 360)
    assert torch.allclose(out, nn.functional.interpolate(lr, scale_factor=4, mode="nearest"))
    assert runner.stats.tile == 75
    assert runner.stats.tiled == 1
    assert "automatic tiling" in runner.stats.warning("big", torch.device("cuda"))
    assert src.tile is None  # the configured value is restored


def test_training_restarts_with_gradient_accumulation(monkeypatch):
    calls = []
    real = engine._train_once

    def fake(backbone, num_classes, train_items, val_items, rec, tr, seed, device, accumulation, log):
        calls.append(accumulation)
        if accumulation < 4:
            raise OOM
        return engine.TrainResult({}, 1, 0.5, 0.0, accumulation, device.type)

    monkeypatch.setattr(engine, "_train_once", fake)
    res = engine.train_recognizer("resnet18", 3, [], [], RecognizerConfig(pretrained=False),
                                  TrainingConfig(batch_size=16), 0, torch.device("cpu"))
    assert calls == [1, 2, 4] and res.accumulation == 4 and len(res.notes) == 2
    assert "effective batch 16" in res.notes[0]
    monkeypatch.setattr(engine, "_train_once", real)


def test_accumulated_training_runs_for_real(tmp_path):
    from sr4rec.rec.data import Item
    from tests.fixtures.synthetic import make_dataset

    root = make_dataset(tmp_path / "d", n_classes=2, per_class=6, size=(40, 40))
    items = [Item(f"c{c}/img_{i:02d}.png", root / "images" / f"c{c}/img_{i:02d}.png", c) for c in range(2)
             for i in range(6)]
    rec = RecognizerConfig(backbones=["resnet18"], pretrained=False, input_size=64, seeds=[0])
    tr = TrainingConfig(epochs=1, batch_size=8, num_workers=0)
    res = engine._train_once("resnet18", 2, items, items, rec, tr, 0, torch.device("cpu"), 4, None)
    assert res.accumulation == 4 and res.state and 0.0 <= res.best_val_rank1 <= 1.0


def test_prediction_halves_the_batch(monkeypatch, tmp_path):
    from sr4rec.rec.data import Item
    from sr4rec.rec.models import create_backbone
    from tests.fixtures.synthetic import make_dataset

    root = make_dataset(tmp_path / "d", n_classes=2, per_class=4, size=(40, 40))
    items = [Item(f"c{c}/img_{i:02d}.png", root / "images" / f"c{c}/img_{i:02d}.png", c) for c in range(2)
             for i in range(4)]
    state = create_backbone("resnet18", 2, pretrained=False).state_dict()
    rec = RecognizerConfig(backbones=["resnet18"], pretrained=False, input_size=64, seeds=[0])
    ref = engine.predict("resnet18", 2, state, items, rec, 8, 0, torch.device("cpu"))
    real = engine.rank_of_true

    def picky(model, loader, device, return_pred=False):
        if loader.batch_size > 2:
            raise OOM
        return real(model, loader, device, return_pred)

    monkeypatch.setattr(engine, "rank_of_true", picky)
    notes = []
    out = engine.predict("resnet18", 2, state, items, rec, 8, 0, torch.device("cpu"), notes)
    assert np.array_equal(out[0], ref[0]) and np.array_equal(out[1], ref[1])
    assert notes and "reduced to 2" in notes[-1]


def test_non_oom_errors_are_not_swallowed(monkeypatch):
    def broken(*a, **k):
        raise RuntimeError("shape mismatch")

    monkeypatch.setattr(engine, "_train_once", broken)
    with pytest.raises(RuntimeError, match="shape mismatch"):
        engine.train_recognizer("resnet18", 3, [], [], RecognizerConfig(pretrained=False), TrainingConfig(), 0,
                                torch.device("cpu"))
