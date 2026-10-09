"""Fixed training recipe and evaluation of the recognizer.

Recipe (not configurable except through the `training` block): AdamW, cosine
learning-rate schedule with one warm-up epoch, cross-entropy loss, mixed
precision on CUDA (float32 on Apple GPUs and the CPU), light augmentation (RandomResizedCrop scale 0.8-1.0, small
ColorJitter, optional horizontal flip). The checkpoint with the best val
Rank-1 accuracy is kept (earliest epoch on ties). The test split is never used.
"""

from __future__ import annotations

import math
import os
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from ..config import RecognizerConfig, TrainingConfig
from ..device import CPU, free_memory, is_oom, synchronize
from .data import Item, RecognitionDataset
from .models import create_backbone

MAX_RANK = 10


def set_determinism(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _worker_init(worker_id: int) -> None:
    s = torch.initial_seed() % (2**32)
    random.seed(s)
    np.random.seed(s)


def _loader(ds, batch_size, shuffle, workers, seed, device) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=workers, generator=g,
                      worker_init_fn=_worker_init, pin_memory=device.type == "cuda",
                      persistent_workers=False, drop_last=False)


@dataclass
class TrainResult:
    state: dict
    best_epoch: int
    best_val_rank1: float
    seconds: float
    accumulation: int = 1  # micro-batches per optimizer step (1 = no out-of-memory fallback)
    device: str = ""  # device actually used for training
    notes: list[str] = field(default_factory=list)


def _autocast(device: torch.device):
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.float16)
    return torch.autocast(device.type if device.type in ("cpu", "mps") else "cpu", enabled=False)


def _train_once(backbone, num_classes, train_items, val_items, rec, tr, seed, device, accumulation, log):
    """One complete training run. ``accumulation`` micro-batches of ``batch_size / accumulation``
    images form one optimizer step, so the effective batch size, the sample order and the
    learning-rate schedule do not change."""
    set_determinism(seed)
    model = create_backbone(backbone, num_classes, rec.pretrained).to(device)
    micro = max(1, math.ceil(tr.batch_size / accumulation))
    train_ds = RecognitionDataset(train_items, rec.input_size, train=True, horizontal_flip=tr.horizontal_flip)
    val_ds = RecognitionDataset(val_items, rec.input_size, train=False)
    train_dl = _loader(train_ds, micro, True, tr.num_workers, seed, device)
    val_dl = _loader(val_ds, micro, False, tr.num_workers, seed, device)

    opt = torch.optim.AdamW(model.parameters(), lr=tr.learning_rate, weight_decay=tr.weight_decay)
    steps_per_epoch = max(1, math.ceil(len(train_ds) / tr.batch_size))
    total = tr.epochs * steps_per_epoch
    warm = steps_per_epoch if tr.epochs > 1 else 0

    def lr_lambda(step: int) -> float:
        if step < warm:
            return (step + 1) / warm
        progress = (step - warm) / max(1, total - warm)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    best_state, best_acc, best_epoch = None, -1.0, 0
    start = time.perf_counter()
    for epoch in range(1, tr.epochs + 1):
        model.train()
        group: list = []
        batches = iter(train_dl)
        done = False
        while not done:
            try:
                group.append(next(batches))
            except StopIteration:
                done = True
            if group and (len(group) == accumulation or done):
                n = sum(len(y) for _, y in group)
                opt.zero_grad(set_to_none=True)
                for x, y in group:
                    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                    with _autocast(device):
                        loss = loss_fn(model(x), y) / n  # mean over the full batch
                    scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                sched.step()
                group = []
        ranks = rank_of_true(model, val_dl, device)
        acc = float((ranks == 1).mean())
        if acc > best_acc:
            best_acc, best_epoch = acc, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if log:
            log(f"      epoch {epoch}/{tr.epochs}  val Rank-1 {100 * acc:.1f}%")
    return TrainResult(best_state, best_epoch, best_acc, time.perf_counter() - start, accumulation, device.type)


def train_recognizer(backbone: str, num_classes: int, train_items: list[Item], val_items: list[Item],
                     rec: RecognizerConfig, tr: TrainingConfig, seed: int, device: torch.device,
                     log: Callable[[str], None] | None = None) -> TrainResult:
    """Train with the fixed recipe. Out of memory never stops the run: training restarts from the
    beginning with twice the gradient accumulation (same effective batch size) until the
    micro-batch is one image, then restarts on the CPU."""
    accumulation = 1
    notes: list[str] = []
    while True:
        try:
            res = _train_once(backbone, num_classes, train_items, val_items, rec, tr, seed, device, accumulation, log)
            res.notes = notes
            return res
        except RuntimeError as err:
            if not is_oom(err):
                raise
            free_memory(device)
            micro = math.ceil(tr.batch_size / accumulation)
            if micro > 1:
                accumulation *= 2
                msg = (f"{device.type.upper()} memory ran out; restarting with {accumulation} micro-batches of "
                       f"{math.ceil(tr.batch_size / accumulation)} images per step (effective batch {tr.batch_size})")
            elif device.type != "cpu":
                device, accumulation = CPU, 1
                msg = "memory ran out even with micro-batches of 1 image; restarting on the CPU"
            else:
                raise
            notes.append(msg)
            if log:
                log(f"      {msg}")


@torch.no_grad()
def rank_of_true(model: nn.Module, loader: DataLoader, device: torch.device, return_pred: bool = False):
    """Rank (1 = top prediction) of the true class for every image, in loader order."""
    model.eval()
    ranks, preds = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x).float()  # evaluation in float32 (no autocast)
        true_score = logits.gather(1, y[:, None])
        # Rank of the true class in the order used by argmax: higher score first, and on equal
        # scores the lower class index first. Hence rank == 1 exactly when argmax == true class.
        idx = torch.arange(logits.shape[1], device=logits.device)[None, :]
        ahead = (logits > true_score) | ((logits == true_score) & (idx < y[:, None]))
        ranks.append((1 + ahead.sum(1)).cpu().numpy())
        preds.append(logits.argmax(1).cpu().numpy())
    r = np.concatenate(ranks) if ranks else np.zeros(0, dtype=int)
    if return_pred:
        return r, (np.concatenate(preds) if preds else np.zeros(0, dtype=int))
    return r


def predict(backbone: str, num_classes: int, state: dict, items: list[Item], rec: RecognizerConfig,
            batch_size: int, workers: int, device: torch.device,
            notes: list[str] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return (true-class rank, predicted class index) for ``items``. On out of memory the batch is
    halved (results do not depend on the batch size in eval mode), then the CPU is used."""
    model = create_backbone(backbone, num_classes, pretrained=False)
    model.load_state_dict(state)
    ds = RecognitionDataset(items, rec.input_size, train=False)
    bs = batch_size
    while True:
        try:
            model.to(device)
            return rank_of_true(model, _loader(ds, bs, False, workers, 0, device), device, return_pred=True)
        except RuntimeError as err:
            if not is_oom(err):
                raise
            free_memory(device)
            if bs > 1:
                bs = max(1, bs // 2)
                msg = f"evaluation batch reduced to {bs} ({device.type.upper()} memory ran out)"
            elif device.type != "cpu":
                device = CPU
                msg = "evaluation moved to the CPU (memory ran out)"
            else:
                raise
            if notes is not None:
                notes.append(msg)
        finally:
            model.to(CPU)


def _timed_step(model, opt, loss_fn, x, y, device) -> None:
    opt.zero_grad()
    with _autocast(device):
        loss = loss_fn(model(x), y)
    loss.backward()
    opt.step()


def time_training_step(backbone: str, num_classes: int, rec: RecognizerConfig, tr: TrainingConfig,
                       device: torch.device, iters: int = 2) -> tuple[float, str | None]:
    """Seconds per optimizer step of ``batch_size`` images, timed on random data (used by --dry-run).
    Returns (seconds, note); the note says when the full batch does not fit in memory."""
    bs = min(tr.batch_size, 16) if device.type == "cpu" else tr.batch_size
    note = None
    while True:
        try:
            model = create_backbone(backbone, num_classes, pretrained=False).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=tr.learning_rate)
            x = torch.rand(bs, 3, rec.input_size, rec.input_size, device=device)
            y = torch.randint(0, num_classes, (bs,), device=device)
            loss_fn = nn.CrossEntropyLoss()
            model.train()

            _timed_step(model, opt, loss_fn, x, y, device)
            synchronize(device)
            t0 = time.perf_counter()
            for _ in range(iters):
                _timed_step(model, opt, loss_fn, x, y, device)
            synchronize(device)
            per_step = (time.perf_counter() - t0) / iters
            return per_step * (tr.batch_size / bs), note
        except RuntimeError as err:
            if not is_oom(err):
                raise
            free_memory(device)
            if bs > 1:
                bs = max(1, bs // 2)
                if device.type != "cpu":
                    note = (f"{backbone}: a batch of {tr.batch_size} does not fit in {device.type.upper()} memory; "
                            "training will use gradient accumulation")
            elif device.type != "cpu":
                device, bs = CPU, min(tr.batch_size, 16)
                note = f"{backbone}: does not fit in GPU memory; training will fall back to the CPU"
            else:
                raise
        finally:
            free_memory(device)
