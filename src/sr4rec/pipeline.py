"""The single entry point of SR4Rec: the :class:`Run` class.

The command-line interface only builds a :class:`Run` and calls
:meth:`Run.execute`, so the CLI and the Python API give identical results::

    from sr4rec import Run, load_config
    result = Run(load_config("sr4rec.yaml")).execute()
    print(result.report)
"""

from __future__ import annotations

import math
import os
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from . import __version__, stats
from .config import LoadedConfig, config_to_dict, unconfirmed_lines
from .data.dataset import Dataset, check_mode_constraints, load_dataset
from .data.sizes import MIN_IMAGES_PER_BIN, SIZE_BINS, lr_size, size_bin
from .data.split import SplitResult, make_split, write_split
from .device import name as device_name
from .device import resolve as resolve_device
from .errors import ConfigError, MissingResource, SR4RecError, SRModelError
from .eval.metrics import CMC_MAX_RANK, cmc, psnr_y, recognition_metrics, ssim_y
from .fingerprint import environment, now_text
from .lr.letterbox import PREPROCESS_VERSION
from .lr.resize import RESIZE_VERSION, downsample
from .rec.data import Item
from .rec.engine import predict, set_determinism, time_training_step, train_recognizer
from .rec.models import check_backbone_name
from .sr.base import SRSource
from .sr.folder_source import FolderSR
from .sr.runner import RobustRunner, compute_sr
from .sr.validated import REAL_WORLD
from .utils import load_rgb, mod_crop, pil_to_tensor, png_name, save_png, stable_key

HR = "hr"
BICUBIC = "bicubic"
SPLITS = ("train", "val", "test")


@dataclass
class Training:
    """One recognizer: backbone x seed x image pipeline used for train/val."""

    backbone: str
    seed: int
    source: str  # method name whose train/val images are used
    key: str = ""


@dataclass
class RunResult:
    run_dir: Path | None
    report: Path | None
    summary: list[str] = field(default_factory=list)
    dry_run_text: str | None = None


class Run:
    """Run the whole SR4Rec workflow for one configuration."""

    def __init__(self, loaded: LoadedConfig, dry_run: bool = False, checkpoints_dir: Path | None = None,
                 log: Callable[[str], None] | None = print, command: str = "sr4rec run"):
        self.loaded = loaded
        self.cfg = loaded.config
        self.dry_run = dry_run
        self.checkpoints_dir = checkpoints_dir
        self.log = log or (lambda _msg: None)
        self.command = command
        self.warnings: list[str] = []
        self.timings: dict[str, float] = {}

    # ------------------------------------------------------------------ preparation

    def _check_markers(self) -> None:
        text = self.loaded.raw_text
        if not text:
            return
        found = unconfirmed_lines(text)
        if found:
            lines = "\n".join(f"  line {n}: {line.strip()}" for n, line in found)
            raise ConfigError(
                "The configuration still contains [CHECK] markers. Confirm or edit each value, then delete the "
                f"\"[CHECK]\" marker:\n{lines}"
            )

    def _device(self) -> torch.device:
        device, warnings = resolve_device(self.cfg.runtime.device)
        self.warnings += warnings
        return device

    def prepare(self) -> None:
        cfg = self.cfg
        self._check_markers()
        self.device = self._device()
        self.dataset_path = self.loaded.resolve(cfg.dataset)
        self.ds: Dataset = load_dataset(self.dataset_path)
        check_mode_constraints(self.ds, cfg.mode, cfg.scale)
        self.warnings += self.ds.notes
        for b in cfg.recognizer.backbones:
            w = check_backbone_name(b)
            if w:
                self.warnings.append(w)
        self.output_dir = self.loaded.resolve(cfg.runtime.output_dir)
        self.cache_dir = self.loaded.resolve(cfg.runtime.cache_dir) if cfg.runtime.cache_dir else self.output_dir / ".cache"
        self.split: SplitResult = make_split(self.ds, cfg.split, self.cache_dir)
        self.warnings += self.split.warnings or []
        tab = self.ds.table.drop(columns=["split"]).merge(self.split.table[["path", "split"]], on="path")
        self.table = tab
        self.label_of = dict(zip(tab["path"], tab["label"], strict=False))
        self.class_index = {c: i for i, c in enumerate(self.ds.classes)}
        self.lr_dims = {p: lr_size(w, h, cfg.mode, cfg.scale) for p, w, h in zip(tab["path"], tab["width"], tab["height"], strict=False)}
        self.bin_of = {p: size_bin(min(d)) for p, d in self.lr_dims.items()}
        self.paths = {s: tab.loc[tab["split"] == s, "path"].tolist() for s in SPLITS}
        self.lr_key = stable_key("lr", self.ds.content_key, cfg.mode, cfg.scale, RESIZE_VERSION)
        self.synthetic = cfg.mode == "synthetic"
        self.methods = [BICUBIC] + [m.name for m in cfg.sr]
        self.eval_methods = self.methods + ([HR] if self.synthetic else [])
        self.fixed = cfg.protocol == "fixed_recognizer"
        self.fixed_source = HR if self.synthetic else BICUBIC

    def needed_splits(self, method: str) -> tuple[str, ...]:
        if not self.fixed:
            return SPLITS
        if method == self.fixed_source:
            return SPLITS
        return ("test",)

    def load_sources(self, collect_errors: bool = False) -> tuple[list[SRSource], list[tuple[str, str]]]:
        from .sr.bicubic import BicubicSR
        from .sr.module_source import ModuleSR
        from .sr.pth_source import SpandrelSR

        sources: list[SRSource] = [BicubicSR(self.cfg.scale)]
        errors: list[tuple[str, str]] = []
        for entry in self.cfg.sr:
            try:
                if entry.method == "images":
                    src = FolderSR(entry.name, self.loaded.resolve(entry.images), self.cfg.scale)
                    expected = {p: self.lr_dims[p] for s in self.needed_splits(entry.name) for p in self.paths[s]}
                    status, warns = src.check(expected, set(self.label_of))
                    src.description = f"images   {status}"
                    self.warnings += warns
                elif entry.method == "module":
                    src = ModuleSR(entry.name, entry.module, self.loaded.resolve, self.cfg.scale,
                                   self.loaded.resolve(entry.weights), entry.tile)
                else:
                    src = SpandrelSR(entry.name, self.loaded.resolve(entry.weights), self.cfg.scale, entry.tile)
                sources.append(src)
            except SRModelError as err:
                if not collect_errors:
                    raise
                errors.append((entry.name, str(err)))
        self.sources = {s.name: s for s in sources}
        return sources, errors

    # ------------------------------------------------------------------ image files

    def lr_file(self, path: str) -> Path:
        if self.synthetic:
            return self.cache_dir / "lr" / self.lr_key[:20] / png_name(path)
        return self.ds.image_path(path)

    def method_key(self, method: str) -> str:
        if method == HR:
            return stable_key("hr", self.ds.content_key, self.cfg.scale)
        src = self.sources[method]
        if src.kind == "images":
            return src.key
        return stable_key(src.key, self.lr_key)

    def method_dir(self, method: str) -> Path:
        return self.cache_dir / "sr" / f"{method}_{self.method_key(method)[:16]}"

    def image_item(self, method: str, path: str) -> Item:
        target = self.class_index[self.label_of[path]]
        if method == HR:
            return Item(path, self.ds.image_path(path), target, crop_scale=self.cfg.scale)
        src = self.sources[method]
        if isinstance(src, FolderSR):
            return Item(path, src.file_for(path), target)
        return Item(path, self.method_dir(method) / png_name(path), target)

    def make_lr_images(self, run_dir: Path | None) -> None:
        if not self.synthetic:
            return
        todo = [p for p in self.table["path"] if not self.lr_file(p).is_file()]
        if todo:
            self.log(f"  Downsampling {len(todo):,} HR images x{self.cfg.scale} (MATLAB-style bicubic)")
        for p in todo:
            hr = mod_crop(load_rgb(self.ds.image_path(p)), self.cfg.scale)
            save_png(downsample(pil_to_tensor(hr), self.cfg.scale), self.lr_file(p))
        if run_dir is not None:
            for s in SPLITS:
                for p in self.paths[s]:
                    dst = run_dir / "lr" / s / png_name(p)
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        os.link(self.lr_file(p), dst)
                    except OSError:
                        shutil.copyfile(self.lr_file(p), dst)

    def make_sr_images(self) -> None:
        for name in self.methods:
            src = self.sources[name]
            if not src.computes:
                continue
            paths = [p for s in self.needed_splits(name) for p in self.paths[s]]
            items = [(p, self.lr_file(p)) for p in paths]
            out = self.method_dir(name)
            n_new = sum(1 for p in paths if not (out / png_name(p)).is_file())
            if n_new:
                self.log(f"  SR {name}: {n_new:,} images")
            secs, sr_stats = compute_sr(src, items, out, self.device, self.log)
            self.timings[f"sr_{name}"] = secs
            w = sr_stats.warning(name, self.device)
            if w:
                self.warnings.append(w)
                self.log(f"  Warning: {w}")

    # ------------------------------------------------------------------ recognizers

    def trainings(self) -> list[Training]:
        rec = self.cfg.recognizer
        sources = [self.fixed_source] if self.fixed else self.eval_methods
        out = []
        for b in rec.backbones:
            for s in sources:
                for seed in rec.seeds:
                    t = Training(b, seed, s)
                    t.key = stable_key("ckpt-v1", PREPROCESS_VERSION, b, seed, self.ds.classes, rec.pretrained, rec.input_size,
                                       self.cfg.training.model_dump(mode="json"),
                                       self.split.key, self.method_key(s))
                    out.append(t)
        return out

    def _checkpoint_path(self, t: Training) -> Path:
        if self.checkpoints_dir is not None:
            return self.checkpoints_dir / f"{t.backbone}__{t.source}__seed{t.seed}.pt"
        return self.cache_dir / "checkpoints" / f"{t.key[:32]}.pt"

    def get_state(self, t: Training, index: int, total: int) -> tuple[dict, dict]:
        path = self._checkpoint_path(t)
        if path.is_file():
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
            if ckpt.get("key") != t.key:
                raise SR4RecError(
                    f"Checkpoint {path.name} was trained with different data, split, SR model or training settings "
                    "(its key does not match). Delete it, or check that the dataset and weights match the demo."
                )
            return ckpt["state"], ckpt["meta"]
        if self.checkpoints_dir is not None:
            raise MissingResource(f"Recognizer checkpoint not found: {path}. Download the published checkpoints "
                                  "with: python scripts/fetch_checkpoints.py <demo>")
        self.log(f"  Training {index}/{total}: {t.backbone}, train images: {t.source}, seed {t.seed}")
        rec, tr = self.cfg.recognizer, self.cfg.training
        train_items = [self.image_item(t.source, p) for p in self.paths["train"]]
        val_items = [self.image_item(t.source, p) for p in self.paths["val"]]
        res = train_recognizer(t.backbone, len(self.ds.classes), train_items, val_items, rec, tr, t.seed, self.device,
                               log=self.log)
        meta = {"backbone": t.backbone, "seed": t.seed, "source": t.source, "best_epoch": res.best_epoch,
                "best_val_rank1": res.best_val_rank1, "seconds": res.seconds, "sr4rec_version": __version__,
                "accumulation": res.accumulation, "device": res.device}
        if res.notes:
            self.warnings.append(
                f"Recognizer {t.backbone} (train images: {t.source}, seed {t.seed}): " + "; ".join(res.notes)
                + (f". It was trained with {res.accumulation} micro-batches per step" if res.accumulation > 1 else "")
                + (" on the CPU" if res.device == "cpu" and self.device.type != "cpu" else "")
                + ". The effective batch size is unchanged; BatchNorm statistics use the smaller micro-batches.")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        torch.save({"key": t.key, "state": res.state, "meta": meta}, tmp)
        tmp.replace(path)
        return res.state, meta

    def evaluate(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        rec, tr = self.cfg.recognizer, self.cfg.training
        trainings = self.trainings()
        test = self.paths["test"]
        rows, ckpt_rows = [], []
        for i, t in enumerate(trainings, 1):
            state, meta = self.get_state(t, i, len(trainings))
            ckpt_rows.append({"backbone": t.backbone, "source": t.source, "seed": t.seed, "key": t.key,
                              "file": str(self._checkpoint_path(t)), "best_epoch": meta.get("best_epoch"),
                              "best_val_rank1": meta.get("best_val_rank1")})
            targets = self.eval_methods if self.fixed else [t.source]
            for m in targets:
                items = [self.image_item(m, p) for p in test]
                notes: list[str] = []
                ranks, pred = predict(t.backbone, len(self.ds.classes), state, items, rec, tr.batch_size,
                                      tr.num_workers, self.device, notes)
                if notes:
                    self.warnings.append(f"Evaluation of {t.backbone} on {m} (seed {t.seed}): {notes[-1]}; "
                                         "results do not depend on the evaluation batch size.")
                for p, r, k in zip(test, ranks, pred, strict=False):
                    rows.append({"backbone": t.backbone, "method": m, "seed": t.seed, "path": p,
                                 "label": self.label_of[p], "pred": self.ds.classes[int(k)], "true_rank": int(r),
                                 "correct": int(r == 1)})
        return pd.DataFrame(rows), pd.DataFrame(ckpt_rows)

    # ------------------------------------------------------------------ metrics

    def quality(self) -> pd.DataFrame:
        if not self.synthetic:
            return pd.DataFrame(columns=["method", "path", "psnr", "ssim"])
        rows = []
        s = self.cfg.scale
        for p in self.paths["test"]:
            hr = np.asarray(mod_crop(load_rgb(self.ds.image_path(p)), s))
            for m in self.methods:
                sr = np.asarray(load_rgb(self.image_item(m, p).file))
                if sr.shape != hr.shape:
                    raise SR4RecError(f"SR output of {m!r} for {p} has shape {sr.shape}, HR has {hr.shape}.")
                rows.append({"method": m, "path": p, "psnr": psnr_y(sr, hr, s), "ssim": ssim_y(sr, hr, s)})
        return pd.DataFrame(rows)

    def metrics(self, pred: pd.DataFrame, quality: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        n_cls = len(self.ds.classes)
        pred = pred.assign(size_bin=pred["path"].map(self.bin_of))
        qual = quality.assign(size_bin=quality["path"].map(self.bin_of)) if len(quality) else quality
        rows, cmc_rows = [], []
        for (b, m, seed), g in pred.groupby(["backbone", "method", "seed"], sort=False):
            for sb in ("all",) + SIZE_BINS:
                gg = g if sb == "all" else g[g["size_bin"] == sb]
                if not len(gg):
                    continue
                yt = gg["label"].map(self.class_index).to_numpy()
                yp = gg["pred"].map(self.class_index).to_numpy()
                met = recognition_metrics(gg["true_rank"].to_numpy(), yt, yp, n_cls,
                                          labels=None if sb == "all" else np.unique(yt))
                q = qual[(qual["method"] == m)] if len(qual) else qual
                if len(q) and sb != "all":
                    q = q[q["size_bin"] == sb]
                psnr = float(q["psnr"].mean()) if len(q) else float("nan")
                ssim = float(q["ssim"].mean()) if len(q) else float("nan")
                rows.append({"backbone": b, "method": m, "seed": seed, "size_bin": sb, "n_images": len(gg), **met,
                             "psnr": psnr, "ssim": ssim})
            cm = cmc(g["true_rank"].to_numpy(), n_cls, CMC_MAX_RANK)
            cmc_rows.append({"backbone": b, "method": m, "seed": seed,
                             **{f"rank{k}": v for k, v in enumerate(cm, 1)}})
        return pd.DataFrame(rows), pd.DataFrame(cmc_rows)

    def statistics(self, pred: pd.DataFrame) -> pd.DataFrame:
        rows = []
        test = self.paths["test"]
        seeds = self.cfg.recognizer.seeds
        sr_methods = self.methods[1:]
        for b in self.cfg.recognizer.backbones:
            sub = pred[pred["backbone"] == b]
            mats = {}
            for m in [BICUBIC] + sr_methods:
                piv = sub[sub["method"] == m].pivot(index="seed", columns="path", values="correct")
                mats[m] = piv.loc[seeds, test].to_numpy(dtype=float)
            for sb in ("all",) + SIZE_BINS:
                cols = np.arange(len(test)) if sb == "all" else np.flatnonzero([self.bin_of[p] == sb for p in test])
                if len(cols) == 0:
                    continue
                group = []
                for m in sr_methods:
                    base = {"backbone": b, "size_bin": sb, "method": m, "n_images": len(cols)}
                    if sb != "all" and len(cols) < MIN_IMAGES_PER_BIN:
                        d = 100 * float(mats[m][:, cols].mean() - mats[BICUBIC][:, cols].mean())
                        rows.append({**base, "delta": d, "ci_low": np.nan, "ci_high": np.nan, "p_raw": np.nan,
                                     "p_holm": np.nan, "same_sign": "", "verdict": stats.INSUFFICIENT})
                        continue
                    c = stats.compare(mats[m][:, cols], mats[BICUBIC][:, cols])
                    group.append({**base, **{k: c[k] for k in ("delta", "ci_low", "ci_high", "p_raw")},
                                  "same_sign": f"{c['same_sign']}/{c['n_seeds']}"})
                for r, ph in zip(group, stats.holm([r["p_raw"] for r in group]), strict=False):
                    r["p_holm"] = ph
                    r["verdict"] = stats.verdict(r["delta"], r["ci_low"], r["ci_high"], ph)
                    rows.append(r)
        cols = ["backbone", "size_bin", "method", "n_images", "delta", "ci_low", "ci_high", "p_raw", "p_holm",
                "same_sign", "verdict"]
        return pd.DataFrame(rows, columns=cols)

    # ------------------------------------------------------------------ dry run

    def _plan(self) -> tuple[int, int, int]:
        trainings = self.trainings()
        todo_train = sum(1 for t in trainings if not self._checkpoint_path(t).is_file())
        n_sr = 0
        for name in self.methods:
            src = self.sources.get(name)
            if src is None or not src.computes:
                continue
            out = self.method_dir(name)
            n_sr += sum(1 for s in self.needed_splits(name) for p in self.paths[s]
                        if not (out / png_name(p)).is_file())
        return len(trainings), todo_train, n_sr

    def dry_run_report(self) -> str:
        cfg = self.cfg
        lines = ["Config              OK"]
        lines.append(f"Dataset             OK  ({len(self.table):,} images, {len(self.ds.classes):,} classes)")
        c = self.split.counts()
        lines.append(f"Split               OK  train {c['train']:,} / val {c['val']:,} / test {c['test']:,} images")
        sources, errors = self.load_sources(collect_errors=True)
        lines.append("SR models")
        sample = self.paths["test"][:2]
        sr_seconds = {}
        err_map = dict(errors)
        for entry_name in self.methods:
            if entry_name in err_map:
                first = err_map[entry_name].splitlines()[0]
                lines.append(f"  {entry_name:<17} FAILED  {first}")
                continue
            src = self.sources[entry_name]
            if src.computes:
                try:
                    if self.synthetic:
                        lr_imgs = [downsample(pil_to_tensor(mod_crop(load_rgb(self.ds.image_path(p)), cfg.scale)),
                                              cfg.scale) for p in sample]
                    else:
                        lr_imgs = [pil_to_tensor(load_rgb(self.ds.image_path(p))) for p in sample]
                    runner = RobustRunner(src, self.device)
                    t0 = time.perf_counter()
                    for k, lr in enumerate(lr_imgs):
                        if k == 1:
                            t0 = time.perf_counter()
                        y = runner.upscale(lr[None])
                        exp = (1, 3, lr.shape[1] * cfg.scale, lr.shape[2] * cfg.scale)
                        if tuple(y.shape) != exp:
                            raise SRModelError(f"output shape {tuple(y.shape)}, expected {exp}")
                        if not torch.isfinite(y).all():
                            raise SRModelError("output contains NaN or infinite values")
                    sr_seconds[entry_name] = time.perf_counter() - t0
                    runner.close()
                    fallback = runner.stats.warning(entry_name, self.device)
                    if fallback:
                        src.description += "  (memory fallback: " + (
                            "CPU" if runner.stats.on_cpu else f"tile {runner.stats.tile} px") + ")"
                except Exception as err:  # noqa: BLE001
                    errors.append((entry_name, str(err)))
                    lines.append(f"  {entry_name:<17} FAILED  {err}")
                    continue
            lines.append(f"  {entry_name:<17} OK  x{src.scale}  {src.description}")
        if errors:
            text = "\n".join(lines)
            details = "\n\n".join(f"{n}:\n{e}" for n, e in errors)
            raise SRModelError(f"{text}\n\nDry run failed. Details:\n{details}")
        total, todo, n_sr = self._plan()
        rec = cfg.recognizer
        pipes = f"{len(self.methods)} image pipelines (bicubic + {len(self.methods) - 1} SR)"
        if self.synthetic:
            pipes += " + HR"
        lines.append(f"Plan                protocol {cfg.protocol}: {pipes}, {len(rec.backbones)} backbone(s), "
                     f"{len(rec.seeds)} seed(s)")
        lines.append(f"                    = {total:,} recognizer trainings ({todo:,} not cached), "
                     f"{n_sr:,} SR images to compute")
        est = 0.0
        n_train = len(self.paths["train"])
        n_eval = len(self.paths["val"]) * cfg.training.epochs + len(self.paths["test"]) * (
            len(self.eval_methods) if self.fixed else 1)
        per_backbone = {}
        memory_notes: list[str] = []
        for t in self.trainings():
            if self._checkpoint_path(t).is_file():
                continue
            if t.backbone not in per_backbone:
                secs_step, note = time_training_step(t.backbone, len(self.ds.classes), rec, cfg.training,
                                                     self.device)
                per_backbone[t.backbone] = secs_step
                if note:
                    memory_notes.append(note)
            step = per_backbone[t.backbone]
            batches = math.ceil(n_train / cfg.training.batch_size) * cfg.training.epochs
            est += step * batches + step / 3 * math.ceil(n_eval / cfg.training.batch_size)
        from .latency import RUNS, WARMUP

        sample_area = float(np.mean([self.lr_dims[p][0] * self.lr_dims[p][1] for p in sample])) if sample else 1.0
        for name, secs in sr_seconds.items():
            n = sum(1 for s in self.needed_splits(name) for p in self.paths[s]
                    if not (self.method_dir(name) / png_name(p)).is_file())
            est += secs * n
            if cfg.latency.enabled:  # timed on the CPU, SR alone and once per backbone in the pipeline
                est += secs * (cfg.latency.lr_size**2 / sample_area) * (WARMUP + RUNS) * (1 + len(rec.backbones))
        from .report.formatting import fmt_duration

        lines.append(f"Estimated time      about {fmt_duration(est)} on {device_name(self.device)} "
                     "(from a timed mini-batch)")
        for note in memory_notes + [w for w in self.warnings if w.startswith("runtime.device")]:
            lines.append(f"Note                {note}")
        src_file = self.loaded.source_file
        lines.append(f"Dry run passed. Start the real run with: sr4rec run {src_file if src_file else '<config>'}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ outputs

    def _run_dir(self) -> Path:
        name = self.cfg.runtime.run_name or f"{self.ds.name}_{time.strftime('%Y-%m-%d_%H%M')}"
        base = self.output_dir / name
        run_dir, k = base, 2
        while run_dir.exists():
            run_dir = base.with_name(f"{base.name}_{k}")
            k += 1
        run_dir.mkdir(parents=True)
        return run_dir

    def _write_config_copy(self, run_dir: Path) -> None:
        data = config_to_dict(self.cfg)

        def rel(p: str | None) -> str | None:
            if p is None:
                return None
            target = self.loaded.resolve(p)
            try:
                return Path(os.path.relpath(target, run_dir)).as_posix()
            except ValueError:  # another drive on Windows
                return str(target)

        data["dataset"] = rel(self.cfg.dataset)
        for e in data["sr"]:
            for k in ("weights", "images"):
                e[k] = rel(e[k])
            if e["module"]:
                f, c = e["module"].rsplit(":", 1)
                e["module"] = f"{rel(f)}:{c}"
        data["sr"] = [{k: v for k, v in e.items() if v is not None} for e in data["sr"]]
        data["runtime"]["output_dir"] = rel(self.cfg.runtime.output_dir)
        data["runtime"]["cache_dir"] = rel(str(self.cache_dir))
        data["runtime"]["run_name"] = None
        header = (f"# Exact configuration of this run (SR4Rec {__version__}).\n"
                  f"# Copied from {self.loaded.source_file or '<Python API>'}; relative paths were rewritten\n"
                  "# relative to this folder. Re-run with: sr4rec run <this file>\n")
        (run_dir / "sr4rec.yaml").write_text(header + yaml.safe_dump(data, sort_keys=False, allow_unicode=False),
                                             encoding="utf-8")

    def execute(self) -> RunResult:
        t_start = time.perf_counter()
        self.prepare()
        if self.dry_run:
            text = self.dry_run_report()
            return RunResult(None, None, dry_run_text=text)
        # SR inference has no seed of its own, but cuDNN's default convolution algorithms are
        # not deterministic between runs; without this, SR outputs (and everything computed
        # from them) can drift slightly even on identical hardware. train_recognizer() sets the
        # same flags again per seed; that call is idempotent for them.
        set_determinism(0)
        sources, _ = self.load_sources()
        self._add_source_warnings()
        run_dir = self._run_dir()
        self.log(f"SR4Rec {__version__}: run folder {run_dir}")
        self._write_config_copy(run_dir)
        split_sha = write_split(self.split, run_dir / "split.csv")
        self.make_lr_images(run_dir)
        self.make_sr_images()
        quality = self.quality()
        pred, ckpts = self.evaluate()
        metrics, cmc_df = self.metrics(pred, quality)
        stat = self.statistics(pred)
        latency = []
        if self.cfg.latency.enabled:
            self.log("  Measuring CPU latency")
            from .latency import measure

            latency = measure(sources, self.cfg.recognizer.backbones, len(self.ds.classes), self.cfg.latency.lr_size,
                              self.cfg.latency.threads, self.cfg.recognizer.input_size)
        pred.to_csv(run_dir / "predictions.csv", index=False)
        metrics.to_csv(run_dir / "metrics.csv", index=False)
        cmc_df.to_csv(run_dir / "cmc.csv", index=False, na_rep="n/a")
        stat.to_csv(run_dir / "stats.csv", index=False)
        ckpts.to_csv(run_dir / "checkpoints.csv", index=False)
        pd.DataFrame(latency, columns=["component", "sr", "backbone", "median_ms", "p95_ms"]).to_csv(
            run_dir / "latency.csv", index=False)
        if len(quality):
            quality.to_csv(run_dir / "quality.csv", index=False)

        from .report.render import write_outputs

        finished, _ = now_text()
        total = time.perf_counter() - t_start
        env = environment(self.device)
        report, summary = write_outputs(self, run_dir, pred, metrics, cmc_df, stat, quality, latency,
                                        finished, total, env)
        fp = {
            "sr4rec_version": __version__,
            "created": finished,
            "command": self.command,
            "config": config_to_dict(self.cfg),
            "seeds": {"recognizer": list(self.cfg.recognizer.seeds), "split": self.cfg.split.seed,
                      "statistics": stats.STATS_SEED},
            "environment": env,
            "dataset": {"path": str(self.dataset_path), "images": len(self.table), "classes": len(self.ds.classes),
                        "labels_csv_sha256": self.ds.labels_sha256, "content_sha256": self.ds.content_key,
                        "split_csv_sha256": split_sha, "split_source": self.split.source},
            "sr_models": [{"name": s.name, "method": s.kind, "scale": s.scale,
                           "files": [{"path": k, "sha256": v} for k, v in s.files.items()]} for s in sources],
            "total_seconds": round(total, 1),
        }
        (run_dir / "fingerprint.yaml").write_text(yaml.safe_dump(fp, sort_keys=False), encoding="utf-8")
        for line in summary:
            self.log(line)
        return RunResult(run_dir, report, summary)

    def _add_source_warnings(self) -> None:
        if any(s.kind == "weights" for s in self.sources.values()):
            self.warnings.insert(0, "SR weights were trained on natural images (DIV2K and similar) and may not match "
                                    "this dataset.")
        rw = [s.name for s in self.sources.values() if s.trained_for == REAL_WORLD]
        if rw and self.synthetic:
            names = " and ".join(rw) if len(rw) <= 2 else ", ".join(rw[:-1]) + " and " + rw[-1]
            verb = "was" if len(rw) == 1 else "were"
            self.warnings.append(
                f"{names} {verb} trained for real-world degradation, but the LR images of this run were made by clean "
                "bicubic downsampling (synthetic mode). Their results here reflect that mismatch; see a native-lr run "
                "for real low-resolution images.")
        if self.fixed:
            self.warnings.append(
                "protocol fixed_recognizer: the recognizer was trained on "
                + ("HR images" if self.synthetic else "bicubic-upscaled images")
                + " and never saw SR outputs, so SR models are evaluated out of their training domain"
                + ("" if self.synthetic else " while bicubic is in-domain")
                + ". This answers \"does adding SR in front of an existing recognizer help?\"; use protocol matched "
                  "to compare systems trained with each SR model.")
        for s in self.sources.values():
            if s.kind == "images":
                self.warnings.append(f"CPU latency is not measured for {s.name} (pre-computed images).")
