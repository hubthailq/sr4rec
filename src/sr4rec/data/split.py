"""Train / val / test split for closed-set recognition.

Every class appears in train, val and test, and no image (not even a renamed
copy with the same content) appears in two splits. See section "Split" of
docs/methods.md for the exact rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import SplitConfig
from ..errors import DatasetError
from ..utils import round_half_up, sha256_file, stable_key
from .dataset import GUIDE, MAX_LISTED, Dataset


@dataclass
class SplitResult:
    table: pd.DataFrame  # columns: path, label, split (order of labels.csv)
    source: str  # "column" or "ratios"
    description: str  # human-readable, for the report
    key: str
    moved_to_val: int = 0
    warnings: list[str] | None = None

    def paths(self, split: str) -> list[str]:
        return self.table.loc[self.table["split"] == split, "path"].tolist()

    def counts(self) -> dict[str, int]:
        return {s: int((self.table["split"] == s).sum()) for s in ("train", "val", "test")}


def _ratio_counts(n: int, val: float, test: float) -> tuple[int, int, int]:
    n_test = max(1, round_half_up(n * test))
    n_val = max(1, round_half_up(n * val))
    return n - n_test - n_val, n_val, n_test


def minimum_class_size(val: float, test: float, horizon: int = 200) -> int:
    """Smallest n such that every class of n or more images gets at least one train image."""
    ok = [(_ratio_counts(n, val, test)[0] >= 1) for n in range(1, horizon + 1)]
    need = horizon
    while need > 1 and ok[need - 2]:
        need -= 1
    return need


def _groups(table: pd.DataFrame, idx) -> list[np.ndarray]:
    """Rows of ``idx`` grouped by identical content (SHA-256), in path order. Identical images are
    always placed in the same split."""
    sub = table.loc[idx].sort_values("path")
    key = sub["sha256"].where(sub["sha256"] != "", sub["path"])
    return [g.index.to_numpy() for _, g in sub.groupby(key, sort=False)]


def _take(groups: list[np.ndarray], order: np.ndarray, k: int, used: set[int]) -> list[int]:
    """Take whole groups (in ``order``) until at least ``k`` rows are collected."""
    out: list[int] = []
    for gi in order:
        if len(out) >= k:
            break
        if gi in used:
            continue
        used.add(int(gi))
        out.extend(groups[gi].tolist())
    return out


def _list_error(title: str, problems: list[str]) -> DatasetError:
    shown = problems[:MAX_LISTED]
    lines = [title] + [f"  - {p}" for p in shown]
    if len(problems) > len(shown):
        lines.append(f"  ... and {len(problems) - len(shown)} more")
    lines.append(GUIDE)
    return DatasetError("\n".join(lines))


def resolve_source(ds: Dataset, cfg: SplitConfig) -> str:
    if cfg.source == "column":
        if not ds.has_split_column:
            raise DatasetError(f"split.source is 'column' but labels.csv has no split column.\n{GUIDE}")
        return "column"
    if cfg.source == "ratios":
        return "ratios"
    return "column" if ds.has_split_column else "ratios"


def split_by_ratios(ds: Dataset, cfg: SplitConfig) -> pd.DataFrame:
    r = cfg.ratios
    rng = np.random.default_rng(cfg.seed)
    assign = pd.Series("", index=ds.table.index, dtype=object)
    too_small = []
    need = minimum_class_size(r.val, r.test)
    for label in ds.classes:
        idx = ds.table.index[ds.table["label"] == label]
        groups = _groups(ds.table, idx)
        n = len(idx)
        _, n_val, n_test = _ratio_counts(n, r.val, r.test)
        order = rng.permutation(len(groups))
        used: set[int] = set()
        test_rows = _take(groups, order, n_test, used)
        val_rows = _take(groups, order, n_val, used)
        train_rows = [i for gi in order if gi not in used for i in groups[gi]]
        if not train_rows or len(val_rows) < 1 or len(test_rows) < 1:
            too_small.append(f"class {label!r} has {n} image(s); with these ratios it gets no train image "
                             f"(classes of at least {need} different images always work)")
            continue
        assign.loc[test_rows] = "test"
        assign.loc[val_rows] = "val"
        assign.loc[train_rows] = "train"
    if too_small:
        raise _list_error(
            f"With split.ratios train {r.train} / val {r.val} / test {r.test}, every class needs at least one "
            "image in each split. These classes are too small:",
            too_small,
        )
    out = ds.table[["path", "label"]].copy()
    out["split"] = assign.values
    return out


def split_from_column(ds: Dataset, cfg: SplitConfig) -> tuple[pd.DataFrame, int]:
    out = ds.table[["path", "label", "split"]].copy()
    present = set(out["split"])
    if "train" not in present or "test" not in present:
        raise DatasetError(f"The split column must contain at least 'train' and 'test' rows (found: {sorted(present)}).\n{GUIDE}")
    moved = 0
    if "val" not in present:
        rng = np.random.default_rng(cfg.seed)
        problems = []
        for label in ds.classes:
            idx = out.index[(out["label"] == label) & (out["split"] == "train")]
            groups = _groups(ds.table, idx)
            n = len(idx)
            k = max(1, round_half_up(n * cfg.val_from_train))
            rows = _take(groups, rng.permutation(len(groups)), k, set())
            if n - len(rows) < 1:
                problems.append(f"class {label!r} has {n} train image(s) ({len(groups)} distinct); at least 2 "
                                "distinct images are needed to move one to val")
                continue
            out.loc[rows, "split"] = "val"
            moved += len(rows)
        if problems:
            raise _list_error("The split column has no val rows, so SR4Rec moves part of train to val; this fails for:", problems)
    problems = []
    for label in ds.classes:
        have = set(out.loc[out["label"] == label, "split"])
        lacking = [s for s in ("train", "val", "test") if s not in have]
        if lacking:
            problems.append(f"class {label!r} has no image in: {', '.join(lacking)}")
    if problems:
        raise _list_error("Closed-set recognition needs every class in train, val and test:", problems)
    return out, moved


def check_duplicates(ds: Dataset, split: pd.DataFrame) -> list[str]:
    """Stop if identical image content appears in two splits; return warnings for duplicates within a split."""
    tab = ds.table[["path", "sha256", "line"]].merge(split[["path", "split"]], on="path")
    tab = tab[tab["sha256"] != ""]
    problems, warnings = [], []
    for _sha, group in tab.groupby("sha256"):
        if len(group) < 2:
            continue
        where = ", ".join(f"{p} ({s}, line {ln})" for p, s, ln in zip(group["path"], group["split"], group["line"], strict=False))
        if group["split"].nunique() > 1:
            problems.append(f"identical images in different splits: {where}")
        else:
            warnings.append(f"identical images within one split: {where}")
    if problems:
        raise _list_error("The same image content (SHA-256) appears in more than one split:", problems)
    return warnings


def make_split(ds: Dataset, cfg: SplitConfig, cache_dir: Path | None = None) -> SplitResult:
    """Create (or reuse) the split. The split is stored in ``cache_dir/splits`` and reused
    while the data and the ``split`` block stay the same."""
    source = resolve_source(ds, cfg)
    key = stable_key("split-v2", ds.content_key, source, cfg.model_dump(mode="json") if source == "ratios" else
                     {"val_from_train": cfg.val_from_train, "seed": cfg.seed})
    warnings: list[str] = []
    stored = None
    if cache_dir is not None:
        folder = cache_dir / "splits" / stable_key("dataset-location", str(ds.root))[:16]
        stored = folder / f"split_{key[:16]}.csv"
        last = folder / "last_key.txt"
        if last.is_file() and last.read_text(encoding="utf-8").strip() != key:
            warnings.append("The data or the split block changed since the previous run on this dataset, so the split differs from that run.")
    moved = 0
    if stored is not None and stored.is_file():
        table = pd.read_csv(stored, dtype=str, keep_default_na=False)
    elif source == "ratios":
        table = split_by_ratios(ds, cfg)
    else:
        table, moved = split_from_column(ds, cfg)
    warnings += check_duplicates(ds, table)
    if cache_dir is not None and stored is not None:
        stored.parent.mkdir(parents=True, exist_ok=True)
        if not stored.is_file():
            table.to_csv(stored, index=False)
        (stored.parent / "last_key.txt").write_text(key, encoding="utf-8")
    counts = {s: int((table["split"] == s).sum()) for s in ("train", "val", "test")}
    count_text = f"train {counts['train']:,} / val {counts['val']:,} / test {counts['test']:,} images"
    if source == "ratios":
        r = cfg.ratios
        desc = f"created by SR4Rec from ratios {r.train:.2f} / {r.val:.2f} / {r.test:.2f} (split seed {cfg.seed}): {count_text}"
    else:
        desc = "from the split column of labels.csv"
        if "val" not in set(ds.table["split"]):
            desc += f"; val made from {cfg.val_from_train:.2f} of train (split seed {cfg.seed})"
        desc += f": {count_text}"
    return SplitResult(table=table, source=source, description=desc, key=key, moved_to_val=moved, warnings=warnings)


def write_split(result: SplitResult, path: Path) -> str:
    result.table.to_csv(path, index=False)
    return sha256_file(path)
