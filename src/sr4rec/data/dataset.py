"""Read and validate a dataset in the SR4Rec format.

Required layout (see docs/preparing_dataset.md)::

    <dataset>/
    |-- images/      all images; any sub-folder structure is allowed
    `-- labels.csv   one row per image: path,label[,split]

``path`` is relative to ``images/`` and uses ``/``. ``label`` is a string.
``split`` is optional and must be ``train``, ``val`` or ``test``.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from PIL import Image, UnidentifiedImageError

from ..errors import DatasetError
from ..utils import IMAGE_EXTENSIONS, png_name, sha256_file, stable_key

SPLIT_VALUES = ("train", "val", "test")
MAX_LISTED = 20
GUIDE = "See 'Preparing your dataset' in the README (docs/preparing_dataset.md)."


@dataclass
class Dataset:
    """A validated dataset.

    ``table`` has one row per image, in the order of labels.csv, with columns
    ``path``, ``label``, ``split`` (empty string when labels.csv has no split
    column), ``width``, ``height``, ``sha256`` and ``line`` (line number in
    labels.csv, header = line 1).
    """

    root: Path
    table: pd.DataFrame
    has_split_column: bool
    labels_sha256: str
    content_key: str
    classes: list[str]
    n_converted_to_rgb: int
    notes: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.root.name

    @property
    def images_dir(self) -> Path:
        return self.root / "images"

    @property
    def labels_csv(self) -> Path:
        return self.root / "labels.csv"

    def image_path(self, rel: str) -> Path:
        return self.images_dir / rel

    def short_sides(self) -> pd.Series:
        return self.table[["width", "height"]].min(axis=1)


def _fail(title: str, problems: list[str]) -> DatasetError:
    shown = problems[:MAX_LISTED]
    more = len(problems) - len(shown)
    lines = [title] + [f"  - {p}" for p in shown]
    if more > 0:
        lines.append(f"  ... and {more} more")
    lines.append(GUIDE)
    return DatasetError("\n".join(lines))


def load_dataset(root: str | Path, hash_images: bool = True) -> Dataset:
    """Read ``root`` and check the format. Raises :class:`DatasetError` listing every problem."""
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise DatasetError(f"Dataset folder not found: {root}\n{GUIDE}")
    images_dir = root / "images"
    labels_csv = root / "labels.csv"
    missing = [p.name + ("/" if p.name == "images" else "") for p in (images_dir, labels_csv) if not p.exists()]
    if missing:
        raise DatasetError(f"The dataset folder {root} must contain images/ and labels.csv; missing: {', '.join(missing)}\n{GUIDE}")

    try:
        df = pd.read_csv(labels_csv, dtype=str, keep_default_na=False, quoting=csv.QUOTE_MINIMAL)
    except Exception as err:  # noqa: BLE001 - any parser error is reported the same way
        raise DatasetError(f"labels.csv could not be read as CSV: {err}\n{GUIDE}") from None
    df.columns = [c.strip() for c in df.columns]
    absent = [c for c in ("path", "label") if c not in df.columns]
    if absent:
        raise DatasetError(
            f"labels.csv must have the columns path and label (optional: split); missing: {', '.join(absent)}. "
            f"Found columns: {', '.join(df.columns)}\n{GUIDE}"
        )
    unknown = [c for c in df.columns if c not in ("path", "label", "split")]
    if unknown:
        raise DatasetError(f"labels.csv has unknown columns: {', '.join(unknown)}. Allowed: path, label, split.\n{GUIDE}")
    if len(df) == 0:
        raise DatasetError(f"labels.csv has no rows.\n{GUIDE}")

    has_split = "split" in df.columns
    df["path"] = df["path"].str.strip()
    df["label"] = df["label"].str.strip()
    df["split"] = df["split"].str.strip().str.lower() if has_split else ""
    df["line"] = range(2, len(df) + 2)

    problems: list[str] = []
    for row in df.itertuples():
        if not row.path:
            problems.append(f"line {row.line}: empty path")
        elif "\\" in row.path or row.path.startswith("/") or ".." in Path(row.path).parts:
            problems.append(f"line {row.line}: path {row.path!r} must be relative to images/ and use '/'")
        elif Path(row.path).suffix.lower() not in IMAGE_EXTENSIONS:
            problems.append(
                f"line {row.line}: {row.path!r} has an unsupported extension (allowed: {', '.join(IMAGE_EXTENSIONS)})"
            )
        if not row.label:
            problems.append(f"line {row.line}: empty label")
        if has_split and row.split not in SPLIT_VALUES:
            problems.append(f"line {row.line}: split value {row.split!r} is not allowed. Allowed values: train, val, test")
    dup = df[df["path"].duplicated(keep=False) & (df["path"] != "")]
    for path, group in dup.groupby("path"):
        problems.append(f"lines {', '.join(map(str, group['line']))}: path {path!r} is listed more than once")
    stems = df["path"].map(lambda p: png_name(p).lower() if p else "")
    clash = df[stems.duplicated(keep=False) & (stems != "")]
    for _, group in clash.groupby(stems[clash.index]):
        if group["path"].nunique() > 1:
            problems.append(
                f"lines {', '.join(map(str, group['line']))}: paths {', '.join(group['path'])} differ only by extension "
                "or letter case; SR4Rec stores processed images as <path without extension>.png, so they would collide"
            )
    if problems:
        raise _fail("labels.csv does not follow the required format:", problems)

    widths, heights, hashes, converted = [], [], [], 0
    for row in df.itertuples():
        f = images_dir / row.path
        if not f.is_file():
            problems.append(f"line {row.line}: file not found: images/{row.path}")
            widths.append(0), heights.append(0), hashes.append("")
            continue
        try:
            with Image.open(f) as im:
                im.load()
                w, h = im.size
                if im.mode != "RGB":
                    converted += 1
        except (UnidentifiedImageError, OSError) as err:
            problems.append(f"line {row.line}: images/{row.path} cannot be read as an image ({err})")
            widths.append(0), heights.append(0), hashes.append("")
            continue
        widths.append(w)
        heights.append(h)
        hashes.append(sha256_file(f) if hash_images else "")
    if problems:
        raise _fail("Some images listed in labels.csv cannot be used:", problems)

    df["width"] = widths
    df["height"] = heights
    df["sha256"] = hashes
    labels_sha = sha256_file(labels_csv)
    # Parsed rows, not file bytes, so that line endings or quoting of labels.csv do not matter.
    content_key = stable_key("dataset-v2", list(zip(df["path"], df["label"], df["split"], df["sha256"], strict=True)))
    classes = sorted(df["label"].unique().tolist())
    if len(classes) < 2:
        raise DatasetError(f"The dataset has {len(classes)} class; closed-set recognition needs at least 2 classes.\n{GUIDE}")
    notes = []
    if converted:
        notes.append(f"{converted:,} images were not RGB (for example grayscale) and were converted to 3-channel RGB.")
    return Dataset(
        root=root,
        table=df[["path", "label", "split", "width", "height", "sha256", "line"]].reset_index(drop=True),
        has_split_column=has_split,
        labels_sha256=labels_sha,
        content_key=content_key,
        classes=classes,
        n_converted_to_rgb=converted,
        notes=notes,
    )


def check_mode_constraints(ds: Dataset, mode: str, scale: int) -> None:
    """Checks that depend on the configured mode and scale."""
    if mode != "synthetic":
        return
    small = ds.table[(ds.table["width"] < scale) | (ds.table["height"] < scale)]
    if len(small):
        problems = [
            f"line {r.line}: images/{r.path} is {r.width} x {r.height} px, smaller than scale {scale}"
            for r in small.itertuples()
        ]
        raise _fail(
            f"In synthetic mode every image is downsampled by {scale}; these images are too small "
            "(use mode: native-lr for low-resolution images):",
            problems,
        )
