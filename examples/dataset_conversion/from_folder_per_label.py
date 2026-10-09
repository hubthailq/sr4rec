"""Folder-per-label layout -> SR4Rec format.

Input:  <root>/images/<label>/<any sub-folders>/<image>
Output: <root>/labels.csv with columns path,label (the label is the first folder under images/).

Usage:
    python examples/dataset_conversion/from_folder_per_label.py data/my_dataset
"""

import argparse
from pathlib import Path

import pandas as pd

EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def convert(root: Path) -> pd.DataFrame:
    images = root / "images"
    rows = [{"path": p.relative_to(images).as_posix(), "label": p.relative_to(images).parts[0]}
            for p in sorted(images.rglob("*")) if p.suffix.lower() in EXTS and len(p.relative_to(images).parts) > 1]
    df = pd.DataFrame(rows, columns=["path", "label"])
    df.to_csv(root / "labels.csv", index=False)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="dataset folder that contains images/")
    df = convert(ap.parse_args().root)
    print(f"Wrote labels.csv: {len(df)} images, {df['label'].nunique()} labels")
