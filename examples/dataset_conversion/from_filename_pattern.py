"""Label encoded in the file name -> SR4Rec format.

Example: images/017_s1_03.jpg belongs to person 017. The label is extracted with a regular
expression that has a named group `label` (default: everything before the first underscore).

Usage:
    python examples/dataset_conversion/from_filename_pattern.py data/my_dataset
    python examples/dataset_conversion/from_filename_pattern.py data/my_dataset --pattern "^(?P<label>[a-z]+)-\\d+"
"""

import argparse
import re
from pathlib import Path

import pandas as pd

EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def convert(root: Path, pattern: str = r"^(?P<label>[^_]+)_") -> pd.DataFrame:
    rx = re.compile(pattern)
    images = root / "images"
    rows, skipped = [], []
    for p in sorted(images.rglob("*")):
        if p.suffix.lower() not in EXTS:
            continue
        m = rx.search(p.name)
        if not m:
            skipped.append(p.name)
            continue
        rows.append({"path": p.relative_to(images).as_posix(), "label": m.group("label")})
    if skipped:
        print(f"Warning: {len(skipped)} file names do not match the pattern, for example {skipped[0]}")
    df = pd.DataFrame(rows, columns=["path", "label"])
    df.to_csv(root / "labels.csv", index=False)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="dataset folder that contains images/")
    ap.add_argument("--pattern", default=r"^(?P<label>[^_]+)_", help="regular expression with a group named label")
    a = ap.parse_args()
    df = convert(a.root, a.pattern)
    print(f"Wrote labels.csv: {len(df)} images, {df['label'].nunique()} labels")
