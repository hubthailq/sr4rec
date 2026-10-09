"""Annotation file with other column names -> SR4Rec format.

Example input (annotations.csv): file,identity,subset  with subset values training/validation/testing.
The script renames the columns to path,label[,split] and maps the split values.

Usage:
    python examples/dataset_conversion/from_annotation_csv.py data/my_dataset annotations.csv \\
        --path-col file --label-col identity --split-col subset \\
        --split-map training=train validation=val testing=test
"""

import argparse
from pathlib import Path

import pandas as pd


def convert(root: Path, annotations: Path, path_col: str, label_col: str, split_col: str | None = None,
            split_map: dict | None = None) -> pd.DataFrame:
    src = pd.read_csv(annotations, dtype=str, keep_default_na=False)
    df = pd.DataFrame({"path": src[path_col].str.replace("\\", "/", regex=False), "label": src[label_col]})
    if split_col:
        df["split"] = src[split_col].map(lambda v: (split_map or {}).get(v, v))
        bad = sorted(set(df["split"]) - {"train", "val", "test"})
        if bad:
            raise SystemExit(f"Unmapped split values: {bad}. Use --split-map, for example training=train")
    df.to_csv(root / "labels.csv", index=False)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="dataset folder that contains images/")
    ap.add_argument("annotations", type=Path, help="annotation CSV file")
    ap.add_argument("--path-col", required=True)
    ap.add_argument("--label-col", required=True)
    ap.add_argument("--split-col")
    ap.add_argument("--split-map", nargs="*", default=[], help="value=split pairs, for example training=train")
    a = ap.parse_args()
    mapping = dict(x.split("=", 1) for x in a.split_map)
    df = convert(a.root, a.annotations, a.path_col, a.label_col, a.split_col, mapping)
    print(f"Wrote labels.csv: {len(df)} images, {df['label'].nunique()} labels")
