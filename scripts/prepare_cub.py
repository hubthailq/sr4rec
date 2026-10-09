"""Convert CUB-200-2011 to the SR4Rec format for demo D3 (`d3_cub`), keeping the official split.

Output: <out>/images/<class folder>/<file>.jpg and <out>/labels.csv (path,label,split with
train/test from train_test_split.txt; SR4Rec moves split.val_from_train of train to val).

Usage:
    python scripts/prepare_cub.py --cub ~/datasets/CUB_200_2011 --out data/cub
"""

import argparse
import shutil
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cub", type=Path, required=True, help="CUB_200_2011 folder (contains images.txt)")
    ap.add_argument("--out", type=Path, default=Path("data/cub"))
    a = ap.parse_args()

    def read(name, cols):
        return pd.read_csv(a.cub / name, sep=" ", header=None, names=cols, dtype=str)

    images = read("images.txt", ["id", "path"])
    split = read("train_test_split.txt", ["id", "is_train"])
    df = images.merge(split, on="id")
    df["label"] = df["path"].str.split("/").str[0]
    df["split"] = df["is_train"].map({"1": "train", "0": "test"})
    for p in df["path"]:
        dst = a.out / "images" / p
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(a.cub / "images" / p, dst)
    df[["path", "label", "split"]].to_csv(a.out / "labels.csv", index=False)
    print(f"{len(df)} images, {df['label'].nunique()} classes written to {a.out}")


if __name__ == "__main__":
    main()
