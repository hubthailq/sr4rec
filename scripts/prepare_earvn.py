"""Convert EarVN1.0 to the SR4Rec format for demo D1 (`d1_earvn`; native-lr, all images at their original size).

Input: one sub-folder per identity (for example Images/001.NAME/...). The label is the part of the
folder name before the first dot. Output: <out>/images/<label>/<file> and <out>/labels.csv
(path,label); SR4Rec creates the split from the configured ratios.

Usage:
    python scripts/prepare_earvn.py --earvn ~/datasets/EarVN1.0 --out data/earvn
"""

import argparse
import shutil
from pathlib import Path

import pandas as pd

EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--earvn", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("data/earvn"))
    a = ap.parse_args()
    base = a.earvn / "Images" if (a.earvn / "Images").is_dir() else a.earvn
    rows = []
    for d in sorted(p for p in base.iterdir() if p.is_dir()):
        label = d.name.split(".")[0]
        for f in sorted(p for p in d.rglob("*") if p.suffix.lower() in EXTS):
            rel = f"{label}/{f.relative_to(d).as_posix()}"
            dst = a.out / "images" / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(f, dst)
            rows.append({"path": rel, "label": label})
    pd.DataFrame(rows).to_csv(a.out / "labels.csv", index=False)
    print(f"{len(rows)} images of {len({r['label'] for r in rows})} identities written to {a.out}")


if __name__ == "__main__":
    main()
