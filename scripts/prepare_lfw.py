"""Convert LFW to the SR4Rec format for demo D2 (`d2_lfw`): people with at least N images (default 20).

Output: <out>/images/<Person>/<file>.jpg (copied unchanged) and <out>/labels.csv (path,label).
The split is created by SR4Rec from the configured ratios.

Usage:
    python scripts/prepare_lfw.py --lfw ~/datasets/lfw --out data/lfw
"""

import argparse
import shutil
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lfw", type=Path, required=True, help="LFW folder with one sub-folder per person")
    ap.add_argument("--out", type=Path, default=Path("data/lfw"))
    ap.add_argument("--min-images", type=int, default=20)
    a = ap.parse_args()
    base = a.lfw / "lfw" if (a.lfw / "lfw").is_dir() else a.lfw
    rows = []
    for d in sorted(p for p in base.iterdir() if p.is_dir()):
        files = sorted(d.glob("*.jpg"))
        if len(files) < a.min_images:
            continue
        for f in files:
            rel = f"{d.name}/{f.name}"
            (a.out / "images" / d.name).mkdir(parents=True, exist_ok=True)
            shutil.copyfile(f, a.out / "images" / rel)
            rows.append({"path": rel, "label": d.name})
    pd.DataFrame(rows).to_csv(a.out / "labels.csv", index=False)
    print(f"{len(rows)} images of {len({r['label'] for r in rows})} people written to {a.out}")


if __name__ == "__main__":
    main()
