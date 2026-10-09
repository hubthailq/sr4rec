"""Write SR images with the names SR4Rec expects (method 3), using PIL Lanczos as a stand-in for
an external SR model. Replace `upscale()` with your own model or tool.

The LR inputs are the images SR4Rec feeds to SR models:
  * synthetic mode: runs/<run_name>/lr/<split>/<path>.png (written by every synthetic run);
  * native-lr mode: the dataset images themselves (data/<dataset>/images/<path>).
Each output is <out>/<path with the extension replaced by .png>, exactly `scale` times the LR size.
With protocol matched, SR4Rec needs the train, val and test images; with fixed_recognizer only test.

Usage (from the repository root):
    python examples/sr_images/make_sr_images.py --run runs/<run_name> --out examples/sr_images/lanczos
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml
from PIL import Image

from sr4rec.utils import load_rgb


def upscale(im: Image.Image, scale: int) -> Image.Image:
    return im.resize((im.width * scale, im.height * scale), Image.LANCZOS)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="a finished SR4Rec run folder (contains split.csv and sr4rec.yaml)")
    ap.add_argument("--out", required=True, help="output folder, for example sr_images/<method_name>")
    ap.add_argument("--splits", default="train,val,test", help="comma-separated splits (default: all three)")
    args = ap.parse_args()

    run = Path(args.run)
    cfg = yaml.safe_load((run / "sr4rec.yaml").read_text(encoding="utf-8"))
    split = pd.read_csv(run / "split.csv", dtype=str, keep_default_na=False)
    scale = int(cfg["scale"])
    dataset = (run / cfg["dataset"]).resolve()
    out = Path(args.out)
    wanted = set(args.splits.split(","))
    n = 0
    for row in split.itertuples():
        if row.split not in wanted:
            continue
        rel_png = Path(row.path).with_suffix(".png")
        src = run / "lr" / row.split / rel_png if cfg["mode"] == "synthetic" else dataset / "images" / row.path
        dst = out / rel_png
        dst.parent.mkdir(parents=True, exist_ok=True)
        upscale(load_rgb(src), scale).save(dst)
        n += 1
    print(f"Wrote {n} SR images (x{scale}) to {out}")


if __name__ == "__main__":
    main()
