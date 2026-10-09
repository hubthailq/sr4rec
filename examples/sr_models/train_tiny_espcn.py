"""Train the example SR model (TinyESPCN, x4) on the train split of pets_mini.

Only the rows marked train in labels.csv are used, so the model never sees the test images of the
examples that use the split column (examples 01, 04, 06). Example 05 re-splits the data and
therefore uses no SR model trained on pets_mini.
LR patches are made with the same MATLAB-style bicubic downsampling that SR4Rec
uses in synthetic mode. Runs in a few minutes on a laptop CPU.

Usage (from the repository root):
    python examples/sr_models/train_tiny_espcn.py
    python examples/sr_models/train_tiny_espcn.py --data examples/data/pets_mini --iters 3000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sr4rec.lr.resize import downsample
from sr4rec.utils import load_rgb, pil_to_tensor

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from tiny_espcn import TinyESPCN  # noqa: E402

PATCH = 64  # HR patch size (multiple of 4)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(HERE.parent / "data" / "pets_mini"), help="dataset folder")
    ap.add_argument("--out", default=str(HERE / "tiny_espcn_x4_pets.pth"), help="weights file to write")
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    root = Path(args.data)
    df = pd.read_csv(root / "labels.csv", dtype=str, keep_default_na=False)
    if "split" in df.columns:
        df = df[df["split"] == "train"]
    images = [pil_to_tensor(load_rgb(root / "images" / p)) for p in df["path"]]
    images = [im for im in images if min(im.shape[1:]) >= PATCH]
    if not images:
        sys.exit(f"No train image of at least {PATCH} px found in {root}")
    print(f"Training TinyESPCN x4 on {len(images)} train images for {args.iters} iterations")
    model = TinyESPCN()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.iters)
    for it in range(1, args.iters + 1):
        hr, lr = [], []
        for _ in range(args.batch):
            im = images[rng.integers(len(images))]
            y = int(rng.integers(0, im.shape[1] - PATCH + 1))
            x = int(rng.integers(0, im.shape[2] - PATCH + 1))
            patch = im[:, y : y + PATCH, x : x + PATCH]
            hr.append(patch)
            lr.append(downsample(patch, 4))
        hr_t, lr_t = torch.stack(hr), torch.stack(lr)
        loss = torch.nn.functional.l1_loss(model(lr_t), hr_t)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if it % 500 == 0 or it == 1:
            print(f"  iteration {it:5d}  L1 {loss.item():.4f}")
    torch.save(model.state_dict(), args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
